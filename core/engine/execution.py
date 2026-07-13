"""Unified execution engine — merges pipeline chain + single-agent paths.

Layer 5. Replaces PipelineEngine.execute_chain + execute_single_agent.
"""
from __future__ import annotations

import time, traceback, sys
from dataclasses import dataclass, field
from typing import Any, Callable

from core.agent import MiniAgent
from core.codec.owui import extract_text, parse_history_format
from core.infra.templates import THINKING_PRESETS, resolve_template, normalize_thinking
from core.infra.settings import load_settings, load_routes, get_route, get_agent_workspace, collect_done_tasks
from core.http_utils import preprocess_base_url
from core.infra.logger import log_request_start, log_request_end, trace_log
from core.engine.pipeline import PipelineEngine
from core.adapter.owui import rebuild_conversation, make_on_event
from core.store.ua import save as ua_save


@dataclass
class ExecRequest:
    session_id:    str
    user_text:     str
    messages:      list[dict]
    route:         dict | None = None
    agent_model:   str = ""
    chain_cfg:     dict | None = None
    variables:     dict = field(default_factory=dict)
    separator:     str = ""
    debug_ctx:     Any | None = None
    stop_event:    Any | None = None
    stop_reason:   str = ""


@dataclass
class ExecResult:
    text:          str
    blocks:        list[dict]
    usage:         dict
    phase:         str = "main"


# ── Public entry point ────────────────────────────────────

def execute(req: ExecRequest,
            emit_cb: Callable[[str, str], None],
            phase_set_cb: Callable[[str], None] | None = None) -> ExecResult:
    """Unified execution. Internally dispatches to side-path or single path."""
    try:
        side_paths = (req.chain_cfg or {}).get("side_paths", [])
        if side_paths:
            return _run_chain(req, side_paths, emit_cb, phase_set_cb)
        else:
            return _run_single(req, emit_cb, phase_set_cb)
    except RuntimeError as e:
        traceback.print_exc(file=sys.stderr)
        try:
            emit_cb("main", f"\n\n❌ {e}\n")
        except Exception:
            pass
        return ExecResult(text="", blocks=[], usage={}, phase="main")


# ── Agent construction ─────────────────────────────────────

def _prepare_agent(req: ExecRequest, main_cfg: dict | None = None) -> MiniAgent:
    """Build and configure a MiniAgent from route/chain/fallback."""
    s = load_settings()
    chain_vars = req.variables
    model = req.agent_model
    route = req.route

    if main_cfg:
        if not route:
            routes = load_routes()
            route = get_route("") or (routes[0] if routes else {})
        thinking_val = main_cfg.get("thinking") or normalize_thinking(main_cfg.get("thinking_budget")) or ""
        agent = PipelineEngine.build_agent_from_route(route, model, thinking=thinking_val)
        sp_template = main_cfg.get("system_prompt", "")
        extra = {"user_text": req.user_text, "side_text": ""}
        if sp_template:
            sp_resolved = resolve_template(sp_template, chain_vars, extra)
            if sp_resolved.strip():
                agent.set_system_prompt(sp_resolved)
    elif route:
        agent = PipelineEngine.build_agent_from_route(route, model)
        sp = s.get("system", {}).get("system_prompt", "").strip()
        if sp:
            agent.set_system_prompt(sp)
    else:
        # Fallback: no route, no chain — build directly
        routes = load_routes()
        if not routes:
            raise RuntimeError(
                "No routes configured. Add a route in Admin → Routes "
                "(API key + base URL + model) before sending messages.")
        r0 = routes[0]
        api_key = r0.get("api_key", "").strip()
        base_url = preprocess_base_url(r0.get("base_url", ""))
        thinking_budget = THINKING_PRESETS.get(r0.get("thinking", "off"), 0)
        max_tokens = 128000 if r0.get("context_1m") else 32000
        proxy_config = s.get("proxy", {}) if isinstance(s.get("proxy"), dict) else {"address": s.get("proxy", "")}
        proxy_url = proxy_config.get("address", "")

        agent = MiniAgent(
            workspace=get_agent_workspace(model=model), model=model,
            max_iterations=r0.get("max_iterations", 20), thinking_budget=thinking_budget,
            max_tokens=max_tokens, api_key=api_key, base_url=base_url,
            proxy_url=proxy_url, allow_bash=True,
            vision_config=s.get("vision", {}),
            protocol=r0.get("protocol", "anthropic"),
            custom_md_text=s.get("system", {}).get("custom_md", ""),
        )
        agent.set_provider_config(s)
        agent.set_proxy_config(proxy_config if r0.get("use_proxy", True) else {**proxy_config, "use_for_api": False})
        sp = s.get("system", {}).get("system_prompt", "").strip()
        if sp:
            agent.set_system_prompt(sp)
    # Apply read_root from settings (workspace must be within read_root)
    rr = s.get("read_root", "").strip()
    if rr:
        from pathlib import Path as _P
        from core.paths import WORKSPACE as _WS
        rp = _P(rr).resolve()
        try:
            _WS.resolve().relative_to(rp)
            agent._read_root = rp
        except ValueError:
            pass
    return agent


# ── Single-agent path (0 side paths) ───────────────────────

def _run_single(req: ExecRequest, emit_cb, phase_set_cb=None) -> ExecResult:
    chain = req.chain_cfg or {}
    main_cfg = chain.get("main_path") if chain else None
    chain_vars = chain.get("variables", {}) if chain else {}

    # Resolve actual user text from injection template
    actual_user_text = req.user_text
    if main_cfg:
        extra = {"user_text": req.user_text, "side_text": ""}
        inj_template = main_cfg.get("user_injection", "{user_text}")
        actual_user_text = resolve_template(inj_template, chain_vars, extra)

    # History: format parsing
    messages = list(req.messages)
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = extract_text(m.get("content"))
            break
    parsed = parse_history_format(last_user_msg)
    if parsed:
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        messages = [sys_msg] + parsed if sys_msg else parsed
        for m in reversed(messages):
            if m.get("role") == "user":
                actual_user_text = extract_text(m.get("content"))
                break

    # Build agent
    agent = _prepare_agent(req, main_cfg)
    keep_think = (main_cfg or {}).get("keep_think", -1)
    separator = req.separator or chain.get("separator", "")

    # Rebuild conversation
    t0 = time.time()
    conv, turn, ua_hits, ua_total, _ = rebuild_conversation(
        messages, separator=separator, keep_think=keep_think)
    agent.conversation = list(conv)
    agent.set_turn(turn)
    agent.clean_orphaned_tool_results()

    if req.debug_ctx:
        agent.debug_ctx = req.debug_ctx
        req.debug_ctx.record_ua(ua_hits, ua_total)

    phase = "main" if chain else "main"
    route_label = (req.route or {}).get("label", "")
    log_entry = log_request_start(req.session_id, chain.get("label", ""), route_label, req.agent_model)
    t_start = time.time()

    if main_cfg and main_cfg.get("_show_history_stats", False):
        history_chars = sum(len(extract_text(m.get("content", ""))) for m in messages if m.get("role") in ("user", "assistant"))
        emit_cb(phase, f"<think>\n轮数: {agent.turn}  字符数: {history_chars}\n</think>\n")

    chain_keep = (main_cfg or {}).get("keep_turns", -1)
    if chain_keep >= 0:
        PipelineEngine.clean_conversation(agent, chain_keep)

    _rebuild_ms = int((time.time() - t0) * 1000)
    _conv_chars = sum(len(str(m.get("content", ""))) for m in agent.conversation)
    trace_log(
        f"REBUILD: {_rebuild_ms}ms {len(agent.conversation)}msgs {_conv_chars}chars "
        f"keep_turns={chain_keep} keep_think={keep_think}")

    tool_render = load_settings().get("tool_render", "think")
    on_event, _get_state = make_on_event(req.session_id, phase, emit_cb, log_entry,
                                          tool_render=tool_render, model=req.agent_model or "")
    think_open = False

    done_output = collect_done_tasks(agent.workspace)
    if done_output:
        actual_user_text = f"[后台任务完成]\n{done_output}\n\n{actual_user_text}"

    if req.stop_event:
        agent.set_stop_event(req.stop_event)
    pre_usage = dict(agent.token_usage)
    _hm_t0 = time.time()
    try:
        agent.handle_message(actual_user_text, on_event=on_event)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        try:
            emit_cb(phase, "\n\n❌ Internal error occurred. Please try again.\n")
        except Exception:
            pass
    finally:
        _hm_ms = int((time.time() - _hm_t0) * 1000)
        trace_log(
            f"HANDLE_MSG: {_hm_ms}ms "
            f"conv_grew={len(agent.conversation) - len(messages)} "
            f"tokens_in={agent.token_usage['in']-pre_usage['in']}")
        think_open, _, _, _owui_text = _get_state()
        if think_open:
            try:
                emit_cb(phase, "</think>\n\n")
            except Exception:
                pass
        if agent.is_stopped():
            agent.cleanup_after_stop()

    raw_blocks = agent.last_raw_blocks
    if raw_blocks:
        if _owui_text:
            ua_save(_owui_text.rstrip("\n"), raw_blocks)
        else:
            texts = []
            for rb in raw_blocks:
                if rb.get("role") == "assistant":
                    c = rb.get("content", [])
                    if isinstance(c, list):
                        t = "".join(b.get("text", "") for b in c if b.get("type") == "text")
                        if t:
                            texts.append(t)
            all_a_text = "\n".join(texts)
            if all_a_text:
                ua_save(all_a_text, raw_blocks)

    dur = int((time.time() - t_start) * 1000)
    tokens = {
        "input": agent.token_usage["in"] - pre_usage["in"],
        "output": agent.token_usage["out"] - pre_usage["out"],
        "cache_read": agent.token_usage["cache_read"] - pre_usage["cache_read"],
        "cache_write": agent.token_usage["cache_write"] - pre_usage["cache_write"],
    }
    status = "stopped" if agent.is_stopped() else "ok"
    exit_reason = req.stop_reason or ""
    if agent.is_stopped() and not exit_reason:
        exit_reason = "timeout" if dur > 300_000 else "user_stop"

    if req.debug_ctx:
        req.debug_ctx.flush()
    log_request_end(log_entry, status, tokens, dur,
                     exit_reason=exit_reason, has_output=bool(agent.get_last_assistant_text()))
    return ExecResult(
        text=agent.get_last_assistant_text(),
        blocks=raw_blocks or [],
        usage=tokens,
        phase=phase,
    )


# ── Chain path (with side paths) ───────────────────────────

def _run_chain(req: ExecRequest, side_paths: list, emit_cb, phase_set_cb) -> ExecResult:
    """Execute chain with side paths. Delegates complex side logic to PipelineEngine."""
    chain = req.chain_cfg or {}
    # Delegate to existing PipelineEngine for the side-path heavy lifting
    result = PipelineEngine.execute_chain(
        chain, req.session_id, req.user_text, req.messages,
        emit_cb, phase_set_cb or (lambda p: None),
        req.stop_event, req.stop_reason, req.debug_ctx,
    )
    return ExecResult(
        text=result.get("assistant_text", ""),
        blocks=result.get("raw_blocks", []),
        usage=result.get("tokens", {}),
        phase="main",
    )
