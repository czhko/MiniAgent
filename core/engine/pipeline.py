"""v0.4 Pipeline engine — chain execution, single-agent routing, OWUI message parsing."""
from __future__ import annotations

import re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from core.agent import MiniAgent
from core.codec.owui import extract_text, parse_history_format
from core.infra.templates import THINKING_PRESETS, resolve_template, normalize_thinking
from core.infra.settings import load_settings, load_routes, load_chains, get_route, get_agent_workspace, collect_done_tasks
from core.http_utils import preprocess_base_url
from core.timeutil import bj_now, bj_epoch
from core.infra.logger import log_request_start, log_round, log_request_end, trace_model, trace_log
from core.store.ua import save as ua_save, load as ua_load


def apply_owui_text(raw_blocks: list[dict], owui_text: str) -> list[dict]:
    """Overwrite text blocks in raw_blocks with OWUI text.
    If OWUI text length matches saved fragment boundaries → split & assign per fragment.
    Otherwise → user edited → replace all text blocks with OWUI text.
    """
    # Extract text fragment lengths (same logic as ua_save joins by \n)
    lens = []
    for rb in raw_blocks:
        if rb.get("role") == "assistant":
            c = rb.get("content", [])
            if isinstance(c, list):
                t = "".join(b.get("text", "") for b in c if b.get("type") == "text")
                if t:
                    lens.append(len(t))

    expected = sum(lens) + len(lens) - 1 if lens else -1

    if lens and len(owui_text) == expected:
        # Lengths match — split OWUI by stored fragment lengths
        pos = 0
        fi = 0
        for rb in raw_blocks:
            if rb.get("role") == "assistant" and fi < len(lens):
                c = rb.get("content", [])
                if isinstance(c, list):
                    frag = owui_text[pos:pos + lens[fi]]
                    for blk in c:
                        if blk.get("type") == "text":
                            blk["text"] = frag
                    pos += lens[fi] + 1
                    fi += 1
    else:
        # Length mismatch — separator trimmed or user edited.
        # Replace only the LAST text block, keep earlier ones intact.
        replaced = False
        for rb in reversed(raw_blocks):
            if rb.get("role") == "assistant":
                c = rb.get("content", [])
                if isinstance(c, list):
                    for blk in reversed(c):
                        if blk.get("type") == "text":
                            blk["text"] = owui_text
                            replaced = True
                            break
                if replaced:
                    break
        if not replaced:
            for rb in reversed(raw_blocks):
                if rb.get("role") == "assistant":
                    c = rb.get("content", [])
                    if isinstance(c, list):
                        c.append({"type": "text", "text": owui_text})
                        break

    return raw_blocks


def _inject_side_cache(agent, cached_blocks: list[dict], keep_think: int) -> None:
    """Inject cached side-path tool context into agent conversation.

    keep_think: -1=all, 0=none (handled by caller), N=last N turns.
    Groups blocks into turns (assistant + its tool_results), keeps last N turns.
    """
    if keep_think < 0:
        agent.conversation.extend(cached_blocks)
        return
    # Group into turns: each turn = assistant + subsequent tool_results
    turns: list[list[dict]] = []
    current: list[dict] = []
    for b in cached_blocks:
        if b.get("role") == "assistant" and current:
            turns.append(current)
            current = []
        current.append(b)
    if current:
        turns.append(current)
    # Keep last N complete turns
    for turn in turns[-keep_think:]:
        agent.conversation.extend(turn)


# ── Pipeline Engine ─────────────────────────────────────

class PipelineEngine:
    """Stateless pipeline execution engine."""

    @staticmethod
    def resolve_chain(model_name: str) -> tuple[dict | None, str]:
        """Match model_name against chain labels. Returns (chain_config, session_id)."""
        chains = load_chains()
        parts = model_name.split(":", 1)
        label = parts[0].strip()
        sid = parts[1].strip() if len(parts) > 1 else "default"
        for c in chains:
            if not c.get("enabled", True):
                continue
            if c.get("label") == label or c.get("display_name") == label:
                return c, sid
        return None, model_name

    @staticmethod
    def resolve_single_agent(model_name: str) -> tuple[dict | None, str]:
        """Parse {provider}/{model}-agent format. Returns (route_config, actual_model)."""
        if "/" in model_name:
            provider_name, rest = model_name.split("/", 1)
            actual_model = rest.rsplit("-agent", 1)[0].strip() if rest.endswith("-agent") else rest.strip()
            route = get_route(provider_name)
            return route, actual_model
        return None, model_name

    @staticmethod
    def build_agent_from_route(route: dict, model: str, system_prompt: str = "",
                               thinking: str = "",
                               extra_kwargs: dict | None = None,
                               chain_label: str = "") -> MiniAgent:
        """Create a MiniAgent from a route card configuration.
        thinking: preset name (off/low/high/max) or "" to inherit from route.
        chain_label: if set, workspace = workspace/{chain_label} for chain isolation."""
        s = load_settings()
        api_key = route.get("api_key", "").strip()
        base_url = preprocess_base_url(route.get("base_url", ""))
        protocol = route.get("protocol", "anthropic")

        # Thinking: explicit preset > route preset > off (0)
        thinking_preset = thinking or route.get("thinking", "off")
        thinking_budget = THINKING_PRESETS.get(thinking_preset, 0)

        # max_tokens: context_1m → 128K, else default
        max_tokens = 32000
        if route.get("context_1m"):
            max_tokens = 128000
        if extra_kwargs and "max_tokens" in extra_kwargs:
            max_tokens = extra_kwargs["max_tokens"]

        proxy_config = s.get("proxy", {}) if isinstance(s.get("proxy"), dict) else {"address": s.get("proxy", "")}
        # Per-route proxy override: if route has use_proxy=False, disable API proxy for this route
        route_proxy_config = dict(proxy_config)
        if not route.get("use_proxy", True):
            route_proxy_config["use_for_api"] = False
        proxy_url = route_proxy_config.get("address", "")

        agent = MiniAgent(
            workspace=get_agent_workspace(model=model, chain_label=chain_label),
            model=model,
            max_iterations=route.get("max_iterations", 20),
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            proxy_url=proxy_url,
            allow_bash=True,
            vision_config=s.get("vision", {}),
            protocol=protocol,
            custom_md_text=s.get("system", {}).get("custom_md", ""),
        )
        agent.set_provider_config(s)
        agent.set_proxy_config(route_proxy_config)

        if system_prompt:
            agent.set_system_prompt(system_prompt)

        return agent

    @staticmethod
    def clean_conversation(agent: MiniAgent, keep_turns: int = -1) -> list[dict]:
        """-1 = keep all, 0 = clear all, N = keep last N turns."""
        if keep_turns == 0:
            agent.clear_all()
        elif keep_turns > 0:
            agent.clear_conversation(keep_user_messages=keep_turns)
        return agent.conversation

    # [实验性] Parallel side path execution
    @staticmethod
    def _execute_side_paths_parallel(
        chain: dict, side_paths: list, shared_conv: list, turn: int,
        variables: dict, extra: dict, emit_cb, stop_event, debug_ctx,
        all_ua_sides: dict[str, list] | None = None,
    ) -> dict:
        """Run all side paths in parallel via ThreadPoolExecutor.

        Returns {"side_texts": [...], "tokens": {...}, "side_blocks": {...}}.
        Each side path emits SSE via emit_cb (wfile_lock is shared → atomic chunks).
        All side output wrapped in one <think> block with unified label.
        side_msg uses only user_text (sides are independent, no sequential dependency).
        """
        n = len(side_paths)
        side_texts: list[str] = [""] * n
        tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        side_blocks_dict: dict[str, list] = {}

        # Open unified think block
        emit_cb("main", "<think>\n")
        emit_cb("main", "=== [旁路] ===\n")

        def _run_one(idx: int, side_cfg: dict) -> tuple[int, str, dict]:
            route = get_route(side_cfg.get("route", ""))
            side_mock = side_cfg.get("mock_content", "").strip()
            if not route and not side_mock:
                return idx, "", {}
            if side_mock:
                mock_resolved = resolve_template(side_mock, variables, extra)
                text_lines = [line for line in mock_resolved.split("\n")
                              if line.strip() and not (line.strip().startswith("<think>") and line.strip().endswith("</think>"))]
                for line in mock_resolved.split("\n"):
                    lc = line.strip()
                    if lc:
                        emit_cb("main", lc + "\n")
                return idx, text_lines[-1] if text_lines else "", {}
            if not route:
                return idx, "", {}
            side_type = side_cfg.get("type", "agent")
            sp_template = side_cfg.get("system_prompt", "")
            side_sp = resolve_template(sp_template, variables, {"user_text": extra.get("user_text", ""), "side_text": ""})
            thinking_val = side_cfg.get("thinking") or normalize_thinking(side_cfg.get("thinking_budget"))
            side_model = side_cfg.get("model", "") or "deepseek-v4-flash"
            agent = PipelineEngine.build_agent_from_route(
                route, side_model, system_prompt=side_sp, thinking=thinking_val or "",
                chain_label=chain.get("label", ""))
            if debug_ctx:
                agent.debug_ctx = debug_ctx
            if side_type == "llm":
                agent.clear_tools()
                agent.max_iterations = 1
            side_conv = []
            for m in shared_conv:
                c = m.get("content", [])
                if isinstance(c, list):
                    text_only = [b for b in c if b.get("type") == "text"]
                    if text_only:
                        side_conv.append({**m, "content": text_only})
                elif isinstance(m.get("content"), str) and m.get("content", "").strip():
                    side_conv.append(m)
            agent.set_conversation(side_conv, turn)
            keep_turns_val = side_cfg.get("keep_turns", chain.get("keep_turns", -1))
            PipelineEngine.clean_conversation(agent, keep_turns_val)
            inj_template = side_cfg.get("user_injection", "{user_text}")
            side_msg = resolve_template(inj_template, variables, {"user_text": extra.get("user_text", ""), "side_text": ""})
            chain_label = chain.get("label", "")
            side_label_par = side_cfg.get("label") or f"旁路{idx}"
            side_keep_think = side_cfg.get("keep_think", -1)
            if side_keep_think != 0 and all_ua_sides:
                cached = all_ua_sides.get(side_label_par)
                if cached:
                    _inject_side_cache(agent, cached, side_keep_think)
            if stop_event:
                agent.set_stop_event(stop_event)
            pre_usage = dict(agent.token_usage)
            side_buf: list[str] = []
            def _on_side(etype: str, data: dict):
                if etype == "text":
                    side_buf.append(data.get("delta", ""))
                    emit_cb("main", data.get("delta", ""))
                elif etype in ("thinking", "tool_use", "tool_result"):
                    emit_cb("main", f"[{side_label_par}] {etype}\n")
            try:
                agent.handle_message(side_msg, on_event=_on_side)
            except Exception:
                pass
            side_text = "".join(side_buf)
            side_raw = agent.last_raw_blocks
            usage = {
                "input": agent.token_usage["in"] - pre_usage["in"],
                "output": agent.token_usage["out"] - pre_usage["out"],
                "cache_read": agent.token_usage["cache_read"] - pre_usage["cache_read"],
                "cache_write": agent.token_usage["cache_write"] - pre_usage["cache_write"],
            }
            return idx, side_text, usage, side_raw, side_label_par

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_run_one, idx, cfg): idx for idx, cfg in enumerate(side_paths)}
            for f in as_completed(futures, timeout=120):
                try:
                    idx, side_text, usage, side_raw, side_lbl = f.result()
                    side_texts[idx] = side_text
                    for k in tokens:
                        tokens[k] += usage.get(k, 0)
                    if side_raw:
                        side_blocks_dict[side_lbl] = side_raw
                except Exception:
                    idx = futures[f]
                    emit_cb("main", f"⚠ 旁路{idx}超时/异常\n")

        emit_cb("main", "\n</think>\n")
        return {"side_texts": side_texts, "tokens": tokens, "side_blocks": side_blocks_dict}

    @staticmethod
    def build_model_list() -> dict:
        """Build OWUI /v1/models response."""
        all_models: list[dict] = []

        # Chains as models
        for chain in load_chains():
            if chain.get("enabled", True):
                all_models.append({
                    "id": chain.get("display_name") or chain["label"],
                    "object": "model", "owned_by": "da",
                })

        # Route models
        for route in load_routes():
            if not route.get("enabled", True):
                continue
            api_key = route.get("api_key", "")
            if not api_key:
                continue
            route_label = route["label"]
            selected = route.get("selected_models", [])
            if selected:
                for mid in selected:
                    all_models.append({
                        "id": f"{route_label}/{mid}-agent", "object": "model",
                        "owned_by": route_label,
                    })

        if not all_models:
            all_models = [
                {"id": "roleplay", "object": "model", "created": int(bj_epoch()), "owned_by": "da"},
            ]
        return {"object": "list", "data": all_models}

    @staticmethod
    def execute_chain(
        chain: dict, session_id: str, user_text: str,
        messages: list[dict],
        emit_cb: Callable[[str, str], None],
        phase_set_cb: Callable[[str], None],
        stop_event: Any,
        stop_reason: str = "",
        debug_ctx=None,
    ) -> dict:
        """Execute a full pipeline chain (side paths + main path) via SSE."""
        if debug_ctx:
            debug_ctx.start_turn()
            debug_ctx.record_owui({"chain": chain.get("label", ""), "model": chain.get("main_path", {}).get("model", "")}, messages)

        variables = chain.get("variables", {})
        keep_think = chain.get("keep_think", -1)
        total_tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

        # Normalize OWUI History: format → structured messages
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
                    user_text = extract_text(m.get("content"))
                    break

        # ── Rebuild MAIN conversation from OWUI messages ──
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user_idx = i

        shared_conv: list[dict] = []
        turn = 0
        ua_hits = 0
        ua_total = 0
        all_ua_sides: dict[str, list] = {}
        total_user_turns = sum(1 for i, m in enumerate(messages)
                               if m.get("role") == "user" and i != last_user_idx)
        for i, m in enumerate(messages):
            if i == last_user_idx:
                continue
            role = m.get("role", "")
            content = extract_text(m.get("content"))
            if role == "user" and content.strip():
                turn += 1
                shared_conv.append({
                    "role": "user", "content": content, "turn": turn,
                    "turn_time": bj_now().isoformat(),
                })
            elif role == "assistant":
                text = content or "(thinking)"
                # Try to restore raw blocks from UA store BEFORE separator trimming.
                # UA store key = hash of full text; separator strips the text → hash mismatch.
                _do_restore = keep_think < 0 or turn > total_user_turns - keep_think
                if _do_restore:
                    ua_total += 1
                    raw_blocks, ua_sides = ua_load(text)
                    if raw_blocks:
                        ua_hits += 1
                        if ua_sides:
                            for s_label, s_blocks in ua_sides.items():
                                all_ua_sides.setdefault(s_label, []).extend(s_blocks)
                else:
                    raw_blocks = None
                # Separator trim (after UA lookup to preserve hash match)
                sep = chain.get("separator", "")
                if sep:
                    idx = text.rfind(sep)
                    if idx != -1:
                        trimmed = text[idx + len(sep):].strip()
                        if trimmed:
                            text = trimmed
                if raw_blocks:
                    apply_owui_text(raw_blocks, text)
                    for rb in raw_blocks:
                        if isinstance(rb, dict) and rb.get("role") in ("assistant", "user"):
                            if "turn" not in rb:
                                rb["turn"] = turn
                            shared_conv.append(rb)
                else:
                    shared_conv.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                        "turn": turn,
                    })

        # ── Emit history stats (before any side/main output) ──
        if debug_ctx:
            debug_ctx.record_ua(ua_hits, ua_total)
        if chain.get("show_history_stats", True):
            history_chars = sum(len(extract_text(m.get("content", ""))) for m in shared_conv)
            emit_cb("main", f"<think>\n轮数: {turn}  字符数: {history_chars}\n</think>\n")

        # ── Execute side paths ──
        side_texts: list[str] = []
        extra = {"user_text": user_text, "side_text": ""}
        log_entry = log_request_start(session_id, chain.get("label", ""),
                                      chain.get("main_path", {}).get("route", ""),
                                      chain.get("main_path", {}).get("model", ""))
        t_start = bj_epoch()
        all_debug_blocks: list[dict] = []

        side_paths = chain.get("side_paths", [])
        all_side_blocks: dict[str, list] = {}
        if chain.get("parallel_side") and len(side_paths) > 1:
            # [实验性] Parallel side path execution
            _result = PipelineEngine._execute_side_paths_parallel(
                chain, side_paths, shared_conv, turn, variables, extra,
                emit_cb, stop_event, debug_ctx, all_ua_sides)
            side_texts = _result["side_texts"]
            for k in total_tokens:
                total_tokens[k] += _result["tokens"][k]
            # Merge side blocks from parallel execution
            for s_lbl, s_blocks in _result.get("side_blocks", {}).items():
                all_side_blocks[s_lbl] = s_blocks
            # Skip serial loop
            side_paths = []
        else:
            extra["side_text"] = ""
            for i, st in enumerate(side_texts):
                extra[f"side{i}"] = ""

        for idx, side_cfg in enumerate(side_paths):
            if stop_event and stop_event.is_set():
                break
            phase_set_cb("side")
            route = get_route(side_cfg.get("route", ""))
            side_mock = side_cfg.get("mock_content", "").strip()

            if side_mock:
                extra["side_text"] = "\n\n".join(side_texts)
                for i, st in enumerate(side_texts):
                    extra[f"side{i}"] = st
                mock_resolved = resolve_template(side_mock, variables, extra)
                # Each line in mock is its own event → own <think> block
                # Label goes inside the first block
                first = True
                for line in mock_resolved.split("\n"):
                    lc = line.strip()
                    if not lc:
                        continue
                    is_think = lc.startswith("<think>") and lc.endswith("</think>")
                    content = re.sub(r'</?think>', '', lc).strip() if is_think else lc
                    emit_cb("main", "<think>\n")
                    if first:
                        emit_cb("main", f"=== [{side_cfg.get('label') or f'旁路{idx}'}] ===\n")
                        first = False
                    emit_cb("main", content + "\n")
                    emit_cb("main", "</think>\n")
                # {sideN} = final text only (match DA side.handle_message() return)
                text_lines = [
                    line for line in mock_resolved.split("\n")
                    if line.strip() and not (line.strip().startswith("<think>") and line.strip().endswith("</think>"))
                ]
                side_texts.append(text_lines[-1] if text_lines else "")
                continue

            if not route:
                continue

            side_type = side_cfg.get("type", "agent")
            sp_template = side_cfg.get("system_prompt", "")
            extra["side_text"] = "\n\n".join(side_texts)
            for i, st in enumerate(side_texts):
                extra[f"side{i}"] = st
            side_sp = resolve_template(sp_template, variables, extra)

            thinking_val = side_cfg.get("thinking") or normalize_thinking(side_cfg.get("thinking_budget"))
            # Guard against empty model (same as main path line 559)
            side_model = side_cfg.get("model", "") or "deepseek-v4-flash"
            agent = PipelineEngine.build_agent_from_route(
                route, side_model, system_prompt=side_sp,
                thinking=thinking_val or "",
                chain_label=chain.get("label", ""),
            )
            if debug_ctx:
                agent.debug_ctx = debug_ctx

            # For llm type: strip tools, single iteration
            if side_type == "llm":
                agent.clear_tools()
                agent.max_iterations = 1

            # Side-only conversation: keep only text blocks, no main path's thinking/tools
            side_conv = []
            for m in shared_conv:
                c = m.get("content", [])
                if isinstance(c, list):
                    text_only = [b for b in c if b.get("type") == "text"]
                    if text_only:
                        side_conv.append({**m, "content": text_only})
                elif isinstance(m.get("content"), str) and m.get("content", "").strip():
                    side_conv.append(m)
            agent.set_conversation(side_conv, turn)

            # Apply context cleaning: keep last N turns (use side_cfg's own keep_turns)
            keep_turns = side_cfg.get("keep_turns", chain.get("keep_turns", -1))
            PipelineEngine.clean_conversation(agent, keep_turns)

            # Resolve user injection
            inj_template = side_cfg.get("user_injection", "{user_text}")
            side_msg = resolve_template(inj_template, variables, extra)

            # Restore side path tool context from UA store (unified with main path hash)
            chain_label = chain.get("label", "")
            side_keep_think = side_cfg.get("keep_think", -1)
            side_label = side_cfg.get("label") or f"旁路{idx}"
            if side_keep_think != 0:
                cached = all_ua_sides.get(side_label)
                if debug_ctx:
                    debug_ctx.record_side_cache(side_label, cached is not None)
                if cached:
                    _inject_side_cache(agent, cached, side_keep_think)

            # Side path: each event type in its own <think> block (match user diagram)
            output_cfg = side_cfg.get("context_output", {})
            side_blocks: list[dict] = []
            side_buf: list[str] = []
            side_block_open = False
            side_label_done = False
            side_last_type = ""

            def _side_open_block(etype: str):
                nonlocal side_block_open, side_label_done, side_last_type
                if side_block_open and etype != side_last_type:
                    emit_cb("main", "</think>\n")
                    side_block_open = False
                if not side_block_open:
                    emit_cb("main", "<think>\n")
                    if not side_label_done:
                        emit_cb("main", f"=== [{side_cfg.get('label') or f'旁路{idx}'}] ===\n")
                        side_label_done = True
                    side_block_open = True
                side_last_type = etype

            def _side_close_block():
                nonlocal side_block_open
                if side_block_open:
                    emit_cb("main", "</think>\n")
                    side_block_open = False

            def on_side_event(etype: str, data: dict):
                nonlocal side_last_type
                trace_model(session_id, f"side_{idx}", etype, data)


                if etype == "thinking":
                    side_blocks.append({"type": "thinking", "thinking": data.get("delta", ""), "source": "model"})
                    if output_cfg.get("think", True):
                        _side_open_block("thinking")
                        emit_cb("main", data.get("delta", ""))
                elif etype == "text":
                    side_blocks.append({"type": "text", "text": data.get("delta", ""), "source": "model"})
                    side_buf.append(data.get("delta", ""))
                    if output_cfg.get("text", True):
                        _side_open_block("text")
                        emit_cb("main", data.get("delta", ""))
                elif etype == "tool_use":
                    side_blocks.append({"type": "tool_use", "name": data.get("name","?"), "input": data.get("input",{}), "source": "model"})
                    if output_cfg.get("tool_use", True):
                        _side_open_block("tool_use")
                        emit_cb("main", f"\U0001f527 {data.get('name','?')} {str(data.get('input',{}))[:200]}\n")
                        _side_close_block()
                elif etype == "tool_result":
                    c = str(data.get("content", ""))[:300]
                    is_err = data.get("is_error", False)
                    side_blocks.append({"type": "tool_result", "content": c, "is_error": is_err, "source": "model"})
                    if output_cfg.get("tool_result", True):
                        _side_open_block("tool_result")
                        prefix = "❌" if is_err else "\U0001f4cb"
                        emit_cb("main", f"{prefix} {data.get('name','?')} {c}\n")
                        _side_close_block()

            # Wire external stop signal to agent
            if stop_event:
                agent.set_stop_event(stop_event)
            pre_usage = dict(agent.token_usage)
            try:
                agent.handle_message(side_msg, on_event=on_side_event)
            except Exception:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                err_msg = traceback.format_exc()
                trace_log(
                    f"SIDE_ERR chain={chain_label} side={side_label}: {err_msg[-500:]}")
                try:
                    emit_cb("main", f"\n❌ Error in {side_cfg.get('label') or f'旁路{idx}'}: internal error.\n")
                except Exception:
                    pass
            _side_close_block()
            total_tokens["input"] += agent.token_usage["in"] - pre_usage["in"]
            total_tokens["output"] += agent.token_usage["out"] - pre_usage["out"]
            total_tokens["cache_read"] += agent.token_usage["cache_read"] - pre_usage["cache_read"]
            total_tokens["cache_write"] += agent.token_usage["cache_write"] - pre_usage["cache_write"]

            # Collect side path raw_blocks (saved together with main path later)
            side_raw = agent.last_raw_blocks
            if side_raw:
                all_side_blocks[side_label] = side_raw

            side_texts.append("".join(side_buf))
            # Update extra with all accumulated side texts for next iteration
            extra["side_text"] = "\n\n".join(side_texts)
            for i, st in enumerate(side_texts):
                extra[f"side{i}"] = st
            if side_blocks:
                all_debug_blocks.append({"phase": f"side_{idx}", "tool_round": 1, "blocks": side_blocks})

        # ── Execute main path ──
        phase_set_cb("main")
        chain_mock = chain.get("mock_content", "").strip()
        if chain_mock:
            for line in chain_mock.split("\n"):
                lc = line.strip()
                if not lc:
                    continue
                is_think = lc.startswith("<think>") and lc.endswith("</think>")
                if is_think:
                    content = re.sub(r'</?think>', '', lc).strip()
                    emit_cb("main", "<think>\n")
                    emit_cb("main", content + "\n")
                    emit_cb("main", "</think>\n")
                else:
                    emit_cb("main", lc + "\n")
            total_tokens["input"] += 1
            log_request_end(log_entry, "ok", total_tokens, int((bj_epoch() - t_start) * 1000), {})
            if debug_ctx:
                debug_ctx.flush()
            return {**total_tokens, "assistant_text": ""}

        main_cfg = chain.get("main_path", {})
        main_route = get_route(main_cfg.get("route", ""))
        if not main_route:
            routes = load_routes()
            main_route = get_route("") or (routes[0] if routes else {})

        main_sp = resolve_template(main_cfg.get("system_prompt", ""), variables, extra)
        main_model = main_cfg.get("model", "")

        thinking_val = main_cfg.get("thinking") or normalize_thinking(main_cfg.get("thinking_budget"))
        agent = PipelineEngine.build_agent_from_route(
            main_route, main_model if main_model else "deepseek-v4-pro",
            system_prompt=main_sp,
            thinking=thinking_val or "",
            chain_label=chain.get("label", ""),
        )
        if debug_ctx:
            agent.debug_ctx = debug_ctx

        agent.set_conversation([dict(m) for m in shared_conv], turn)

        # Apply context cleaning
        keep_turns = main_cfg.get("keep_turns", chain.get("keep_turns", -1))
        PipelineEngine.clean_conversation(agent, keep_turns)

        # Resolve user injection with side text + individual side variables
        inj_template = main_cfg.get("user_injection", "{user_text}")
        main_msg = resolve_template(inj_template, variables, extra)

        # Inject completed background task output before conversation
        done_output = collect_done_tasks(agent.workspace)
        if done_output:
            main_msg = f"[后台任务完成]\n{done_output}\n\n{main_msg}"

        # Injections for logging
        injections = {
            "side": f"side_texts={len(side_texts)} total_chars={sum(len(s) for s in side_texts)}",
            "main": main_msg[:500],
        }

        # Run main agent
        think_open = False
        think_first = True
        text_emitted = False
        output_cfg = main_cfg.get("context_output", {})
        main_blocks: list[dict] = []

        def on_main_event(etype: str, data: dict):
            nonlocal think_open, think_first, text_emitted
            trace_model(session_id, "main", etype, data)


            if etype == "thinking":
                if output_cfg.get("think", True):
                    if not think_open:
                        emit_cb("main", "<think>\n")
                        if think_first:
                            emit_cb("main", "=== [主路] ===\n")
                            think_first = False
                        think_open = True
                    emit_cb("main", data.get("delta", ""))
                main_blocks.append({"type": "thinking", "thinking": data.get("delta", ""), "source": "model"})
            elif etype == "text":
                if think_open:
                    emit_cb("main", "</think>\n\n")
                    think_open = False
                t = data.get("delta", "")
                if t:
                    text_emitted = True
                    if output_cfg.get("text", True):
                        if think_first:
                            emit_cb("main", "=== [主路] ===\n")
                            think_first = False
                        emit_cb("main", t)
                main_blocks.append({"type": "text", "text": t, "source": "model"})
            elif etype == "tool_use":
                if think_open:
                    emit_cb("main", "</think>\n")
                    think_open = False
                if output_cfg.get("tool_use", True):
                    if not think_open:
                        emit_cb("main", "<think>\n")
                        if think_first:
                            emit_cb("main", "=== [主路] ===\n")
                            think_first = False
                        think_open = True
                    emit_cb("main", f"\U0001f527 {data.get('name','?')} {str(data.get('input',{}))[:200]}\n")
                main_blocks.append({
                    "type": "tool_use", "name": data.get("name", "?"),
                    "input": data.get("input", {}), "source": "model",
                })
            elif etype == "tool_result":
                if think_open:
                    emit_cb("main", "</think>\n")
                    think_open = False
                if output_cfg.get("tool_result", True):
                    if not think_open:
                        emit_cb("main", "<think>\n")
                        if think_first:
                            emit_cb("main", "=== [主路] ===\n")
                            think_first = False
                        think_open = True
                    c = data.get("content", "")
                    is_err = data.get("is_error", False)
                    prefix = "❌" if is_err else "\U0001f4cb"
                    emit_cb("main", f"{prefix} {data.get('name','?')} {c[:300]}\n")
                main_blocks.append({
                    "type": "tool_result", "content": data.get("content", ""),
                    "is_error": data.get("is_error", False), "source": "model",
                })
            elif etype == "usage":
                if think_open:
                    emit_cb("main", "</think>\n\n")
                    think_open = False

        # Wire external stop signal to agent
        if stop_event:
            agent.set_stop_event(stop_event)
        pre_usage = dict(agent.token_usage)
        main_err = None
        try:
            agent.handle_message(main_msg, on_event=on_main_event)
        except Exception as e:
            main_err = e

        if think_open:
            emit_cb("main", "</think>\n\n")

        # Fallback: emit last text if nothing emitted
        if not text_emitted and not main_err:
            final_text = agent.get_last_assistant_text()
            if final_text:
                if final_text.strip():
                    emit_cb("main", final_text)

        # Save UA pair for tool context recovery.
        # Hash key = all assistant text joined by \n, matching OWUI's concatenation.
        raw_blocks = agent.last_raw_blocks
        if raw_blocks:
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
                ua_save(all_a_text, raw_blocks,
                        side_blocks=all_side_blocks if all_side_blocks else None)

        total_tokens["input"] += agent.token_usage["in"] - pre_usage["in"]
        total_tokens["output"] += agent.token_usage["out"] - pre_usage["out"]
        total_tokens["cache_read"] += agent.token_usage["cache_read"] - pre_usage["cache_read"]
        total_tokens["cache_write"] += agent.token_usage["cache_write"] - pre_usage["cache_write"]

        if main_blocks:
            all_debug_blocks.append({"phase": "main", "tool_round": 1, "blocks": main_blocks})

        # Clean up
        if agent.is_stopped():
            agent.cleanup_after_stop()

        # Log completion
        duration_ms = int((bj_epoch() - t_start) * 1000)
        status = "ok"
        if main_err:
            status = "error"
        elif agent.is_stopped():
            status = "stopped"

        for debug_round in all_debug_blocks:
            log_round(log_entry, debug_round["phase"], debug_round["tool_round"],
                      debug_round["blocks"])
        log_request_end(log_entry, status, total_tokens, duration_ms, injections,
                        actual_route=main_route.get("label", "") if main_route else "",
                        actual_model=main_model if main_model else agent.model,
                        exit_reason=(
                            "error" if main_err else
                            stop_reason if stop_reason else
                            "timeout" if agent.is_stopped() and duration_ms > 300_000 else
                            "user_stop" if agent.is_stopped() else
                            "normal"
                        ),
                        has_output=bool(agent.get_last_assistant_text()))

        if debug_ctx:
            debug_ctx.flush()

        return {
            **total_tokens,
            "assistant_text": agent.get_last_assistant_text() or "",
        }
