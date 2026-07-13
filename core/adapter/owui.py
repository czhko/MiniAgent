"""OWUI protocol adaptation — conversation rebuild, text merge, &lt;think&gt; wrapping.

Pure OWUI adapters. No orchestration, no chain logic, no plugins.
"""
from __future__ import annotations

import json, threading

from core.codec.owui import extract_text
from core.timeutil import bj_now
from core.infra.logger import trace_model, log_round
from core.store.ua import load as ua_load


# ── OWUI text overlay ─────────────────────────────────────

def apply_owui_text(raw_blocks: list[dict], owui_text: str) -> list[dict]:
    """Overwrite text blocks in raw_blocks with OWUI text.
    If OWUI text length matches saved fragment boundaries -> split & assign per fragment.
    Otherwise -> user edited -> replace all text blocks with OWUI text.
    """
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


# ── Conversation rebuild ──────────────────────────────────

def rebuild_conversation(
    messages: list[dict],
    separator: str = "",
    keep_think: int = -1,
) -> tuple[list[dict], int, int, int, dict[str, list]]:
    """Rebuild structured conversation from OWUI messages with UA store restoration.

    Returns (conv, turn, ua_hits, ua_total, all_ua_sides).
    Caller is responsible for setting agent.conversation / agent.set_turn / clean_orphaned_tool_results.
    """
    last_user_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i

    conv: list[dict] = []
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
            conv.append({
                "role": "user", "content": content, "turn": turn,
                "turn_time": bj_now().isoformat(),
            })
        elif role == "assistant":
            text = content or "(thinking)"
            # Strip injected trailing " over" from SSE before any processing
            if text and text.endswith(" over"):
                text = text[:-5]
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
            if separator:
                idx = text.rfind(separator)
                if idx != -1:
                    trimmed = text[idx + len(separator):].strip()
                    if trimmed:
                        text = trimmed
            if raw_blocks:
                apply_owui_text(raw_blocks, text)
                for rb in raw_blocks:
                    if isinstance(rb, dict) and rb.get("role") in ("assistant", "user"):
                        if "turn" not in rb:
                            rb["turn"] = turn
                        conv.append(rb)
            else:
                conv.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "turn": turn,
                })
    for entry in conv:
        c = entry.get("content", [])
        if isinstance(c, list):
            for b in c:
                if b.get("type") == "text":
                    diff = b["text"].count("<thinking>") - b["text"].count("</thinking>")
                    if diff > 0:
                        b["text"] += "</thinking>" * diff
    return conv, turn, ua_hits, ua_total, all_ua_sides


# ── Think-wrapping event callback factory ─────────────────

def make_on_event(session_id: str, phase: str, emit_cb, log_entry,
                  tool_render: str = "think", model: str = ""):
    """Create on_event callback with &lt;think&gt;-wrapping and logging.

    tool_render: "think" (default) — wrap tools in &lt;think&gt; for collapsible UI.
                "details" — emit &lt;details type=tool_calls&gt; cards inside &lt;think&gt;.
    model: model name — for MiniMax-specific filtering.
    """
    _is_minimax = "minimax" in model.lower()
    think_open = False
    _thinking_depth = 0  # MiniMax: track <thinking> opens without closes
    _think_buf = ""  # MiniMax: bridge partial thinking tags across chunks
    _text_buf = ""  # MiniMax: bridge split <thinking>/</thinking> in text events
    _prev_char = ""  # MiniMax: last 2 chars of stripped text for cross-chunk regex
    _owui_text = ""  # accumulated emitted text for UA hash (stripped by ua.py)
    tool_round = 0
    round_blocks: list[dict] = []
    _last_etype = ""
    _pending_tools: dict[str, tuple] = {}
    _tool_running = False
    _tool_stop = threading.Event()

    def _start_dots():
        """Background daemon: emit '.' every 0.2s while tool is running."""
        def _pulse():
            while not _tool_stop.is_set():
                emit_cb(phase, ".")
                _tool_stop.wait(0.2)
        threading.Thread(target=_pulse, daemon=True).start()

    def on_event(etype: str, data: dict):
        nonlocal think_open, _thinking_depth, _think_buf, _text_buf, _prev_char, _owui_text, tool_round, round_blocks, _last_etype, _tool_running
        trace_model(session_id, phase, etype, data)

        if etype == "thinking":
            d = data.get("delta", "")
            if _is_minimax:
                d = _think_buf + d
                d = d.replace("<thinking>", "").replace("</thinking>", "")
                tag_start = d.rfind("<")
                if tag_start >= 0 and len(d) - tag_start <= 11:
                    _think_buf = d[tag_start:]
                    d = d[:tag_start]
                else:
                    _think_buf = ""
            round_blocks.append({"type": "thinking", "thinking": d, "source": "model"})
            if d:
                if _is_minimax and _thinking_depth > 0:
                    emit_cb(phase, d)
                else:
                    if not think_open:
                        emit_cb(phase, "<think>\n")
                        think_open = True
                    emit_cb(phase, d)
            _last_etype = "thinking"
        elif etype == "text":
            round_blocks.append({"type": "text", "text": data.get("delta", ""), "source": "model"})
            if think_open:
                emit_cb(phase, "</think>\n\n")
                think_open = False
            t = data.get("delta", "")
            if t:
                if _is_minimax:
                    t = _text_buf + t
                    _text_buf = ""
                    t = t.replace("<mm:think>", "</thinking>").replace("</mm:think>", "</thinking>")
                    _thinking_depth += t.count("<thinking>") - t.count("</thinking>")
                    if _thinking_depth < 0:
                        _thinking_depth = 0
                    _check = _prev_char + t.replace("<thinking>", "").replace("</thinking>", "")
                    import re as _re
                    if _thinking_depth > 0 and _prev_char and _re.search(r'[a-zA-Z](?:[一-鿿]|#)|[a-z]\.[A-Z]', _check[:len(_prev_char)+1]):
                        emit_cb(phase, "</thinking>" * _thinking_depth)
                        _thinking_depth = 0
                    _prev_char = _check[-2:] if len(_check) >= 2 else _check
                    tag_start = t.rfind("<")
                    if tag_start >= 0 and len(t) - tag_start <= 11 and not (t[tag_start:].startswith("</thinking>") or t[tag_start:].startswith("<thinking>")):
                        _text_buf = t[tag_start:]
                        t = t[:tag_start]
                    else:
                        _text_buf = ""
                if _is_minimax and t:
                    _owui_text += t
                emit_cb(phase, t)
            _last_etype = "text"
        elif etype == "tool_use":
            if _is_minimax:
                if _think_buf:
                    _think_buf = ""
                if _text_buf:
                    _text_buf = ""
                _prev_char = ""
                if _thinking_depth > 0:
                    emit_cb(phase, "</thinking>" * _thinking_depth)
                    _thinking_depth = 0
            round_blocks.append({"type": "tool_use", "name": data.get("name", "?"), "input": data.get("input", {}), "source": "model"})
            name = data.get("name", "?")
            inp = str(data.get("input", {}))
            _last_etype = "tool_use"
            if tool_render == "details":
                if not _pending_tools:
                    if think_open:
                        emit_cb(phase, "</think>\n\n")
                        think_open = False
                    emit_cb(phase, "<think>\n")
                    think_open = True
                    if not _tool_running:
                        emit_cb(phase, "工具调用中")
                        _tool_running = True
                        _tool_stop.clear()
                        _start_dots()
                import html as _html
                args_escaped = _html.escape(json.dumps(data.get("input", {}), ensure_ascii=False))
                cid = data.get("id", "")
                _pending_tools[cid] = (name, args_escaped)
            else:
                if not think_open:
                    emit_cb(phase, "<think>\n")
                    think_open = True
                emit_cb(phase, f"\U0001f527 {name} {inp}\n")
                if not _tool_running:
                    emit_cb(phase, "工具调用中")
                    _tool_running = True
                    _tool_stop.clear()
                    _start_dots()
        elif etype == "tool_result":
            c = data.get("content", "") or ""
            is_err = data.get("is_error", False)
            model = data.get("model", "")
            round_blocks.append({"type": "tool_result", "content": c, "is_error": is_err, "source": "model"})
            _last_etype = "tool_result"
            if c:
                prefix = "❌" if is_err else "\U0001f4cb"
                label = f"[{model}] " if model else ""
                # Stop dot pulse when all pending tools have results
                if _tool_running and (not tool_render == "details" or len(_pending_tools) <= 1):
                    _tool_stop.set()
                    emit_cb(phase, "\n调用完毕\n")
                    _tool_running = False
                if tool_render == "details":
                    tu_id = data.get("tool_use_id", "")
                    t_name, t_args = _pending_tools.pop(tu_id, ("?", ""))
                    emit_cb(phase,
                        f'\n<details type="tool_calls" done="true" id="{tu_id}" name="{t_name}" arguments="{t_args}">\n'
                        f'<summary>\U0001f527 {t_name}</summary>\n'
                        f'{prefix} {label}{c}\n</details>\n')
                else:
                    emit_cb(phase, f"{prefix} {data.get('name','?')} {label}{c}\n")
        elif etype == "usage":
            _pending_tools.clear()
            if _tool_running:
                _tool_stop.set()
                emit_cb(phase, "\n调用完毕\n")
                _tool_running = False
            if _is_minimax:
                if _think_buf:
                    _think_buf = ""
                if _text_buf:
                    _text_buf = ""
                _prev_char = ""
                _owui_text += "\n"
                if _thinking_depth > 0:
                    emit_cb(phase, "</thinking>" * _thinking_depth)
                    _thinking_depth = 0
            if think_open:
                emit_cb(phase, "</think>\n\n")
                think_open = False
            tool_round += 1
            log_round(log_entry, phase, tool_round, round_blocks)
            round_blocks.clear()
            _last_etype = ""

    return on_event, lambda: (think_open, tool_round, round_blocks, _owui_text)
