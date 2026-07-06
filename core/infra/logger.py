"""v0.4 Structured logging — 3 data streams (owui / model_req / model_resp) + JSONL."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from core.paths import LOG_DIR
from core.timeutil import bj_now, bj_epoch

_trace_lock = threading.Lock()
_jsonl_lock = threading.Lock()

REQUESTS_JSONL = LOG_DIR / "requests.jsonl"


def get_requests_path() -> Path:
    """Public accessor for the JSONL requests log path."""
    return REQUESTS_JSONL
LOG_FILES = {
    "owui":        LOG_DIR / "owui.log",
    "model_req":   LOG_DIR / "model_req.log",
    "model_resp":  LOG_DIR / "model_resp.log",
}
MAX_LOG_SIZE = 500_000
MAX_JSONL_LINES = 5000
TRUNCATE_INTERVAL = 300
_last_truncate = 0.0


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


_ensure_log_dir()


def trace_log(message: str, dest: str = "model_resp") -> None:
    """Append a timestamped line to one of the 3 main log files. dest: 'owui'|'model_req'|'model_resp'."""
    _maybe_truncate_at_runtime()
    ts = bj_now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}\n"
    try:
        with _trace_lock:
            with open(LOG_FILES[dest], "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        print(f"[logger] error: {e}", file=sys.stderr)


# ── model_req.log — framework → model API requests ─────

def log_model_req(protocol: str, entry: dict) -> None:
    """Log a model API request/response entry as JSON."""
    _maybe_truncate_at_runtime()
    entry["protocol"] = protocol
    entry.setdefault("ts", bj_epoch())
    try:
        with _trace_lock:
            with open(LOG_FILES["model_req"], "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[logger] error: {e}", file=sys.stderr)


# ── model_resp.log — model → framework response events ─

def trace_model(session_id: str, phase: str, etype: str, data: dict) -> None:
    """Trace model response events (thinking/text/tool_use/tool_result/usage).
    Also writes raw text/thinking content inline (was trace_model_text)."""
    _maybe_truncate_at_runtime()
    ts = bj_now().strftime("%H:%M:%S.%f")[:-3]
    detail: str
    if etype == "thinking":
        delta = data.get("delta", "")
        detail = f"THINKING len={len(delta)} preview={repr(delta[:80])}"
    elif etype == "text":
        delta = data.get("delta", "")
        detail = f"TEXT len={len(delta)} preview={repr(delta[:80])}"
    elif etype == "tool_use":
        detail = f"TOOL_USE name={data.get('name','?')} id={str(data.get('id','?'))[:20]}"
    elif etype == "tool_result":
        c = data.get("content", "")
        detail = f"TOOL_RESULT name={data.get('name','?')} is_error={data.get('is_error',False)} len={len(c)} preview={repr(c[:80])}"
    elif etype == "usage":
        detail = f"USAGE in={data.get('in',0)} out={data.get('out',0)}"
    else:
        detail = f"UNKNOWN keys={list(data.keys())[:5]}"
    line = f"[{ts}] [{session_id}] {phase} | {detail}\n"
    try:
        with _trace_lock:
            with open(LOG_FILES["model_resp"], "a", encoding="utf-8") as f:
                f.write(line)
                # Raw text/thinking content inline (was separate model_text.log)
                if etype in ("thinking", "text"):
                    delta = data.get("delta", "")
                    if delta:
                        f.write(delta)
    except Exception as e:
        print(f"[logger] error: {e}", file=sys.stderr)


# ── owui.log — OWUI → framework request dumps ──────────

def dump_owui(session_id: str, messages: list[dict], source: str = "v1") -> None:
    """Dump OWUI request messages as raw JSON. source='v1'|'ollama'."""
    ts = bj_now().strftime("%Y-%m-%d %H:%M:%S")
    source_tag = f"OWUI[{source}]"
    try:
        with _trace_lock:
            with open(LOG_FILES["owui"], "a", encoding="utf-8") as f:
                f.write(f"===== {source_tag} [{session_id}] {ts} msg_count={len(messages)} =====\n")
                f.write(json.dumps(messages, ensure_ascii=False, indent=2) + "\n\n")
    except Exception as e:
        print(f"[logger] error: {e}", file=sys.stderr)


# ── Structured JSONL request logging ────────────────────

def log_request_start(session_id: str, chain: str, route: str, model: str) -> dict:
    """Create a new request log entry. Returns the mutable dict to be updated by caller."""
    return {
        "session_id": session_id,
        "timestamp": bj_now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "chain": chain,
        "route": route,
        "model": model,
        "status": "pending",
        "duration_ms": 0,
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
        "injections": {},
        "debug": {"rounds": []},
    }


def log_round(entry: dict, phase: str, tool_round: int, blocks: list[dict]) -> None:
    """Compute structured summary from raw blocks."""
    tools: list[str] = []
    errors = 0
    thinking_chars = 0
    text_chars = 0
    for b in blocks:
        t = b.get("type", "")
        if t == "tool_use":
            tools.append(b.get("name", "?"))
        elif t == "tool_result" and b.get("is_error"):
            errors += 1
        elif t == "thinking":
            thinking_chars += len(b.get("thinking", ""))
        elif t == "text":
            text_chars += len(b.get("text", ""))
    entry["debug"]["rounds"].append({
        "phase": phase,
        "tool_round": tool_round,
        "tools": tools,
        "errors": errors,
        "thinking_chars": thinking_chars,
        "text_chars": text_chars,
    })


def log_request_end(entry: dict, status: str, tokens: dict, duration_ms: int,
                    injections: dict | None = None,
                    actual_route: str = "", actual_model: str = "",
                    exit_reason: str = "", has_output: bool = False) -> None:
    """Complete the entry and write as a compact JSONL line."""
    entry["status"] = status
    entry["tokens"] = {
        "input": tokens.get("input", 0),
        "output": tokens.get("output", 0),
        "cache_read": tokens.get("cache_read", 0),
        "cache_write": tokens.get("cache_write", 0),
    }
    entry["duration_ms"] = duration_ms
    if actual_route:
        entry["route"] = actual_route
    if actual_model:
        entry["model"] = actual_model
    if exit_reason:
        entry["exit_reason"] = exit_reason
    entry["has_output"] = has_output
    if injections:
        entry["injections"] = injections
    try:
        with _jsonl_lock:
            with open(REQUESTS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[logger] error: {e}", file=sys.stderr)
    _truncate_jsonl()


def query_logs(limit: int = 100, since: str | None = None,
               status: str | None = None, chain: str | None = None) -> dict:
    """Query requests.jsonl. Returns entries WITHOUT full debug blocks for list display."""
    entries: list[dict] = []
    stats = {"total": 0, "total_in": 0, "total_out": 0, "ok": 0, "stopped": 0, "error": 0,
             "cache_read": 0, "cache_write": 0, "by_chain": {}}

    if not REQUESTS_JSONL.exists():
        return {"entries": [], "stats": stats}

    try:
        with _jsonl_lock:
            lines = REQUESTS_JSONL.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return {"entries": [], "stats": stats}

    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if since and entry.get("timestamp", "") < since:
            continue
        if status and entry.get("status") != status:
            continue
        if chain and entry.get("chain") != chain:
            continue

        tok = entry.get("tokens", {})
        stats["total"] += 1
        stats["total_in"] += tok.get("input", 0)
        stats["total_out"] += tok.get("output", 0)
        stats["cache_read"] += tok.get("cache_read", 0)
        stats["cache_write"] += tok.get("cache_write", 0)
        st = entry.get("status", "")
        if st == "ok":
            stats["ok"] += 1
        elif st == "stopped":
            stats["stopped"] += 1
        elif st not in ("", "pending"):
            stats["error"] += 1
        c = entry.get("chain", "unknown")
        if c not in stats["by_chain"]:
            stats["by_chain"][c] = {"requests": 0, "in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        stats["by_chain"][c]["requests"] += 1
        stats["by_chain"][c]["in"] += tok.get("input", 0)
        stats["by_chain"][c]["out"] += tok.get("output", 0)
        stats["by_chain"][c]["cache_read"] += tok.get("cache_read", 0)
        stats["by_chain"][c]["cache_write"] += tok.get("cache_write", 0)

        if limit == 0:
            continue

        clean = {
            "timestamp": entry.get("timestamp", ""),
            "session_id": entry.get("session_id", ""),
            "chain": entry.get("chain", ""),
            "route": entry.get("route", ""),
            "model": entry.get("model", ""),
            "status": entry.get("status", "?"),
            "exit_reason": entry.get("exit_reason", ""),
            "has_output": entry.get("has_output", False),
            "duration_ms": entry.get("duration_ms", 0),
            "tokens": entry.get("tokens", {}),
        }
        entries.append(clean)

        if len(entries) >= limit:
            break

    return {"entries": entries, "stats": stats}


def get_log_detail(session_id: str) -> dict | None:
    """Get FULL debug detail for one request."""
    if not REQUESTS_JSONL.exists():
        return None
    try:
        with _jsonl_lock:
            lines = REQUESTS_JSONL.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("session_id") == session_id:
            return entry
    return None


def _maybe_truncate_at_runtime():
    global _last_truncate
    now = bj_epoch()
    if now - _last_truncate >= TRUNCATE_INTERVAL:
        _last_truncate = now
        _truncate_logs()


def _truncate_jsonl():
    """Keep last MAX_JSONL_LINES in requests.jsonl."""
    try:
        with _jsonl_lock:
            if REQUESTS_JSONL.exists():
                lines = REQUESTS_JSONL.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) > MAX_JSONL_LINES:
                    REQUESTS_JSONL.write_text("\n".join(lines[-MAX_JSONL_LINES:]) + "\n",
                                              encoding="utf-8")
    except Exception as e:
        print(f"[logger] error: {e}", file=sys.stderr)


def _truncate_logs():
    """Truncate all log files if they exceed MAX_LOG_SIZE bytes. Line-aware."""
    with _trace_lock:
        for label, path in {**LOG_FILES, "da_api": LOG_DIR / "da-api.log"}.items():
            try:
                if path.exists() and path.stat().st_size > MAX_LOG_SIZE:
                    lines = path.read_text(encoding="utf-8").splitlines()
                    target = MAX_LOG_SIZE // 2
                    kept, total = [], 0
                    for line in reversed(lines):
                        total += len(line.encode("utf-8")) + 1
                        kept.append(line)
                        if total >= target:
                            break
                    path.write_text("\n".join(reversed(kept)) + "\n", encoding="utf-8")
            except Exception as e:
                print(f"[logger] error: {e}", file=sys.stderr)
    _truncate_jsonl()


_truncate_logs()
