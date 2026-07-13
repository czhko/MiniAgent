"""v0.4 Debug module — per-request tracing for OWUI integration debugging.

Design: DebugContext is a mutable "data bucket" passed by reference through
the entire request pipeline. When debug is disabled, debug_ctx is None and
all operations are no-ops. Zero data duplication — everything is stored by
reference, no copies.

JSONL entry types (for Admin color coding):
  turn_marker   — prominent turn separator
  owui          — OWUI callback data (blue)
  model_request — data sent to model API (orange)
  model_response— model API response events (green)
  ua_stats      — UA store hit ratio (purple)
"""
from __future__ import annotations

import json

from core.paths import LOG_DIR
from core.timeutil import bj_now, bj_epoch


class DebugContext:
    """Per-request debug trace. Mutable, append-only, passed by reference.

    Created in server.py when debug.enabled is True. Flows through:
    server → pipeline (OWUI/UA recording) → agent (model request/response) → flush.
    """
    __slots__ = ("id", "session_id", "timestamp", "entries")

    def __init__(self, session_id: str = ""):
        self.id = _short_id()
        self.session_id = session_id
        self.timestamp = bj_now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        self.entries: list[dict] = []

    def _add(self, etype: str, **kwargs):
        self.entries.append({"type": etype, "ts": bj_now().strftime("%H:%M:%S.%f")[:-3], **kwargs})

    def start_turn(self):
        """Called once per request — writes a prominent turn separator."""
        self._add("turn_marker")

    def record_owui(self, body: dict, messages: list):
        """Record the raw OWUI request (body + messages)."""
        safe_msgs = _strip_images(messages)
        self._add("owui", body=body, messages=safe_msgs, msg_count=len(messages))

    def record_model_request(self, provider: str, model: str, conversation: list, params: dict):
        """Record the conversation and params about to be sent to the model API."""
        self._add("model_request", provider=provider, model=model,
                  conversation=conversation, params=params)

    def record_model_response(self, provider: str, events: list[dict], tokens: dict):
        """Record the model's response events and token usage."""
        self._add("model_response", provider=provider, events=events, tokens=tokens)

    def record_ua(self, hits: int, total: int):
        """Record UA store hit statistics for this request.

        Hit definition: number of assistant messages (before the last user message)
        whose tool context was found in UA store. In continue-chat / edit-last-user
        scenarios, hit rate should be 100%.
        """
        rate = round(hits / total, 4) if total > 0 else 0.0
        self._add("ua_stats", hits=hits, total=total, rate=rate)

    def record_side_cache(self, side_label: str, hit: bool):
        """Record side_store cache lookup per side path."""
        self._add("side_cache", side=side_label, hit=hit)

    def flush(self):
        """Write all entries as one JSONL block to a timestamped file."""
        if not self.entries:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = self.timestamp[:19].replace(":", "-")
        path = LOG_DIR / f"debug-{ts}-{self.id}.jsonl"
        lines = []
        for entry in self.entries:
            entry["_id"] = self.id
            entry["_sid"] = self.session_id
            lines.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass


def _short_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _strip_images(messages: list) -> list:
    """Remove base64 image data from messages to keep debug JSON compact."""
    out = []
    for m in messages:
        m2 = {"role": m.get("role", "?")}
        content = m.get("content", "")
        if isinstance(content, list):
            blocks = []
            for b in content:
                t = b.get("type", "")
                if t == "image_url":
                    blocks.append({"type": "image_url", "image_url": {"url": "[base64 stripped]"}})
                elif t == "image":
                    blocks.append({"type": "image", "source": {"type": "base64", "data": "[stripped]"}})
                elif t == "file":
                    f = dict(b.get("file") or b.get("source") or {})
                    f["data"] = "[stripped]"
                    blocks.append({"type": "file", "file": f})
                else:
                    blocks.append(b)
            m2["content"] = blocks
        elif isinstance(content, str):
            m2["content"] = content
        else:
            m2["content"] = str(content)
        out.append(m2)
    return out


def is_debug_enabled() -> bool:
    """Check if debug logging is currently enabled in settings.json."""
    try:
        from core.infra.settings import load_settings
        s = load_settings()
        return bool(s.get("debug", {}).get("enabled", False))
    except Exception:
        return False


def list_debug_sessions() -> list[dict]:
    """List available debug sessions (newest first). Returns [{file, ts, lines}...]."""
    if not LOG_DIR.exists():
        return []
    sessions = []
    for f in sorted(LOG_DIR.glob("debug-*.jsonl"), reverse=True):
        try:
            lines = f.read_text(encoding="utf-8").strip().splitlines()
            sessions.append({"file": f.name, "ts": f.stem.replace("debug-", ""), "lines": len(lines)})
        except OSError:
            pass
    return sessions


def query_debug_session(filename: str) -> dict:
    """Read all entries from a specific debug session file."""
    path = LOG_DIR / filename
    if not path.exists():
        return {"entries": [], "turns": 0}
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return {"entries": [], "turns": 0}
    entries: list[dict] = []
    turn_count = 0
    for line in lines:
        try:
            e = json.loads(line)
            if e.get("type") == "turn_marker":
                turn_count += 1
            entries.append(e)
        except json.JSONDecodeError:
            continue
    return {"entries": entries, "turns": turn_count}


# ── Auto-off: disable debug after 30 min of inactivity ──────

import threading as _threading
import time as _time

_last_activity = 0.0
_check_started = False
_check_lock = _threading.Lock()


def notify_activity():
    """Call from server.py on each chat request. Records activity time and
    starts the auto-off watchdog if not already running."""
    global _last_activity, _check_started
    _last_activity = bj_epoch()
    if not _check_started:
        with _check_lock:
            if not _check_started:
                _check_started = True
                _last_activity = bj_epoch()
                t = _threading.Thread(target=_auto_off_loop, daemon=True)
                t.start()


def _auto_off_loop():
    """Check every 5 min; disable debug if no activity for 30 min."""
    global _check_started
    while True:
        _time.sleep(300)
        idle = bj_epoch() - _last_activity
        if idle > 1800:
            try:
                from core.infra.settings import load_settings, save_settings
                from core.infra.logger import trace_log
                s = load_settings()
                if s.get("debug", {}).get("enabled"):
                    s["debug"]["enabled"] = False
                    result = save_settings(s)
                    trace_log(
                        f"DEBUG_AUTO_OFF: idle={int(idle)}s ok={result.get('ok')}")
                else:
                    trace_log(
                        f"DEBUG_AUTO_OFF: already disabled (idle={int(idle)}s)")
            except Exception:
                pass
            _check_started = False
            return
