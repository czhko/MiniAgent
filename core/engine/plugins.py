"""v0.4 Plugin engine — user injection, conditional events, session state."""
from __future__ import annotations

import collections
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.infra.settings import load_plugins, get_route
from core.infra.templates import resolve_template
from core.timeutil import bj_epoch


# ── TTL-bounded state store (no external dependencies) ──

class _TTLState:
    """OrderedDict-backed TTL cache. Lazy eviction on access."""

    def __init__(self, maxsize: int = 2000, ttl: float = 1800.0):
        self._data: collections.OrderedDict = collections.OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl

    def _evict(self):
        now = bj_epoch()
        while self._data:
            _key, (_val, ts) = next(iter(self._data.items()))
            if now - ts > self.ttl:
                self._data.popitem(last=False)
            else:
                break
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def get(self, key: str, default: int = 0) -> int:
        self._evict()
        entry = self._data.get(key)
        if entry is not None:
            val, _ts = entry
            self._data.move_to_end(key)
            return val
        return default

    def put(self, key: str, value: int):
        self._evict()
        self._data[key] = (value, bj_epoch())
        self._data.move_to_end(key)

    def delete(self, key: str):
        self._data.pop(key, None)

    def delete_prefix(self, prefix: str):
        to_delete = [k for k in self._data if k.rsplit(":", 1)[-1] == prefix]
        for k in to_delete:
            self._data.pop(k, None)


# ── Module-level state ──

_plugin_state = _TTLState(maxsize=2000, ttl=1800)  # 30 min TTL, 2000 max entries
_plugin_lock = threading.Lock()
_event_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="plg-ev")


# ── Plugin Engine ──

class PluginEngine:
    """Stateless plugin execution engine with session-scoped counters."""

    # ── target matching ──

    @staticmethod
    def _match_target(plugin: dict, route_label: str, model: str) -> bool:
        """Empty list = match all. Non-empty = exact match required."""
        targets_r = plugin.get("target_routes") or []
        if targets_r and route_label not in targets_r:
            return False
        targets_m = plugin.get("target_models") or []
        if targets_m and model not in targets_m:
            return False
        return True

    # ── user injection ──

    @staticmethod
    def apply_user_injection(user_text: str, route_label: str = "",
                             model: str = "") -> str:
        """Walk enabled user_injection plugins. Matching plugins modify user_text."""
        try:
            plugins = load_plugins()
        except Exception:
            return user_text
        for p in plugins:
            if not p.get("enabled", True):
                continue
            if p.get("type") != "user_injection":
                continue
            if not PluginEngine._match_target(p, route_label, model):
                continue
            text = p.get("injection_text", "")
            if not text:
                continue
            try:
                resolved = resolve_template(text, {}, {"user_text": user_text})
            except Exception:
                resolved = text
            pos = p.get("position", "append")
            if pos == "prepend":
                user_text = resolved + user_text
            else:
                user_text = user_text + resolved
        return user_text

    # ── conditional events ──

    @staticmethod
    def process_conditional_events(
        session_id: str, route_label: str, model: str,
        user_text: str, assistant_text: str = "",
    ):
        """Walk enabled conditional_event plugins. Update counters, trigger if threshold met."""
        try:
            plugins = load_plugins()
        except Exception:
            return
        for p in plugins:
            if not p.get("enabled", True):
                continue
            if p.get("type") != "conditional_event":
                continue
            if not PluginEngine._match_target(p, route_label, model):
                continue
            pid = p.get("id", "")
            if not pid:
                continue
            condition_type = p.get("condition_type", "rounds")
            condition_value = p.get("condition_value", 5)
            if not isinstance(condition_value, (int, float)) or condition_value <= 0:
                continue

            state_key = f"{session_id}:{pid}"
            triggered = False

            with _plugin_lock:
                counter = _plugin_state.get(state_key, 0)
                if condition_type == "rounds":
                    counter += 1
                elif condition_type == "length":
                    counter += len(user_text) + len(assistant_text)

                if counter >= condition_value:
                    triggered = True
                    counter = 0
                _plugin_state.put(state_key, counter)

            if triggered:
                _event_executor.submit(PluginEngine._execute_event, p, session_id)

    @staticmethod
    def _execute_event(plugin: dict, session_id: str):
        """Run in daemon thread. Full error isolation — never crash silently."""
        try:
            route_label = plugin.get("event_route", "")
            route = get_route(route_label)
            if not route:
                return
            # Import here to avoid circular dependency at module level
            from core.engine.pipeline import PipelineEngine
            thinking = plugin.get("event_thinking", "")
            agent = PipelineEngine.build_agent_from_route(
                route, plugin.get("event_model", ""),
                thinking=thinking,
            )
            mode = plugin.get("mode", "llm")
            if mode == "llm":
                agent.clear_tools()
                agent.max_iterations = 1

            event_content = plugin.get("event_content", "")
            agent.handle_message(event_content)

            if mode == "llm":
                save_path = plugin.get("save_path", "").strip()
                if save_path:
                    result_text = agent.get_last_assistant_text() or ""
                    if result_text.strip():
                        try:
                            sp = Path(save_path)
                            sp.parent.mkdir(parents=True, exist_ok=True)
                            from core.fsutil import backup_file
                            backup_file(sp)
                            sp.write_text(result_text, encoding="utf-8")
                        except OSError:
                            import traceback as _tb
                            import sys as _sys
                            _tb.print_exc(file=_sys.stderr)
        except Exception:
            import traceback as _tb
            import sys as _sys
            _tb.print_exc(file=_sys.stderr)

    # ── state management ──

    @staticmethod
    def reset_plugin_state(plugin_id: str, session_id: str | None = None):
        """Reset counter for a plugin. session_id=None resets all sessions."""
        with _plugin_lock:
            if session_id:
                _plugin_state.delete(f"{session_id}:{plugin_id}")
            else:
                _plugin_state.delete_prefix(plugin_id)
