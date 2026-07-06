"""v0.4 Admin backend — REST API + inline HTML page."""
from __future__ import annotations

import base64, hashlib, json, re, threading, time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from core.paths import ROOT_DIR
from core.infra.settings import load_settings, save_settings, load_routes, save_routes, load_chains, save_chains, load_plugins, save_plugins, get_route
from core.timeutil import bj_now, bj_epoch
from core.infra.logger import query_logs, get_log_detail
from core.http_utils import preprocess_base_url
from core.fsutil import backup_file

_START_TIME = bj_epoch()
_crud_lock = threading.Lock()

# Cache for log stats — avoids re-reading entire JSONL on every dashboard refresh
_log_stats_cache: dict = {"mtime": 0, "stats": {}}

def _cached_log_stats() -> dict:
    from core.infra.logger import get_requests_path
    REQUESTS_JSONL = get_requests_path()
    try:
        mtime = REQUESTS_JSONL.stat().st_mtime
    except OSError:
        return {}
    if mtime == _log_stats_cache["mtime"]:
        return _log_stats_cache["stats"]
    data = query_logs(limit=0)
    stats = data.get("stats", {})
    _log_stats_cache["mtime"] = mtime
    _log_stats_cache["stats"] = stats
    return stats

class AdminBackend:
    """Stateless admin API backend. All methods are static."""

    @staticmethod
    def handle_request(path: str, method: str, body: dict | None,
                       query: str) -> tuple[bytes, str, int]:
        """Route dispatcher. Returns (body_bytes, content_type, status_code)."""
        # GET routes
        if method == "GET":
            if path in ("/admin", "/admin/"):
                return AdminBackend._admin_html()
            if path == "/admin/api/status":
                return AdminBackend._dashboard_status()
            if path == "/admin/api/status/health":
                return AdminBackend._health_status()
            if path == "/admin/api/settings":
                return AdminBackend._get_settings()
            if path == "/admin/api/routes":
                return AdminBackend._list_routes()
            if path == "/admin/api/chains":
                return AdminBackend._list_chains()
            if path == "/admin/api/plugins":
                return AdminBackend._list_plugins()
            if path == "/admin/api/logs":
                return AdminBackend._query_logs(query)
            if path == "/admin/api/debug/dates":
                return AdminBackend._debug_dates()
            if path == "/admin/api/debug/logs":
                return AdminBackend._debug_logs(query)
            if path == "/admin/api/default-sp":
                return AdminBackend._default_sp()
            if path == "/admin/api/files":
                return AdminBackend._list_files(query)
            if path == "/admin/api/files/read":
                return AdminBackend._read_file(query)

            # Regex routes
            m = re.match(r'^/admin/api/routes/([^/]+)/models$', path)
            if m:
                return AdminBackend._route_models(unquote(m.group(1)))

            m = re.match(r'^/admin/api/logs/([^/]+)$', path)
            if m:
                return AdminBackend._log_detail(unquote(m.group(1)))

            m = re.match(r'^/admin/api/plugins/([^/]+)/reset$', path)
            if m:
                return AdminBackend._reset_plugin_state(unquote(m.group(1)))

        # POST/PUT — create or update
        if method in ("POST", "PUT"):
            if path == "/admin/api/settings":
                return AdminBackend._put_settings(body)
            if path == "/admin/api/routes":
                return AdminBackend._upsert_route(body)
            if path == "/admin/api/chains":
                return AdminBackend._upsert_chain(body)
            if path == "/admin/api/plugins":
                return AdminBackend._upsert_plugin(body)
            if path == "/admin/api/speedtest":
                return AdminBackend._speedtest(body)
            if path == "/admin/api/files" and method == "POST":
                return AdminBackend._create_file(body)
            if path == "/admin/api/files/save":
                return AdminBackend._save_file(body)
            if path == "/admin/api/routes/test-proxy":
                return AdminBackend._test_proxy(body)

            m = re.match(r'^/admin/api/routes/([^/]+)$', path)
            if m:
                return AdminBackend._upsert_route(body, unquote(m.group(1)))

            m = re.match(r'^/admin/api/chains/([^/]+)$', path)
            if m:
                return AdminBackend._upsert_chain(body, unquote(m.group(1)))

            m = re.match(r'^/admin/api/plugins/([^/]+)$', path)
            if m:
                return AdminBackend._upsert_plugin(body, unquote(m.group(1)))

        if method == "DELETE":
            if path == "/admin/api/files":
                return AdminBackend._delete_file(query)
            m = re.match(r'^/admin/api/routes/([^/]+)$', path)
            if m:
                return AdminBackend._delete_route(unquote(m.group(1)))
            m = re.match(r'^/admin/api/chains/([^/]+)$', path)
            if m:
                return AdminBackend._delete_chain(unquote(m.group(1)))
            m = re.match(r'^/admin/api/plugins/([^/]+)$', path)
            if m:
                return AdminBackend._delete_plugin(unquote(m.group(1)))

        # PATCH routes — available for direct API use (not called by the built-in
        # web UI, but can be used by external tooling or custom frontends to
        # update individual setting sections atomically).
        if method == "PATCH":
            if path == "/admin/api/settings/system":
                return AdminBackend._patch_system(body)
            if path == "/admin/api/settings/proxy":
                return AdminBackend._patch_proxy(body)
            if path == "/admin/api/files/rename":
                return AdminBackend._rename_file(body)

        return AdminBackend._json_response({"error": "Not found"}, 404)

    @staticmethod
    def _json_response(data: Any, status: int = 200) -> tuple[bytes, str, int]:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        return body, "application/json", status

    @staticmethod
    def _admin_html() -> tuple[bytes, str, int]:
        html_path = Path(__file__).resolve().parent / "admin.html"
        try:
            return html_path.read_bytes(), "text/html; charset=utf-8", 200
        except (OSError, FileNotFoundError):
            return b"<html><body><h1>admin.html not found</h1></body></html>", "text/html; charset=utf-8", 500

    # ── Dashboard ─────────────────────────────────────

    @staticmethod
    def _dashboard_status() -> tuple[bytes, str, int]:
        s = load_settings()
        routes = load_routes()
        chains = load_chains()
        stats = _cached_log_stats()

        # Route/proxy health checks are deferred to /admin/api/status/health
        # so the dashboard renders immediately.
        proxy_addr = ""
        if isinstance(s.get("proxy"), dict):
            proxy_addr = s.get("proxy", {}).get("address", "")

        return AdminBackend._json_response({
            "routes": {"total": len(routes),
                       "results": [{"label": r.get("label"), "enabled": r.get("enabled", True)}
                                   for r in routes]},
            "proxy": {"address": proxy_addr},
            "chains": {"total": len(chains),
                       "active": sum(1 for c in chains if c.get("enabled", True))},
            "stats": {"total_requests": stats.get("total", 0),
                      "ok": stats.get("ok", 0),
                      "stopped": stats.get("stopped", 0),
                      "errors": stats.get("error", 0),
                      "tokens_in": stats.get("total_in", 0),
                      "tokens_out": stats.get("total_out", 0),
                      "cache_read": stats.get("cache_read", 0)},
            "version": {"build_date": "2026-07-01", "version": "0.4.3",
                        "uptime_seconds": int(bj_epoch() - _START_TIME)},
        })

    @staticmethod
    def _health_status() -> tuple[bytes, str, int]:
        """Route health + proxy check — called async after dashboard renders."""
        routes = load_routes()
        s = load_settings()

        from core.http_utils import resolve_proxy_url, build_httpx_client
        proxy_config = s.get("proxy", {}) if isinstance(s.get("proxy"), dict) else {"address": s.get("proxy", "")}
        global_proxy_url = resolve_proxy_url(proxy_config)

        def _check_route(r):
            if not r.get("enabled", True):
                return ("__route__", r.get("label", ""), False)
            try:
                # Per-route proxy: respect use_proxy flag
                route_pc = dict(proxy_config)
                if not r.get("use_proxy", True):
                    route_pc["use_for_api"] = False
                rp_url = resolve_proxy_url(route_pc) if route_pc.get("use_for_api") else None
                with build_httpx_client(rp_url, timeout=5, connect_timeout=3) as c:
                    resp = c.get(preprocess_base_url(r.get("base_url", "")) + "/v1/models",
                                 headers={"Authorization": f"Bearer {r.get('api_key','')}"})
                    return ("__route__", r.get("label", ""), resp.status_code < 500)
            except Exception:
                return ("__route__", r.get("label", ""), False)

        proxy_url = global_proxy_url

        results = {}
        available = 0
        proxy_online = False
        proxy_addr = ""
        if isinstance(s.get("proxy"), dict):
            proxy_addr = s.get("proxy", {}).get("address", "")

        # Submit all checks together, collect with a single deadline
        with ThreadPoolExecutor(max_workers=min(len(routes) + 1 or 2, 12)) as executor:
            fs = {executor.submit(_check_route, r): r for r in routes}
            if proxy_url:
                def _check_proxy():
                    try:
                        parsed = urlparse(proxy_url)
                        host = (parsed.hostname or "127.0.0.1")
                        port = parsed.port or 8080
                        import socket
                        sock = socket.create_connection((host, port), timeout=5)
                        sock.close()
                        return ("__proxy__", True)
                    except Exception:
                        return ("__proxy__", False)
                fs[executor.submit(_check_proxy)] = None

            deadline = bj_epoch() + 10  # hard deadline for the whole batch
            for future in as_completed(fs):
                remaining = deadline - bj_epoch()
                if remaining <= 0:
                    break
                try:
                    kind, *payload = future.result(timeout=remaining)
                except Exception:
                    continue
                if kind == "__route__":
                    label, ok = payload[0], payload[1]
                    results[label] = ok
                    if ok:
                        available += 1
                elif kind == "__proxy__":
                    proxy_online = payload[0]

        return AdminBackend._json_response({
            "routes": {"available": available, "results": results},
            "proxy": {"online": proxy_online, "address": proxy_addr},
        })

    # ── Settings ───────────────────────────────────────

    @staticmethod
    def _get_settings() -> tuple[bytes, str, int]:
        s = load_settings()
        s["routes"] = load_routes()
        s["chains"] = load_chains()
        return AdminBackend._json_response(s)

    @staticmethod
    def _put_settings(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        with _crud_lock:
            # Save routes to routes.json if present
            if "routes" in body:
                rr = save_routes(body["routes"])
                if not rr.get("ok"):
                    return AdminBackend._json_response(rr, 400)
            # Save chains to chains.json if present
            if "chains" in body:
                rc = save_chains(body["chains"])
                if not rc.get("ok"):
                    return AdminBackend._json_response(rc, 400)
            # Save system/proxy/vision to settings.json
            settings = {k: v for k, v in body.items() if k not in ("routes", "chains")}
            result = save_settings(settings)
        return AdminBackend._json_response(result, 200 if result.get("ok") else 400)

    @staticmethod
    def _patch_system(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        with _crud_lock:
            s = load_settings()
            s.setdefault("system", {})
            s["system"].update(body)
            result = save_settings(s)
        return AdminBackend._json_response(result)

    @staticmethod
    def _patch_proxy(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        with _crud_lock:
            s = load_settings()
            s.setdefault("proxy", {})
            s["proxy"].update(body)
            result = save_settings(s)
        return AdminBackend._json_response(result)

    # ── Routes CRUD ────────────────────────────────────

    @staticmethod
    def _list_routes() -> tuple[bytes, str, int]:
        return AdminBackend._json_response(load_routes())

    @staticmethod
    def _upsert_route(body: dict | None, label: str | None = None) -> tuple[bytes, str, int]:
        """Create or update a route. If body has 'label' or label param, upsert by label."""
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        target = label or body.get("label", "")
        if not target:
            return AdminBackend._json_response({"ok": False, "error": "label required"}, 400)
        with _crud_lock:
            routes = load_routes()
            for r in routes:
                if r.get("label") == target:
                    r.update(body)
                    return AdminBackend._json_response(save_routes(routes))
            # Create new
            entry = {
                "label": target,
                "protocol": body.get("protocol", "anthropic"),
                "api_key": body.get("api_key", ""),
                "base_url": body.get("base_url", ""),
                "thinking": body.get("thinking", "off"),
                "max_iterations": body.get("max_iterations", 20),
                "context_1m": body.get("context_1m", False),
                "enabled": body.get("enabled", True),
                "selected_models": body.get("selected_models", []),
                "use_proxy": body.get("use_proxy", True),
            }
            routes.append(entry)
            return AdminBackend._json_response(save_routes(routes))

    @staticmethod
    def _delete_route(label: str) -> tuple[bytes, str, int]:
        with _crud_lock:
            routes = load_routes()
            new_routes = [r for r in routes if r.get("label") != label]
            if len(new_routes) == len(routes):
                return AdminBackend._json_response({"ok": False, "error": f"Route '{label}' not found"}, 404)
            result = save_routes(new_routes)
        return AdminBackend._json_response(result)

    @staticmethod
    def _route_models(label: str) -> tuple[bytes, str, int]:
        route = get_route(label)
        if not route:
            return AdminBackend._json_response({"data": [], "error": "Route not found"})
        api_key = route.get("api_key", "")
        if not api_key:
            return AdminBackend._json_response({"data": [], "error": "No API key configured for this route"})
        url = preprocess_base_url(route.get("base_url", "")) + "/v1/models"
        # Per-route proxy
        from core.http_utils import resolve_proxy_url, build_httpx_client
        s = load_settings()
        proxy_config = s.get("proxy", {}) if isinstance(s.get("proxy"), dict) else {"address": s.get("proxy", "")}
        route_pc = dict(proxy_config)
        if not route.get("use_proxy", True):
            route_pc["use_for_api"] = False
        rp_url = resolve_proxy_url(route_pc) if route_pc.get("use_for_api") else None
        try:
            with build_httpx_client(rp_url, timeout=15, connect_timeout=5) as c:
                resp = c.get(url,
                             headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code != 200:
                    return AdminBackend._json_response({"data": [], "error": f"HTTP {resp.status_code}"})
                raw = resp.json()
                if isinstance(raw, list):
                    models = raw
                elif isinstance(raw, dict):
                    models = raw.get("data", raw.get("models", []))
                else:
                    models = []
                return AdminBackend._json_response({"data": models})
        except Exception as e:
            return AdminBackend._json_response({"data": [], "error": str(e)[:200]})

    @staticmethod
    def _test_proxy(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"online": False, "error": "Invalid JSON"}, 400)
        addr = body.get("address", "")
        if not addr:
            return AdminBackend._json_response({"online": False, "error": "No address"})
        t0 = bj_epoch()
        try:
            from core.http_utils import resolve_proxy_url
            proxy_url = resolve_proxy_url({"address": addr})
            with httpx.Client(proxy=proxy_url, timeout=httpx.Timeout(10, connect=5)) as c:
                resp = c.get("https://www.google.com")
                return AdminBackend._json_response({
                    "online": resp.status_code < 500,
                    "latency_ms": int((bj_epoch() - t0) * 1000),
                })
        except Exception as e:
            return AdminBackend._json_response({
                "online": False,
                "latency_ms": int((bj_epoch() - t0) * 1000),
                "error": str(e)[:100],
            })

    # ── Chains CRUD ────────────────────────────────────

    @staticmethod
    def _list_chains() -> tuple[bytes, str, int]:
        return AdminBackend._json_response(load_chains())

    @staticmethod
    def _upsert_chain(body: dict | None, label: str | None = None) -> tuple[bytes, str, int]:
        """Create or update a chain. If body has 'label' or label param, upsert by label."""
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        target = label or body.get("label", "")
        if not target:
            return AdminBackend._json_response({"ok": False, "error": "label required"}, 400)
        with _crud_lock:
            chains = load_chains()
            for c in chains:
                if c.get("label") == target:
                    c.update(body)
                    return AdminBackend._json_response(save_chains(chains))
            # Create new
            entry = {
                "label": target,
                "display_name": body.get("display_name", ""),
                "enabled": body.get("enabled", True),
                "side_paths": body.get("side_paths", []),
                "main_path": body.get("main_path", {}),
                "variables": body.get("variables", {}),
                "separator": body.get("separator", ""),
                "keep_turns": body.get("keep_turns", -1),
                "keep_think": body.get("keep_think", -1),
                "show_history_stats": body.get("show_history_stats", True),
                "parallel_side": body.get("parallel_side", False),
                "mock_content": body.get("mock_content", ""),
            }
            chains.append(entry)
            return AdminBackend._json_response(save_chains(chains))

    @staticmethod
    def _delete_chain(label: str) -> tuple[bytes, str, int]:
        with _crud_lock:
            chains = load_chains()
            # Check for protected chain (by flag, by label, or by first default chain)
            target = next((c for c in chains if c.get("label") == label), None)
            if target and (
                target.get("protected") is True
                or label == "测试"
                or target.get("display_name") == "mock-think-test"
            ):
                return AdminBackend._json_response({"ok": False, "error": f"Chain '{label}' is protected and cannot be deleted"}, 403)
            new_chains = [c for c in chains if c.get("label") != label]
            if len(new_chains) == len(chains):
                return AdminBackend._json_response({"ok": False, "error": f"Chain '{label}' not found"}, 404)
            result = save_chains(new_chains)
        return AdminBackend._json_response(result)

    # ── Plugins CRUD ────────────────────────────────────

    @staticmethod
    def _list_plugins() -> tuple[bytes, str, int]:
        return AdminBackend._json_response(load_plugins())

    @staticmethod
    def _upsert_plugin(body: dict | None, pid: str | None = None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        target = pid or body.get("id", "")
        if not target:
            return AdminBackend._json_response({"ok": False, "error": "id required"}, 400)
        with _crud_lock:
            plugins = load_plugins()
            for p in plugins:
                if p.get("id") == target:
                    p.update(body)
                    return AdminBackend._json_response(save_plugins(plugins))
            # Create new with defaults
            import uuid
            ptype = body.get("type", "user_injection")
            entry: dict = {"id": target or uuid.uuid4().hex[:12], "type": ptype}
            if ptype == "user_injection":
                entry.update({
                    "label": body.get("label", "新注入"),
                    "enabled": body.get("enabled", True),
                    "target_routes": body.get("target_routes", []),
                    "target_models": body.get("target_models", []),
                    "injection_text": body.get("injection_text", ""),
                    "position": body.get("position", "append"),
                })
            else:
                entry.update({
                    "label": body.get("label", "新事件"),
                    "enabled": body.get("enabled", True),
                    "target_routes": body.get("target_routes", []),
                    "target_models": body.get("target_models", []),
                    "event_content": body.get("event_content", ""),
                    "mode": body.get("mode", "llm"),
                    "save_path": body.get("save_path", ""),
                    "event_route": body.get("event_route", ""),
                    "event_model": body.get("event_model", ""),
                    "event_thinking": body.get("event_thinking", ""),
                    "condition_type": body.get("condition_type", "rounds"),
                    "condition_value": body.get("condition_value", 5),
                })
            plugins.append(entry)
            return AdminBackend._json_response(save_plugins(plugins))

    @staticmethod
    def _delete_plugin(pid: str) -> tuple[bytes, str, int]:
        with _crud_lock:
            plugins = load_plugins()
            new_plugins = [p for p in plugins if p.get("id") != pid]
            if len(new_plugins) == len(plugins):
                return AdminBackend._json_response({"ok": False, "error": f"Plugin '{pid}' not found"}, 404)
            result = save_plugins(new_plugins)
        return AdminBackend._json_response(result)

    @staticmethod
    def _reset_plugin_state(pid: str) -> tuple[bytes, str, int]:
        from core.engine.plugins import PluginEngine
        PluginEngine.reset_plugin_state(pid)
        return AdminBackend._json_response({"ok": True})

    # ── Speed Test ─────────────────────────────────────

    @staticmethod
    def _speedtest(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"error": "invalid json"}, 400)
        route_label = body.get("route", "")
        model = body.get("model", "")
        if not route_label or not model:
            return AdminBackend._json_response({"error": "route and model required"}, 400)

        route = get_route(route_label)
        if not route:
            return AdminBackend._json_response({"error": f"route '{route_label}' not found"}, 400)

        api_key = route.get("api_key", "").strip()
        base_url = route.get("base_url", "").strip()
        protocol = route.get("protocol", "anthropic")

        # Preprocess base_url — same as build_agent_from_route
        from core.http_utils import (preprocess_base_url, resolve_proxy_url,
                                     build_httpx_client, build_anthropic_messages_url,
                                     build_openai_chat_url)
        base_url = preprocess_base_url(base_url)

        # Proxy logic — mirror build_agent_from_route (pipeline.py lines 149-154)
        s = load_settings()
        proxy_config = s.get("proxy", {}) if isinstance(s.get("proxy"), dict) else {"address": s.get("proxy", "")}
        route_proxy_config = dict(proxy_config)
        if not route.get("use_proxy", True):
            route_proxy_config["use_for_api"] = False
        proxy_url = resolve_proxy_url(route_proxy_config) if route_proxy_config.get("use_for_api") else None

        result = {"status": "error", "ttft_ms": 0, "total_ms": 0, "tokens": 0, "error": ""}
        t0 = bj_epoch()

        if protocol == "anthropic":
            url = build_anthropic_messages_url(base_url)
            headers = {
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            # Realistic multi-turn payload matching actual call structure:
            # system + past turns (think/tool_use/tool_result) + tools + thinking
            speed_messages = [
                {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Simple arithmetic.", "signature": ""},
                    {"type": "text", "text": "Let me calculate that."},
                    {"type": "tool_use", "id": "speedtest_001", "name": "calculator",
                     "input": {"expression": "2+2"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "speedtest_001",
                     "content": [{"type": "text", "text": "4"}]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "The answer is 4."}]},
                {"role": "user", "content": [{"type": "text", "text": "Confirm the answer."}]},
            ]
            req_body: dict[str, Any] = {
                "model": model, "messages": speed_messages,
                "system": [{"type": "text", "text": "You are a helpful assistant. Respond concisely."}],
                "max_tokens": 50, "stream": True,
                "tools": [
                    {"name": "Bash", "description": "Execute bash command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
                    {"name": "Read", "description": "Read file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
                    {"name": "Write", "description": "Write file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
                ],
            }
            # Include thinking if the route has it enabled
            thinking_preset = route.get("thinking", "off")
            thinking_budget = {"off": 0, "low": 4000, "high": 16000, "max": 32000}.get(thinking_preset, 0)
            if thinking_budget:
                req_body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            try:
                with build_httpx_client(proxy_url, timeout=30, connect_timeout=10) as client:
                    with client.stream("POST", url, json=req_body, headers=headers) as resp:
                        if resp.status_code != 200:
                            result["error"] = f"HTTP {resp.status_code}"
                            result["total_ms"] = int((bj_epoch() - t0) * 1000)
                            return AdminBackend._json_response(result)
                        first_token = True
                        tokens = 0
                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if not payload or payload == "[DONE]":
                                continue
                            try:
                                event = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    tokens += 1
                                    if first_token:
                                        result["ttft_ms"] = int((bj_epoch() - t0) * 1000)
                                        first_token = False
                        result["status"] = "ok"
                        result["total_ms"] = int((bj_epoch() - t0) * 1000)
                        result["tokens"] = tokens
            except httpx.ConnectError:
                result["error"] = "ConnectError"
                result["total_ms"] = int((bj_epoch() - t0) * 1000)
            except Exception as e:
                result["error"] = str(e)[:100]
                result["total_ms"] = int((bj_epoch() - t0) * 1000)
        else:
            # OpenAI protocol
            url = build_openai_chat_url(base_url)
            headers = {
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            # Realistic payload matching actual call: system + past turns + tools + thinking
            speed_messages = [
                {"role": "system", "content": "You are a helpful assistant. Respond concisely."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Let me calculate that.",
                 "tool_calls": [{"id": "speedtest_001", "type": "function",
                                 "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}}],
                 "reasoning_content": "Simple arithmetic."},
                {"role": "tool", "tool_call_id": "speedtest_001", "content": "4"},
                {"role": "user", "content": "Confirm the answer."},
            ]
            req_body: dict[str, Any] = {
                "model": model, "messages": speed_messages,
                "stream": True,
                "tools": [
                    {"type": "function", "function": {"name": "Bash", "description": "Execute bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
                    {"type": "function", "function": {"name": "Read", "description": "Read file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
                    {"type": "function", "function": {"name": "Write", "description": "Write file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
                ],
            }
            _use_max_completion = any(
                model.lower().startswith(p) for p in ("o1", "o3", "o4", "gpt-5")
            )
            if _use_max_completion:
                req_body["max_completion_tokens"] = 50
            else:
                req_body["max_tokens"] = 50
            # Include thinking if the route has it enabled
            thinking_preset = route.get("thinking", "off")
            thinking_budget = {"off": 0, "low": 4000, "high": 16000, "max": 32000}.get(thinking_preset, 0)
            if thinking_budget:
                req_body["extra_body"] = {"thinking": {"type": "enabled"}}
                req_body["reasoning_effort"] = "max" if thinking_budget > 16000 else "high"
            try:
                with build_httpx_client(proxy_url, timeout=30, connect_timeout=10) as client:
                    with client.stream("POST", url, json=req_body, headers=headers) as resp:
                        if resp.status_code != 200:
                            result["error"] = f"HTTP {resp.status_code}"
                            result["total_ms"] = int((bj_epoch() - t0) * 1000)
                            return AdminBackend._json_response(result)
                        first_token = True
                        tokens = 0
                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if not payload or payload == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            for choice in chunk.get("choices", []) or []:
                                delta = (choice.get("delta") or {})
                                content = delta.get("content", "")
                                if content:
                                    tokens += 1
                                    if first_token:
                                        result["ttft_ms"] = int((bj_epoch() - t0) * 1000)
                                        first_token = False
                        result["status"] = "ok"
                        result["total_ms"] = int((bj_epoch() - t0) * 1000)
                        result["tokens"] = tokens
            except httpx.ConnectError:
                result["error"] = "ConnectError"
                result["total_ms"] = int((bj_epoch() - t0) * 1000)
            except Exception as e:
                result["error"] = str(e)[:100]
                result["total_ms"] = int((bj_epoch() - t0) * 1000)

        return AdminBackend._json_response(result)

    # ── Logs ────────────────────────────────────────────

    @staticmethod
    def _query_logs(query: str) -> tuple[bytes, str, int]:
        params = parse_qs(query)
        limit = 100
        try:
            limit = min(int(params.get("limit", ["100"])[0]), 500)
        except (ValueError, IndexError):
            pass
        since = params.get("since", [None])[0]
        status = params.get("status", [None])[0]
        chain = params.get("chain", [None])[0]
        return AdminBackend._json_response(query_logs(limit=limit, since=since,
                                                      status=status, chain=chain))

    @staticmethod
    def _log_detail(session_id: str) -> tuple[bytes, str, int]:
        detail = get_log_detail(session_id)
        if detail is None:
            return AdminBackend._json_response({"error": "Not found"}, 404)
        return AdminBackend._json_response(detail)

    # ── Debug logs ─────────────────────────────────────

    @staticmethod
    def _debug_dates() -> tuple[bytes, str, int]:
        from core.infra.debug import list_debug_sessions
        return AdminBackend._json_response({"sessions": list_debug_sessions()})

    @staticmethod
    def _debug_logs(query: str) -> tuple[bytes, str, int]:
        from core.infra.debug import query_debug_session
        params = parse_qs(query) if query else {}
        filename = params.get("file", [""])[0] or ""
        if not filename:
            return AdminBackend._json_response({"entries": [], "turns": 0})
        return AdminBackend._json_response(query_debug_session(filename))

    # ── Default SP ─────────────────────────────────────

    @staticmethod
    def _default_sp() -> tuple[bytes, str, int]:
        from core.prompt import build_system_prompt
        sp = build_system_prompt(".", "claude-sonnet-4-6")
        return AdminBackend._json_response({"system_prompt": sp})

    # ── File Manager ──────────────────────────────────

    @staticmethod
    def _resolve_safe_path(rel: str) -> Path | None:
        """Resolve a path within ROOT_DIR. Handles both absolute and relative."""
        if not rel:
            return ROOT_DIR
        p = Path(rel)
        if p.is_absolute():
            p = p.resolve()
        else:
            p = (ROOT_DIR / rel).resolve()
        root = ROOT_DIR.resolve()
        if not str(p).startswith(str(root)):
            return None
        return p

    @staticmethod
    def _rel_path(p: Path) -> str:
        """Return path relative to ROOT_DIR, with forward slashes."""
        try:
            rel = p.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
            return "" if rel == "." else rel
        except ValueError:
            return p.as_posix()

    @staticmethod
    def _list_files(query: str) -> tuple[bytes, str, int]:
        params = parse_qs(query) if query else {}
        rel = params.get("path", [""])[0] or ""
        target = AdminBackend._resolve_safe_path(rel)
        if target is None:
            return AdminBackend._json_response({"error": "Access denied"}, 403)
        if not target.exists():
            return AdminBackend._json_response({"error": "Path not found"}, 404)
        if not target.is_dir():
            return AdminBackend._json_response({"error": "Not a directory"}, 400)

        entries = []
        try:
            for child in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    st = child.stat()
                    entries.append({
                        "name": child.name,
                        "type": "dir" if child.is_dir() else "file",
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                    })
                except OSError:
                    pass
        except OSError:
            return AdminBackend._json_response({"error": "Cannot read directory"}, 500)

        return AdminBackend._json_response({
            "path": AdminBackend._rel_path(target),
            "entries": entries,
        })

    @staticmethod
    def _read_file(query: str) -> tuple[bytes, str, int]:
        params = parse_qs(query) if query else {}
        rel = params.get("path", [""])[0] or ""
        target = AdminBackend._resolve_safe_path(rel)
        if target is None:
            return AdminBackend._json_response({"error": "Access denied"}, 403)
        if not target.exists():
            return AdminBackend._json_response({"error": "Path not found"}, 404)
        if not target.is_file():
            return AdminBackend._json_response({"error": "Not a file"}, 400)

        # Image preview
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff"}
        mime_map = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
                    ".gif":"image/gif",".webp":"image/webp",".bmp":"image/bmp",
                    ".ico":"image/x-icon",".tiff":"image/tiff"}
        suffix = target.suffix.lower()
        if suffix in image_exts:
            try:
                raw = target.read_bytes()
                if len(raw) > 10 * 1024 * 1024:
                    return AdminBackend._json_response({
                        "path": AdminBackend._rel_path(target),
                        "content": "[图片过大，不支持预览]",
                        "size": target.stat().st_size, "type": "text",
                    })
                b64 = base64.b64encode(raw).decode("ascii")
                return AdminBackend._json_response({
                    "path": AdminBackend._rel_path(target),
                    "content": f"data:{mime_map.get(suffix,'image/png')};base64,{b64}",
                    "size": target.stat().st_size, "type": "image",
                })
            except OSError as e:
                return AdminBackend._json_response({"error": str(e)}, 500)

        # Detect if text file via extension
        text_exts = {".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".jsonl",
                     ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".csv", ".tsv",
                     ".sh", ".bat", ".ps1", ".log", ".env", ".gitignore", ".c", ".h",
                     ".cpp", ".java", ".go", ".rs", ".ts", ".tsx", ".jsx", ".rb",
                     ".php", ".sql", ".r", ".lua", ".vim", ".svg"}
        suffix = target.suffix.lower()
        if suffix not in text_exts:
            try:
                content = target.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                return AdminBackend._json_response({
                    "path": AdminBackend._rel_path(target),
                    "content": "[二进制文件，无法预览]",
                    "size": target.stat().st_size,
                    "truncated": False,
                })
        else:
            try:
                content = target.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                return AdminBackend._json_response({
                    "path": AdminBackend._rel_path(target),
                    "content": "[编码错误，无法预览]",
                    "size": target.stat().st_size,
                    "truncated": False,
                })

        lines = content.split("\n")
        # Pagination support
        try: offset = max(0, int(params.get("offset", ["0"])[0]))
        except (ValueError, IndexError): offset = 0
        try: limit = min(5000, max(100, int(params.get("limit", ["2000"])[0])))
        except (ValueError, IndexError): limit = 2000
        total_lines = len(lines)
        chunk = "\n".join(lines[offset:offset + limit])
        has_more = (offset + limit) < total_lines

        return AdminBackend._json_response({
            "path": AdminBackend._rel_path(target),
            "content": chunk,
            "size": target.stat().st_size,
            "total_lines": total_lines,
            "offset": offset,
            "has_more": has_more,
        })

    # ── File Rename ────────────────────────────────

    @staticmethod
    def _rename_file(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        rel = body.get("path", "")
        new_name = body.get("new_name", "")
        if not rel or not new_name:
            return AdminBackend._json_response({"ok": False, "error": "path and new_name required"}, 400)
        if "/" in new_name or "\\" in new_name:
            return AdminBackend._json_response({"ok": False, "error": "Invalid name"}, 400)
        target = AdminBackend._resolve_safe_path(rel)
        if target is None or not target.exists():
            return AdminBackend._json_response({"ok": False, "error": "Path not found"}, 404)
        new_path = target.parent / new_name
        if new_path.exists():
            return AdminBackend._json_response({"ok": False, "error": "Target name already exists"}, 409)
        try:
            safe_new = new_path.relative_to(ROOT_DIR)
            if AdminBackend._resolve_safe_path(str(safe_new)) is None:
                return AdminBackend._json_response({"ok": False, "error": "Access denied"}, 403)
        except ValueError:
            return AdminBackend._json_response({"ok": False, "error": "Access denied"}, 403)
        try:
            backup_file(target)
            target.rename(new_path)
            return AdminBackend._json_response({"ok": True, "new_path": AdminBackend._rel_path(new_path)})
        except OSError as e:
            return AdminBackend._json_response({"ok": False, "error": str(e)}, 500)

    # ── File Delete ────────────────────────────────

    @staticmethod
    def _delete_file(query: str) -> tuple[bytes, str, int]:
        params = parse_qs(query) if query else {}
        rel = params.get("path", [""])[0] or ""
        target = AdminBackend._resolve_safe_path(rel)
        if target is None:
            return AdminBackend._json_response({"ok": False, "error": "Access denied"}, 403)
        if not target.exists():
            return AdminBackend._json_response({"ok": False, "error": "Path not found"}, 404)
        try:
            backup_file(target)
            # Move to .trash/ preserving relative path to avoid collisions
            trash_dir = ROOT_DIR / "workspace" / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            rel = target.resolve().relative_to(ROOT_DIR.resolve())
            # Flatten path: workspace/sub/file.txt → workspace_sub_file.txt
            trash_name = str(rel).replace('\\', '_').replace('/', '_')
            trash_path = trash_dir / trash_name
            # If name collision, append timestamp
            if trash_path.exists():
                import time as _tm
                trash_path = trash_dir / (trash_name + '_' + str(int(_tm.time())))
            target.rename(trash_path)
            return AdminBackend._json_response({"ok": True})
        except OSError as e:
            return AdminBackend._json_response({"ok": False, "error": str(e)}, 500)

    # ── File Create ────────────────────────────────

    @staticmethod
    def _create_file(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        rel = body.get("path", "")
        name = body.get("name", "")
        ftype = body.get("type", "file")
        if not name:
            return AdminBackend._json_response({"ok": False, "error": "name required"}, 400)
        if "/" in name or "\\" in name:
            return AdminBackend._json_response({"ok": False, "error": "Invalid name"}, 400)
        parent = AdminBackend._resolve_safe_path(rel or "")
        if parent is None or not parent.is_dir():
            return AdminBackend._json_response({"ok": False, "error": "Parent not found"}, 404)
        new_path = parent / name
        if new_path.exists():
            return AdminBackend._json_response({"ok": False, "error": "Already exists"}, 409)
        try:
            new_path.relative_to(ROOT_DIR)
        except ValueError:
            return AdminBackend._json_response({"ok": False, "error": "Access denied"}, 403)
        try:
            if ftype == "dir":
                new_path.mkdir()
            else:
                new_path.touch()
            return AdminBackend._json_response({"ok": True, "path": AdminBackend._rel_path(new_path)})
        except OSError as e:
            return AdminBackend._json_response({"ok": False, "error": str(e)}, 500)

    # ── File Save ──────────────────────────────────

    @staticmethod
    def _save_file(body: dict | None) -> tuple[bytes, str, int]:
        if not isinstance(body, dict):
            return AdminBackend._json_response({"ok": False, "error": "Invalid JSON"}, 400)
        rel = body.get("path", "")
        content = body.get("content", "")
        if not rel:
            return AdminBackend._json_response({"ok": False, "error": "path required"}, 400)
        target = AdminBackend._resolve_safe_path(rel)
        if target is None:
            return AdminBackend._json_response({"ok": False, "error": "Access denied"}, 403)
        if not target.parent.exists():
            return AdminBackend._json_response({"ok": False, "error": "Parent directory not found"}, 404)
        backup_file(target)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(target)
            return AdminBackend._json_response({"ok": True, "size": target.stat().st_size})
        except OSError as e:
            return AdminBackend._json_response({"ok": False, "error": str(e)}, 500)
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except OSError: pass
