#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.4 — OpenWebUI API server with modular pipeline engine.
Usage:   python core/server.py  →  http://localhost:18787/v1
         (entry point is core/server.py shim)
"""
from __future__ import annotations

import base64, hashlib, json, sys, threading, time, uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

from core.paths import ROOT_DIR, LOG_DIR
sys.path.insert(0, str(ROOT_DIR))

from core.infra.logger import dump_owui, trace_log
from core.engine.pipeline import PipelineEngine
from core.engine.execution import execute as exec_execute, ExecRequest
from core.entry.admin import AdminBackend
from core.engine.plugins import PluginEngine
from core.codec.owui import extract_text, extract_content, make_text_chunk, make_stop_chunk
from core.infra.settings import get_route, load_settings
from core.infra.debug import DebugContext, is_debug_enabled, notify_activity
LOG_DIR.mkdir(parents=True, exist_ok=True)


_print_lock = threading.Lock()


def _log(msg: str, dest: str = ""):
    if not dest:
        dest = "owui" if (msg.startswith("REQ") or "400" in msg) else "model_resp"
    trace_log(msg, dest)
    with _print_lock:
        print(msg, flush=True)


def _describe_images(images: list[dict], settings: dict) -> str:
    """Call vision API for each image, return concatenated descriptions."""
    vc = settings.get("vision", {})
    if not vc or not vc.get("api_key"):
        return ""
    from urllib.request import Request as _Req
    from core.http_utils import resolve_proxy_url, build_urllib_opener
    endpoint = vc["base_url"].rstrip("/") + "/chat/completions"
    models = [m.strip() for m in (vc.get("model") or "GLM-4.6V-Flash").split(",") if m.strip()]
    if not models:
        models = ["GLM-4.6V-Flash"]
    proxy_url = resolve_proxy_url(vc) if vc.get("use_proxy") else None
    opener = build_urllib_opener(proxy_url)
    parts = []
    for idx, img in enumerate(images):
        mime = img.get("mime", "png")
        if mime == "jpg":
            mime = "jpeg"
        b64 = img["data"]
        body = {
            "model": models[0], "max_tokens": 800,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": "请用中文详细描述这张图片的内容。"},
            ]}],
        }
        try:
            req = _Req(endpoint,
                       data=json.dumps(body).encode("utf-8"),
                       headers={"Authorization": f"Bearer {vc['api_key']}", "Content-Type": "application/json"})
            resp = opener.open(req, timeout=60)
            data = json.loads(resp.read().decode("utf-8"))
            desc = data["choices"][0]["message"].get("content", "") or "(no description)"
            parts.append(f"[上传图片 {idx+1} 的描述: {desc.strip()}]")
        except Exception as e:
            parts.append(f"[图片 {idx+1} 描述失败: {e}]")
    return "\n".join(parts)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


# ── SSE Helpers ────────────────────────────────────────

def _emit_image_desc(chat_id: str, model: str, image_desc: str, wfile, wfile_lock: threading.Lock):
    """Emit pre-computed image description as a single <think> SSE chunk."""
    if not image_desc:
        return
    text = f"<think>\n{image_desc}\n</think>\n"
    try:
        with wfile_lock:
            wfile.write(
                f"data: {json.dumps(make_text_chunk(chat_id, model, text), ensure_ascii=False)}\n\n".encode())
            wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass


# ── HTTP Handler ───────────────────────────────────────

class AgentHTTPHandler(BaseHTTPRequestHandler):

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/admin"):
            self._delegate_admin("GET")
        elif path == "/v1/models":
            self._handle_v1_models()
        else:
            self._json({"error": {"message": "not found", "type": "not_found", "code": 404}}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/admin"):
            self._delegate_admin("POST")
        elif path == "/v1/chat/completions":
            self._handle_v1_chat_completions()
        else:
            self._json({"error": {"message": "not found", "type": "not_found", "code": 404}}, 404)

    def do_PUT(self):
        path = urlparse(self.path).path
        if path.startswith("/admin"):
            self._delegate_admin("PUT")
        else:
            self._json({"error": {"message": "not found", "type": "not_found", "code": 404}}, 404)

    def do_PATCH(self):
        path = urlparse(self.path).path
        if path.startswith("/admin"):
            self._delegate_admin("PATCH")
        else:
            self._json({"error": {"message": "not found", "type": "not_found", "code": 404}}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/admin"):
            self._delegate_admin("DELETE")
        else:
            self._json({"error": {"message": "not found", "type": "not_found", "code": 404}}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ── Admin delegation ─────────────────────────────

    def _check_admin_auth(self) -> bool:
        pwd = (load_settings().get("admin_password") or "").strip()
        if not pwd:
            return True
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="ignore")
            return decoded.split(":", 1)[-1] == pwd
        except Exception:
            return False

    def _delegate_admin(self, method: str):
        if not self._check_admin_auth():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Admin"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        parsed = urlparse(self.path)
        path = parsed.path
        query = parsed.query
        body = self._read_body() if method in ("POST", "PUT", "PATCH") else None

        resp_bytes, content_type, status = AdminBackend.handle_request(
            path, method, body, query)

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(resp_bytes)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # ── /v1/models ────────────────────────────────────

    def _handle_v1_models(self):
        self._json(PipelineEngine.build_model_list())

    # ── /v1/chat/completions ──────────────────────────

    def _handle_v1_chat_completions(self):
        body = self._read_body()
        if body is None:
            _log("CHAT 400: Invalid JSON body")
            return self._openai_error("Invalid JSON body", 400)

        messages = body.get("messages", [])
        if not messages:
            _log("CHAT 400: No messages")
            return self._openai_error("No messages in request", 400)

        model = body.get("model", "roleplay")

        # Extract user text + images/files from last user message
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = extract_text(m.get("content"))
                break
        if not user_text.strip():
            _log("CHAT 400: No user message")
            return self._openai_error("No user message found", 400)

        # Detect and process images/files in the last user message
        image_desc = ""
        try:
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user:
                content = extract_content(last_user.get("content"))
                images, files = content["images"], content["files"]
                if images:
                    s = load_settings()
                    desc = _describe_images(images, s)
                    if desc:
                        user_text = user_text + "\n\n" + desc
                        image_desc = desc
                if files:
                    ws = ROOT_DIR / "workspace" / "owui_uploads"
                    ws.mkdir(parents=True, exist_ok=True)
                    file_parts = []
                    for f in files:
                        fname = f.get("name", "unknown")
                        fpath = ws / fname
                        from core.fsutil import backup_file
                        backup_file(fpath)
                        fpath.write_bytes(base64.b64decode(f["data"]))
                        file_parts.append(f"[上传文件: {fname} → 已保存到 {fpath}]")
                    user_text = user_text + "\n\n" + "\n".join(file_parts)
        except Exception as e:
            _log(f"IMG/FILE process error: {e}", "owui")

        dump_owui(hashlib.sha256(user_text.encode()).hexdigest()[:12], messages)

        debug_enabled = is_debug_enabled()
        debug_ctx = DebugContext() if debug_enabled else None
        if debug_enabled:
            notify_activity()

        # ── Route: chain ──
        chain, session_id = PipelineEngine.resolve_chain(model)
        if chain is not None:
            # Build stable session ID from first user message + chain label
            for m in messages:
                if m.get("role") == "user":
                    first = extract_text(m.get("content"))
                    if first:
                        session_id = f"{session_id}_{hashlib.sha256(first.encode()).hexdigest()[:12]}"
                    break
            _log(f"REQ [{session_id}]: chain={chain.get('label')}, {len(messages)} msgs")
            if debug_ctx:
                debug_ctx.session_id = session_id
            # Roleplay chain with 0 side paths → single-agent path (preserves thinking)
            if not chain.get("side_paths"):
                route = get_route(chain.get("main_path", {}).get("route", ""))
                agent_model = chain.get("main_path", {}).get("model", "")
                main_cfg = dict(chain.get("main_path", {}))
                main_cfg["_show_history_stats"] = chain.get("show_history_stats", True)
                # Forward chain-level keep_turns and keep_think so single-agent path respects them
                for _k in ("keep_turns", "keep_think"):
                    if _k not in main_cfg:
                        main_cfg[_k] = chain.get(_k, -1)
                chain_vars = chain.get("variables", {})
                return self._handle_single_agent_chat(route, agent_model, session_id, user_text, messages, main_cfg, chain_vars, chain.get("separator", ""), image_desc, debug_ctx)
            return self._handle_chain_chat(chain, session_id, user_text, messages, image_desc, debug_ctx)

        # ── Route: {provider}/{model}-agent ──
        route, actual_model = PipelineEngine.resolve_single_agent(model)
        if route is not None:
            model_for_sid = actual_model.rsplit("-agent", 1)[0].strip()
            session_id = f"default_{model_for_sid}"
            for m in messages:
                if m.get("role") == "user":
                    first = extract_text(m.get("content"))
                    if first:
                        session_id = f"default_{hashlib.sha256(first.encode()).hexdigest()[:12]}"
                    break
            _log(f"REQ [{session_id}]: single-agent model={actual_model} route={route.get('label','?')}")
            if debug_ctx:
                debug_ctx.session_id = session_id
            return self._handle_single_agent_chat(route, actual_model, session_id, user_text, messages, None, None, "", image_desc, debug_ctx)

        # Fallback: try as single-agent with default route
        _log(f"REQ [default]: fallback model={model}")
        return self._handle_single_agent_chat(None, model, "default", user_text, messages, None, None, "", image_desc, debug_ctx)

    # ── Chain chat (delegates to PipelineEngine) ──────

    def _handle_chain_chat(self, chain: dict, session_id: str,
                           user_text: str, messages: list[dict], image_desc: str = "",
                           debug_ctx=None):
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = chain.get("label", "da")
        self._start_sse()

        wfile_lock = threading.Lock()
        heartbeat_stop = threading.Event()
        stop_event = threading.Event()
        stop_reason = [""]

        def _handle_write_error(e):
            # TCP socket errors can't reliably distinguish user clicking Stop
            # from browser closing a tab — they all tear down the connection.
            stop_reason[0] = "connection_lost"
            stop_event.set()

        hb_thread = threading.Thread(
            target=self._heartbeat, args=(heartbeat_stop, wfile_lock), daemon=True)
        hb_thread.start()

        think_open = False
        text_emitted = False
        just_closed = False

        def phase_set_cb(phase: str):
            pass

        def emit_cb(phase: str, text: str):
            nonlocal think_open, text_emitted, just_closed

            # Fix OWUI: = or - after </think> → Setext heading breaks rendering.
            # Match every possible split: = == === - -- --- etc.
            # Use "in" (not startswith) so mid-chunk </think> is also detected.
            # Don't consume just_closed on whitespace-only chunks.
            if "</think>" in text:
                just_closed = True
            elif "<think>" in text:
                just_closed = False
            elif just_closed:
                t = text.lstrip()
                if t:
                    if t.startswith('=') or t.startswith('-'):
                        text = "​\n\n" + text
                    just_closed = False

            if "<think>" in text and "</think>" not in text:
                think_open = True
            elif "</think>" in text and "<think>" not in text:
                think_open = False

            if text.strip() and not text.startswith("<think>") and \
               not text.startswith("</think>") and not text.startswith("=== ["):
                # Check it's not a pure tool output prefix
                if phase == "main" and not think_open:
                    text_emitted = True

            try:
                with wfile_lock:
                    self.wfile.write(
                        f"data: {json.dumps(make_text_chunk(chat_id, model_name, text), ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                _handle_write_error(e)
            except OSError:
                stop_event.set()

        _emit_image_desc(chat_id, model_name, image_desc, self.wfile, wfile_lock)

        # ── Plugin: user injection (before pipeline) ──
        route_label = chain.get("main_path", {}).get("route", "")
        agent_model = chain.get("main_path", {}).get("model", "")
        try:
            user_text = PluginEngine.apply_user_injection(user_text, route_label, agent_model)
        except Exception:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)

        result = PipelineEngine.execute_chain(
            chain, session_id, user_text, messages,
            emit_cb, phase_set_cb, stop_event, stop_reason=stop_reason[0], debug_ctx=debug_ctx)

        # ── Plugin: conditional events (after pipeline) ──
        try:
            assistant_text = result.get("assistant_text", "") if isinstance(result, dict) else ""
            PluginEngine.process_conditional_events(
                session_id, route_label, agent_model, user_text, assistant_text)
        except Exception:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)

        self._send_stop(chat_id, model_name, result, wfile_lock)
        heartbeat_stop.set()
        hb_thread.join(timeout=2)
        _log(f"DONE [{session_id}]: chain={chain.get('label')} "
             f"in={result.get('input',0)} out={result.get('output',0)}")
        self.close_connection = True

    # ── Single-agent chat ──────────────────────────────

    def _handle_single_agent_chat(self, route: dict | None, model: str,
                                  session_id: str, user_text: str,
                                  messages: list[dict], main_cfg: dict | None = None,
                                  chain_vars: dict | None = None,
                                  separator: str = "", image_desc: str = "",
                                  debug_ctx=None):
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self._start_sse()

        wfile_lock = threading.Lock()
        heartbeat_stop = threading.Event()
        stop_event = threading.Event()
        stop_reason = [""]

        def _handle_write_error(e):
            stop_reason[0] = "connection_lost"
            stop_event.set()

        hb_thread = threading.Thread(
            target=self._heartbeat, args=(heartbeat_stop, wfile_lock), daemon=True)
        hb_thread.start()

        just_closed = False

        def emit_cb(phase: str, text: str):
            nonlocal just_closed

            if "</think>" in text:
                just_closed = True
            elif "<think>" in text:
                just_closed = False
            elif just_closed:
                t = text.lstrip()
                if t:
                    if t.startswith('=') or t.startswith('-'):
                        text = "​\n\n" + text
                    just_closed = False

            try:
                with wfile_lock:
                    self.wfile.write(
                        f"data: {json.dumps(make_text_chunk(chat_id, model, text), ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                _handle_write_error(e)
            except OSError:
                stop_event.set()

        _emit_image_desc(chat_id, model, image_desc, self.wfile, wfile_lock)

        # ── Plugin: user injection (before pipeline) ──
        s_route_label = route.get("label", "") if route else ""
        try:
            user_text = PluginEngine.apply_user_injection(user_text, s_route_label, model)
        except Exception:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)

        exec_req = ExecRequest(
            session_id=session_id, user_text=user_text, messages=messages,
            route=route, agent_model=model,
            chain_cfg={"main_path": main_cfg, "variables": chain_vars or {}, "separator": separator} if main_cfg else None,
            separator=separator, image_desc=image_desc, debug_ctx=debug_ctx,
            stop_event=stop_event, stop_reason=stop_reason[0],
        )
        exec_result = exec_execute(exec_req, emit_cb)
        result = {
            "input": exec_result.usage.get("input", 0),
            "output": exec_result.usage.get("output", 0),
            "assistant_text": exec_result.text,
        }

        # ── Plugin: conditional events (after pipeline) ──
        try:
            PluginEngine.process_conditional_events(
                session_id, s_route_label, model, user_text, exec_result.text)
        except Exception:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)

        self._send_stop(chat_id, model, result, wfile_lock)
        heartbeat_stop.set()
        hb_thread.join(timeout=2)
        _log(f"DONE [{session_id}]: single model={model} "
             f"in={result.get('input',0)} out={result.get('output',0)}")
        self.close_connection = True

    # ── SSE Helpers ────────────────────────────────────

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _heartbeat(self, stop_event: threading.Event, lock: threading.Lock):
        while not stop_event.wait(timeout=15):
            try:
                with lock:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
            except Exception:
                break

    def _send_stop(self, chat_id: str, model: str, usage: dict,
                   lock: threading.Lock):
        try:
            with lock:
                self.wfile.write(
                    f"data: {json.dumps(make_stop_chunk(chat_id, model, usage), ensure_ascii=False)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except Exception:
            pass

    # ── HTTP Helpers ───────────────────────────────────

    MAX_BODY = 10 * 1024 * 1024  # 10 MB

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                if length > self.MAX_BODY:
                    return None
                self.connection.settimeout(30)
                raw = self.rfile.read(length)
                return json.loads(raw)
        except Exception:
            pass
        return None

    def _json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _openai_error(self, message: str, code: int = 400):
        self._json({
            "error": {"message": message, "type": "invalid_request_error", "code": code},
        }, code)


# Entry point → see core/server.py (backward-compat shim)
