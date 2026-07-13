"""MiniAgent — Agent loop engine. Multi-protocol (Anthropic / OpenAI)."""
from __future__ import annotations

import fnmatch, hashlib, html, json, os, re, shutil, subprocess, threading, time as _time, urllib.parse, urllib.request, uuid
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Callable

from core.paths import WORKSPACE as _SHARED_WORKSPACE
from core.prompt import build_system_prompt
from core.timeutil import bj_now, bj_epoch
from core.drivers import get_driver
from core.drivers.base import ModelRequest, ThinkingConfig


def _sys_encoding() -> str:
    """Bash output encoding. Git Bash always outputs UTF-8 regardless of system locale."""
    return "utf-8"


def _safe_event_data(data: dict) -> dict:
    """Truncate large string values in event data for debug JSON storage."""
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 2000:
            out[k] = v[:2000] + f"...[{len(v)}]"
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════

def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "Bash", "description": "Execute a shell command. Restricted to workspace directory. Absolute paths outside workspace are blocked.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer", "minimum": 1, "description": "Timeout in ms"},
                    "description": {"type": "string", "description": "What this command does"},
                },
                "required": ["command"], "additionalProperties": False,
            },
        },
        {
            "name": "Read", "description": "Read a file. Returns line-numbered content.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["path"], "additionalProperties": False,
            },
        },
        {
            "name": "Write", "description": "Write or overwrite a file in the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"], "additionalProperties": False,
            },
        },
        {
            "name": "Edit", "description": "Replace text in a file (exact string match).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "old_string": {"type": "string", "description": "Text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
                },
                "required": ["path", "old_string", "new_string"], "additionalProperties": False,
            },
        },
        {
            "name": "Glob", "description": "Find files by glob pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"], "additionalProperties": False,
            },
        },
        {
            "name": "Grep", "description": "Search file contents with regex.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in"},
                    "glob": {"type": "string", "description": "Filter files by glob"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                    "-i": {"type": "boolean"},
                    "head_limit": {"type": "integer", "minimum": 1},
                },
                "required": ["pattern"], "additionalProperties": False,
            },
        },
        {
            "name": "WebSearch", "description": "Search the web. Uses Tavily when API key configured, falls back to DuckDuckGo.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2},
                    "allowed_domains": {"type": "array", "items": {"type": "string"}},
                    "blocked_domains": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"], "additionalProperties": False,
            },
        },
        {
            "name": "WebFetch", "description": "Fetch a URL and extract readable text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "prompt": {"type": "string", "description": "What to extract from the page"},
                },
                "required": ["url"], "additionalProperties": False,
            },
        },
        {
            "name": "TavilyExtract", "description": "Extract clean content from one or more URLs. Returns structured text for each URL.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string", "format": "uri"}, "description": "List of URLs to extract content from"},
                    "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "basic is faster/cheaper. advanced for complex pages."},
                },
                "required": ["urls"], "additionalProperties": False,
            },
        },
        {
            "name": "TavilyCrawl", "description": "Crawl a website starting from a URL. Maps the site structure then extracts content from discovered pages.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri", "description": "Starting URL to crawl from"},
                    "max_pages": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max pages to crawl. Default 10."},
                    "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "basic is faster/cheaper."},
                },
                "required": ["url"], "additionalProperties": False,
            },
        },
        {
            "name": "TavilyMap", "description": "Map a website's structure — discover all pages and subdomains linked from a URL.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "max_links": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max links to return. Default 20."},
                },
                "required": ["url"], "additionalProperties": False,
            },
        },
        {
            "name": "Delete", "description": "Delete a file in the workspace. Creates a backup before deleting.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to delete"},
                },
                "required": ["path"], "additionalProperties": False,
            },
        },
        {
            "name": "TaskCreate", "description": "Create a new task for tracking work.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Brief title"},
                    "description": {"type": "string", "description": "What needs to be done"},
                    "activeForm": {"type": "string", "description": "Present continuous form"},
                },
                "required": ["subject", "description"], "additionalProperties": False,
            },
        },
        {
            "name": "TaskUpdate", "description": "Update a task's status or details.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]},
                },
                "required": ["taskId"], "additionalProperties": False,
            },
        },
        {
            "name": "TaskList", "description": "List all tasks.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "TaskGet", "description": "Get full details of a task by ID.",
            "input_schema": {
                "type": "object",
                "properties": {"taskId": {"type": "string"}},
                "required": ["taskId"], "additionalProperties": False,
            },
        },
        {
            "name": "SubAgent",
            "description": "Delegate a task to a sub-agent. The sub-agent sees the workspace and can use tools (except SubAgent itself). Use this for parallel subtasks or when you need a fresh context window.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task for the sub-agent to complete"},
                    "model": {"type": "string", "description": "Model to use for the sub-agent. Defaults to the main agent's model."},
                    "provider": {"type": "string", "description": "Provider name for the sub-agent. Defaults to the current provider."},
                    "max_tokens": {"type": "integer", "minimum": 1024, "description": "Max output tokens for sub-agent. Default 32000."},
                },
                "required": ["prompt"], "additionalProperties": False,
            },
        },
        {
            "name": "DescribeImage",
            "description": "Analyze an image file using a vision model. Returns a text description of the image contents. Supports jpg, png, gif, webp, bmp.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the image file to analyze"},
                    "prompt": {"type": "string", "description": "What to look for in the image. Default: describe the image in detail."},
                },
                "required": ["path"], "additionalProperties": False,
            },
        },
    ]



# ═══════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════

class MiniAgent:
    def __init__(
        self,
        workspace: str | Path = ".",
        model: str = "deepseek-v4-pro",
        max_iterations: int = 20,
        thinking_budget: int = 0,
        max_tokens: int = 8192,
        api_key: str = "",
        base_url: str = "",
        custom_md_text: str = "",
        proxy_url: str = "",
        allow_bash: bool = False,
        vision_config: dict[str, Any] | None = None,
        protocol: str = "anthropic",
    ):
        self.protocol = protocol
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_iterations = max_iterations
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.proxy_url = proxy_url
        self.allow_bash = allow_bash
        self.custom_md_text = custom_md_text
        self._provider_config: dict[str, Any] = {}
        self._vision_config: dict[str, Any] = vision_config or {}
        self._proxy_config: dict[str, Any] = {}  # {"address":"...","use_for_api":bool,"use_for_websearch":bool}
        self._read_root: Path = _SHARED_WORKSPACE.resolve()  # default: workspace root
        self.system_prompt_text = build_system_prompt(self.workspace, self.model, custom_md_text)

        self._tools_anthropic = build_tools()

        self.conversation: list[dict[str, Any]] = []
        self.token_usage: dict[str, int] = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        self._turn = 0

        self._task_path = self.workspace / ".tasks.json"

        self._msg_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._web_opener: Any = None
        self.debug_ctx: Any = None  # set by pipeline when debug enabled

    # ── Public API ──────────────────────────────────────────

    def set_provider_config(self, config: dict) -> None:
        self._provider_config = config

    def set_proxy_config(self, config: dict) -> None:
        self._proxy_config = config

    def set_stop_event(self, event) -> None:
        self._stop_event = event

    def clear_tools(self) -> None:
        self._tools_anthropic = []

    def remove_tool(self, name: str) -> None:
        """Remove a tool by name. Used by SubAgent to prevent recursive SubAgent spawning."""
        self._tools_anthropic = [t for t in self._tools_anthropic
                                 if t.get("name") != name]

    def set_conversation(self, conv: list[dict], turn: int) -> None:
        with self._msg_lock:
            self.conversation = conv
            self._turn = turn

    def clear_conversation(self, keep_user_messages: int = 2):
        """Clean old rounds to text-only, preserving the last N user-initiated segments with full records (tool_use/tool_result/thinking)."""
        with self._msg_lock:
            user_indices = [i for i, m in enumerate(self.conversation)
                           if m.get("role") == "user" and isinstance(m.get("content"), str)]
            if len(user_indices) <= keep_user_messages:
                return
            boundary = user_indices[-keep_user_messages]
            for i in range(boundary):
                msg = self.conversation[i]
                content = msg.get("content")
                if isinstance(content, list):
                    text_blocks = [b for b in content if b.get("type") == "text"]
                    msg["content"] = text_blocks if text_blocks else [{"type": "text", "text": ""}]
            self.clean_orphaned_tool_results()

    # ── Public API: session control (P4 tool boundary) ────

    @property
    def last_raw_blocks(self) -> list[dict] | None:
        """Last round's raw content blocks (read-only)."""
        return getattr(self, "_last_raw_blocks", None)

    def set_turn(self, n: int) -> None:
        """Set turn counter to a specific value."""
        self._turn = n

    @property
    def turn(self) -> int:
        """Current turn counter (read-only)."""
        return self._turn

    def is_stopped(self) -> bool:
        """Check if stop has been requested (by user or timeout)."""
        return self._stop_event.is_set()

    def clear_all(self) -> None:
        """Clear entire conversation + reset turn counter."""
        with self._msg_lock:
            self.conversation.clear()
            self._turn = 0

    def handle_message(
        self,
        user_input: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> str:
        """Process one user message. Returns final display text."""
        self._stop_event.clear()
        self._turn += 1
        conv_start = len(self.conversation)
        with self._msg_lock:
            self.conversation.append({
                "role": "user", "content": user_input,
                "turn": self._turn, "turn_time": bj_now().isoformat(),
            })

        final_text = ""
        tool_rounds = 0

        while True:
            tool_rounds += 1
            if tool_rounds > self.max_iterations:
                final_text = "[Agent loop exceeded maximum iterations]"
                break

            # Guard against context overflow
            conv_chars = sum(
                len(str(m.get("content", ""))) for m in self.conversation
            )
            ctx_limit = 2_000_000 if self.max_tokens >= 128000 else 500_000
            if conv_chars > ctx_limit:
                final_text = f"[Context limit reached ({conv_chars} chars). Start a new chat or narrow the search scope.]"
                break

            try:
                text, content_blocks = self._call_model(on_event)
                # Auto-close unbalanced <thinking> tags in each text block.
                # M3 often opens <thinking> without closing — COT bleeds into body.
                for b in (content_blocks or []):
                    if b.get("type") == "text":
                        diff = b["text"].count("<thinking>") - b["text"].count("</thinking>")
                        if diff > 0:
                            b["text"] += "</thinking>" * diff
            except Exception:
                if not self._stop_event.is_set():
                    with self._msg_lock:
                        del self.conversation[conv_start:]
                raise

            if self._stop_event.is_set():
                break

            tool_uses = [b for b in (content_blocks or []) if b.get("type") == "tool_use"]

            if on_event:
                for tu in tool_uses:
                    on_event("tool_use", {"id": tu["id"], "name": tu["name"], "input": tu["input"]})

            if not tool_uses:
                with self._msg_lock:
                    final_text = text
                    self.conversation.append({
                        "role": "assistant", "content": content_blocks, "turn": self._turn,
                    })
                break

            with self._msg_lock:
                pre_round = list(self.conversation)
                self.conversation.append({
                    "role": "assistant", "content": content_blocks, "turn": self._turn,
                })
                try:
                    tool_results: list[dict[str, Any]] = [{} for _ in tool_uses]
                    # Parallel execution via threads for independent tool calls
                    threads = []
                    for i, tu in enumerate(tool_uses):
                        def _run(idx=i, name=tu["name"], inp=tu["input"], tu_id=tu["id"]):
                            tool_results[idx] = {
                                "type": "tool_result",
                                "tool_use_id": tu_id,
                                "content": "",
                                "is_error": True,
                            }
                            try:
                                r = self._execute_tool(name, inp)
                                tool_results[idx] = {
                                    "type": "tool_result",
                                    "tool_use_id": tu_id,
                                    "content": r["content"],
                                    "is_error": r["is_error"],
                                }
                                if "model" in r:
                                    tool_results[idx]["model"] = r["model"]
                            except Exception as e:
                                tool_results[idx] = {
                                    "type": "tool_result",
                                    "tool_use_id": tu_id,
                                    "content": f"Tool {name} error: {e}",
                                    "is_error": True,
                                }
                        t = threading.Thread(target=_run, daemon=True)
                        threads.append(t)
                        t.start()
                    for t in threads:
                        deadline = bj_epoch() + 300
                        while t.is_alive() and bj_epoch() < deadline:
                            t.join(timeout=1)
                            if self._stop_event.is_set():
                                break
                    if self._stop_event.is_set():
                        break
                    # Replace any bare {} from timed-out threads with proper error results
                    for i, tr in enumerate(tool_results):
                        if tr == {}:
                            tool_results[i] = {
                                "type": "tool_result",
                                "tool_use_id": tool_uses[i]["id"],
                                "name": tool_uses[i]["name"],
                                "content": "(timeout)",
                                "is_error": True,
                            }
                    # Emit tool_result events in order
                    for tr in tool_results:
                        if on_event:
                            tu_match = next((t for t in tool_uses if t["id"] == tr.get("tool_use_id")), None)
                            on_event("tool_result", {
                                "tool_use_id": tr.get("tool_use_id", ""),
                                "name": tu_match["name"] if tu_match else "?",
                                "content": tr.get("content", ""),
                                "is_error": tr.get("is_error", False),
                                "model": tr.get("model", ""),
                            })
                    for tr in tool_results:
                        tr.pop("model", None)
                    self.conversation.append({
                        "role": "user", "content": tool_results, "turn": self._turn,
                    })
                except Exception:
                    self.conversation[:] = pre_round
                    raise

            if self._stop_event.is_set():
                break

        # Save raw blocks before cleanup (for UA store lookup).
        # Skip conv_start[0] (user message) — pipeline rebuilds it separately.
        self._last_raw_blocks = list(self.conversation[conv_start + 1:])

        # Clean conversation to pure user↔assistant text — no thinking/tool blocks
        with self._msg_lock:
            entries = self.conversation[conv_start:]
            user_msg = entries[0]
            # Find last assistant with text, strip thinking/tool_use blocks
            last_text = None
            for m in reversed(entries):
                if m.get("role") == "assistant":
                    content = m.get("content", [])
                    if isinstance(content, list):
                        text_blocks = [b for b in content if b.get("type") == "text"]
                        if text_blocks:
                            last_text = {"role": "assistant", "content": text_blocks, "turn": m.get("turn")}
                    break
            self.conversation[conv_start:] = [user_msg] + ([last_text] if last_text else [])

        return final_text

    # ── Model Call ──────────────────────────────────────────

    def _build_model_request(self) -> ModelRequest:
        """Build a ModelRequest from current agent state."""
        thinking = None
        if self.thinking_budget:
            thinking = ThinkingConfig(enabled=True, budget_tokens=self.thinking_budget)
        return ModelRequest(
            model=self.model,
            messages=list(self.conversation),
            system_prompt=self.system_prompt_text,
            max_tokens=self.max_tokens,
            protocol=self.protocol,
            api_key=self.api_key,
            base_url=self.base_url,
            proxy_config=getattr(self, "_proxy_config", None),
            tools=self._tools_anthropic,
            thinking=thinking,
        )

    def _call_model(self, on_event=None):
        debug_events: list[dict] = []
        if self.debug_ctx:
            self.debug_ctx.record_model_request(
                self.protocol, self.model,
                list(self.conversation),
                {"thinking_budget": self.thinking_budget, "max_tokens": self.max_tokens,
                 "max_iterations": self.max_iterations,
                 "system_prompt": self.system_prompt_text})
            def _wrap(etype, data):
                debug_events.append({"type": etype, "data": _safe_event_data(data)})
                if on_event:
                    on_event(etype, data)
            _on = _wrap
        else:
            _on = on_event

        req = self._build_model_request()
        driver = get_driver(self.protocol)

        last_err = None
        for attempt in range(3):
            try:
                resp = driver.stream(req, _on)
                # Aggregate per-call usage
                for k in ("in", "out", "cache_read", "cache_write"):
                    self.token_usage[k] += resp.usage.get(k, 0)
                if self.debug_ctx:
                    self.debug_ctx.record_model_response(
                        self.protocol, debug_events,
                        {"input": self.token_usage["in"], "output": self.token_usage["out"],
                         "cache_read": self.token_usage["cache_read"], "cache_write": self.token_usage["cache_write"]})
                return resp.text, resp.blocks
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                try:
                    from core.infra.logger import log_model_req, trace_log
                    log_model_req(self.protocol, {"ts": bj_epoch(), "type": "response", "status": "error",
                             "error": str(e)[:500], "attempt": attempt + 1,
                             "error_type": type(e).__name__,
                             "error_full": repr(e)[:1000]})
                    trace_log(
                        f"MODEL_ERR attempt={attempt+1}/3 {type(e).__name__}: {str(e)[:300]}",
                        dest="model_req")
                except Exception:
                    pass
                # Skip retry for auth/perm errors and malformed requests.
                # 400 errors mean the request itself is invalid (missing fields,
                # wrong format) — retrying won't help.
                # 402 from DeepSeek is often transient (misleading "Insufficient
                # Balance") so we DO retry those.
                if any(k in msg for k in ('400', '401', '403', 'invalid api key', 'auth')):
                    raise
                if self._stop_event.is_set():
                    raise
                if attempt < 2:
                    if on_event:
                        on_event("retry", {"attempt": attempt + 2, "max": 3,
                                           "reason": str(last_err)[:200]})
                    _time.sleep(1 + attempt)
        raise last_err

    # ── Tool Execution ─────────────────────────────────────

    def _execute_tool(self, name: str, inp: dict[str, Any]) -> dict[str, Any]:
        try:
            fn = getattr(self, f"_tool_{name.lower()}", None)
            if fn is None:
                return {"content": f"Unknown tool: {name}", "is_error": True}
            return fn(inp)
        except Exception as exc:
            return {"content": f"Tool {name} failed: {exc}", "is_error": True}

    _UNIX_ROOT_PREFIXES = [
        "/root/workspace/", "/root/", "/workspace/", "/tmp/", "/mnt/", "/home/",
    ]

    def _resolve(self, raw: str) -> Path:
        # Model often sends Unix paths like /root/x, /workspace/x, /root/workspace/x.
        # Strip known Unix prefix cruft and treat the rest as a relative path.
        root = self._read_root
        if raw.startswith("/") and not raw.startswith("//"):
            # Handle /home/user/rest → rest
            if raw.startswith("/home/") and raw.count("/") >= 3:
                parts = raw.split("/")
                raw = "/" + "/".join(parts[3:])
            for prefix in self._UNIX_ROOT_PREFIXES:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
                    break
            # Remaining path → workspace relative (or just filename as fallback)
            p = Path(raw)
            if p.parts and p.parts[0] != "..":
                candidate = (root / p).resolve()
                if candidate.exists():
                    return self._check_read_root(candidate, root)
                # Fallback: search shared workspace for read access
                shared = (_SHARED_WORKSPACE / p).resolve()
                if shared.exists():
                    return self._check_read_root(shared, root)
            # Last resort: search by filename (own workspace first, then shared)
            name = Path(raw).name
            if name:
                matches = list(root.rglob(name))
                if len(matches) == 1:
                    return self._check_read_root(matches[0].resolve(), root)
                matches = list(_SHARED_WORKSPACE.rglob(name))
                if len(matches) == 1:
                    return self._check_read_root(matches[0].resolve(), root)
                return self._check_read_root(root / name, root)
        # Strip workspace prefix if model accidentally included it.
        # The model may pass "workspace/MiniMax-M3/x" (relative to project
        # root) or just "MiniMax-M3/x" (relative to global workspace).
        import core.paths as _paths
        for _base in (root.resolve(), _paths.ROOT_DIR.resolve()):
            try:
                _ws_rel = str(self.workspace.resolve().relative_to(_base))
                _pfx = _ws_rel.replace("\\", "/")
                _raw = raw.replace("\\", "/")
                if _raw == _pfx:
                    raw = "."
                    break
                if _raw.startswith(_pfx + "/"):
                    raw = _raw[len(_pfx) + 1:]
                    break
            except ValueError:
                pass
        p = Path(raw)
        if p.is_absolute():
            return self._check_read_root(p.resolve(), root)
        own = (self.workspace / p).resolve()
        if own.exists():
            return self._check_read_root(own, root)
        candidate = (root / p).resolve()
        if candidate.exists():
            return self._check_read_root(candidate, root)
        shared = (_SHARED_WORKSPACE / p).resolve()
        if shared.exists():
            return self._check_read_root(shared, root)
        return self._check_read_root(own, root)

    @staticmethod
    def _check_read_root(path: Path, root: Path) -> Path:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            return root / "__denied__"

    def _check_write(self, path: Path) -> dict[str, Any] | None:
        try:
            path.resolve().relative_to(self.workspace)
            return None
        except ValueError:
            return {"content": f"Permission denied: outside workspace/", "is_error": True}

    # ── Bash Safety ─────────────────────────────────────────

    def _save_large_output(self, content: str) -> str:
        """Persist oversized tool output to disk. Returns file path."""
        d = self.workspace / ".agent_outputs"
        d.mkdir(parents=True, exist_ok=True)
        ts = bj_now().strftime("%Y%m%d_%H%M%S")
        fname = f"tool_output_{ts}_{uuid.uuid4().hex[:6]}.txt"
        path = d / fname
        path.write_text(content, encoding="utf-8")
        return str(path)

    @staticmethod
    def _truncate_middle(text: str, head: int = 3000, tail: int = 2000) -> str:
        """Truncate text keeping head and tail, with marker in between."""
        if len(text) <= head + tail:
            return text
        return (
            text[:head]
            + f"\n\n... [{len(text) - head - tail} chars truncated, see full output file] ...\n\n"
            + text[-tail:]
        )

    _DANGEROUS_BASH_COMMON = [
        r'\brm\s+-[rR][fR]*\s+[/~]',
        r'\brm\s+-[rR][fR]*\s+\.(?:[/\\\s]|$)',
        r'\bshutdown\b', r'\breboot\b',
        r'\bgit\s+push\s+.*--force',
        r'\bgit\s+reset\s+--hard',
        r'\brmtree\b', r'\bshutil\.rmtree\b',
    ]
    _DANGEROUS_BASH_POSIX = [
        r'>\s*/dev/', r'\bmkfs\.', r'\bdd\s+if=', r'\bfdisk\b',
    ]
    _DANGEROUS_BASH_NT = [
        r'\bdel\s+(/[fFsSqQ]\s+)*[A-Za-z]:[/\\]',
        r'\brmdir\s+(/[sSqQ]\s+)*[A-Za-z]:[/\\]',
        r'\bformat\b',
        r'\bRemove-Item\s+-Recurse\s+[A-Za-z]:',
        r'\btaskkill\b.*\b/IM\b',
    ]
    _DANGEROUS_BASH = _DANGEROUS_BASH_COMMON + (_DANGEROUS_BASH_NT if os.name == "nt" else _DANGEROUS_BASH_POSIX)

    def _check_bash_safety(self, command: str) -> str | None:
        lower = command.lower()
        for pattern in self._DANGEROUS_BASH:
            if re.search(pattern, command) or re.search(pattern, lower):
                return f"Command blocked: matched '{pattern}'"
        # Block absolute paths pointing outside read_root.
        # Handle quoted paths (with spaces) and unquoted paths.
        root = str(self._read_root.resolve())
        for m in re.finditer(r'"([A-Z]:[\\/][^"]+)"|([A-Z]:[\\/]\S+)', command, re.IGNORECASE):
            p_str = m.group(1) or m.group(2)
            p = Path(p_str)
            try:
                if p.is_absolute():
                    rp = str(p.resolve()) if p.exists() else p_str
                    if not rp.lower().startswith(root.lower()):
                        return f"Blocked: path '{p_str}' is outside read root"
            except Exception:
                pass
        # Block Unix absolute paths (e.g. /etc/passwd) and home-directory paths (e.g. ~/secret)
        for m in re.finditer(r'(/(?:[^\s"\';&|><]+)?)', command):
            p_str = m.group(1)
            # Skip double-slash (network) and single-slash-only matches
            if p_str.startswith("//") or p_str == "/":
                continue
            p = Path(p_str)
            try:
                if p.is_absolute():
                    rp = str(p.resolve()) if p.exists() else p_str
                    if not rp.lower().startswith(root.lower()):
                        return f"Blocked: path '{p_str}' is outside read root"
            except Exception:
                pass
        for m in re.finditer(r'(~[^\s"\';&|><]*)', command):
            return f"Blocked: home-directory path '{m.group(1)}' may access outside read root"
        # Block relative path traversal beyond read_root
        for m in re.finditer(r'(\.\.[\\/])', command):
            try:
                test = self._read_root / (m.group(0) + "x")
                test.resolve().relative_to(self._read_root)
            except ValueError:
                return f"Blocked: '..' escapes read root"
            except Exception:
                pass
        # Block env-var paths that may point outside workspace
        env_paths = re.findall(r'%([\w()]+)%[\\/][^\s"\';&|><]*', command)
        blocked_env = {"USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
                       "WINDIR", "SYSTEMROOT", "SYSTEMDRIVE", "HOMEDRIVE",
                       "HOMEPATH", "PROGRAMFILES", "PROGRAMDATA",
                       "ALLUSERSPROFILE", "PUBLIC", "ONEDRIVE", "COMSPEC",
                       "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)",
                       "PROGRAMFILES(X86)"}
        for var in env_paths:
            if var.upper() in blocked_env or var.upper().rstrip(")") in blocked_env:
                return f"Blocked: %{var}% may point outside workspace"
        # Block File I/O targeting code/config outside workspace
        for m in re.finditer(r"""['\"]((?:[.]{0,2}[\\/])*(?:server|prompt|drivers|agent|admin|settings)[^'\"]*\.(?:py|json|bat))['\"]""", command):
            return f"Blocked: '{m.group(1)}' — code files outside workspace are read-only. Use Read tool to view, Write/Edit to modify workspace files."
        return None

    def _tool_bash(self, inp: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_bash:
            return {"content": "Bash disabled by settings (allow_bash: false)", "is_error": True}
        command = inp["command"]
        timeout_ms = inp.get("timeout", 120_000)
        warning = self._check_bash_safety(command)
        if warning:
            return {"content": warning, "is_error": True}
        # Backup files not yet saved to .backup/ before bash executes
        m = self._load_manifest()
        pre_changed = self._scan_and_backup(m)
        pre_keys = set(m.keys())  # snapshot of manifest keys before bash
        try:
            bash_path = shutil.which("bash")  # Git Bash on Windows; None on Unix (uses default)
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"  # Force Python UTF-8 stdout on Windows pipes
            result = subprocess.run(
                command, shell=True, capture_output=True,
                cwd=str(self.workspace),
                timeout=timeout_ms / 1000.0,
                encoding=_sys_encoding(),
                errors="replace",
                executable=bash_path or None,
                env=env,
            )
            # Post-execution: snapshot changed/created files, detect deletions
            deleted: list[str] = []
            for f in self.workspace.rglob("*"):
                if not f.is_file() or f.name == ".file_state.json":
                    continue
                if ".backup" in f.parts or ".agent_outputs" in f.parts:
                    continue
                key = self._manifest_key(f)
                entry = m.get(key)
                try:
                    st = f.stat()
                    if not entry or st.st_mtime != entry["mtime"] or st.st_size != entry["size"]:
                        self._snapshot_file(f, m)
                except OSError:
                    pass
            # Detect deleted files (in manifest before bash, not on disk after)
            for key in pre_keys:
                if not (self.workspace / key).exists():
                    m.pop(key, None)
                    deleted.append(key)
            self._save_manifest(m)
            out = result.stdout or ""
            if result.stderr:
                out += "\n[stderr]\n" + result.stderr
            if result.returncode != 0:
                out += f"\n[exit code: {result.returncode}]"
            out = out.strip() or "(no output)"
            parts: list[str] = []
            if pre_changed:
                parts.append(f"backed up {len(pre_changed)} file(s)")
            if deleted:
                parts.append(f"deleted {len(deleted)}: {', '.join(deleted)}")
            if parts:
                out = f"[{' | '.join(parts)}]\n{out}"
            if len(out) > 8000:
                path = self._save_large_output(out)
                out = self._truncate_middle(out) + f"\n\n[Full output saved to: {path}]"
            return {"content": out, "is_error": result.returncode != 0}
        except subprocess.TimeoutExpired:
            return {"content": f"Command timed out after {timeout_ms}ms", "is_error": True}

    def _tool_read(self, inp: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(inp["path"])
        disp = self._rel_path(path)
        if not path.exists():
            return {"content": f"File not found: {disp}", "is_error": True}
        if path.is_dir():
            return {"content": f"Path is a directory: {disp}", "is_error": True}
        fsize = path.stat().st_size
        if fsize > 500_000 and not inp.get("offset") and not inp.get("limit"):
            return {"content": f"File too large ({fsize:,} bytes). Use offset/limit.", "is_error": True}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, PermissionError, OSError):
            return {"content": f"Cannot read as text: {disp}", "is_error": True}
        offset = inp.get("offset", 0)
        limit = min(inp.get("limit", 500), 500)
        sliced = lines[offset:][:limit]
        numbered = "\n".join(f"{i + 1 + offset:<6}{line}" for i, line in enumerate(sliced))
        if len(numbered) > 50_000:
            path = self._save_large_output(numbered)
            numbered = self._truncate_middle(numbered) + f"\n\n[Full output: {path}]"
        elif len(sliced) < len(lines) - offset:
            numbered += f"\n\n... [showing {len(sliced)} lines, {len(lines)} total. Use offset={offset+limit}]"
        return {"content": numbered if numbered else "(empty file)", "is_error": False}

    def _tool_write(self, inp: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(inp["path"])
        disp = self._rel_path(path)
        if (err := self._check_write(path)):
            return err
        if path.is_dir():
            return {"content": f"Is a directory: {disp}", "is_error": True}
        existed = path.exists()
        bak = self._backup_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = inp["content"]
        path.write_text(content, encoding="utf-8")
        m = self._load_manifest()
        self._snapshot_file(path, m, backed=True)
        self._save_manifest(m)
        msg = f"Wrote {len(content)} bytes to {disp}" + (" (backed up)" if bak else "") + (" [overwritten]" if existed else "")
        return {"content": msg, "is_error": False}

    def _tool_edit(self, inp: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(inp["path"])
        disp = self._rel_path(path)
        if (err := self._check_write(path)):
            return err
        if path.is_dir():
            return {"content": f"Is a directory: {disp}", "is_error": True}
        if not path.exists():
            return {"content": f"File not found: {disp}", "is_error": True}
        bak = self._backup_file(path)
        text = path.read_text(encoding="utf-8")
        old, new = inp["old_string"], inp["new_string"]
        count = text.count(old)
        if count == 0:
            return {"content": f"old_string not found in {path}", "is_error": True}
        if inp.get("replace_all"):
            text = text.replace(old, new)
        elif count > 1:
            return {"content": f"old_string matches {count} times. Use replace_all=true or provide more context.", "is_error": True}
        else:
            text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")
        m = self._load_manifest()
        self._snapshot_file(path, m, backed=True)
        self._save_manifest(m)
        return {"content": f"Replaced {count} occurrence(s) in {self._rel_path(path)}" + (" (backed up)" if bak else ""), "is_error": False}

    def _tool_delete(self, inp: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(inp["path"])
        if (err := self._check_write(path)):
            return err
        disp = self._rel_path(path)
        if not path.exists():
            return {"content": f"File not found: {disp}", "is_error": True}
        if path.is_dir():
            return {"content": f"Path is a directory, use Bash to remove: {disp}", "is_error": True}
        bak = self._backup_file(path)
        try:
            path.unlink()
        except OSError as e:
            return {"content": f"Failed to delete {disp}: {e}", "is_error": True}
        m = self._load_manifest()
        m.pop(self._manifest_key(path), None)
        self._save_manifest(m)
        return {"content": f"Deleted {disp}" + (" (backed up)" if bak else ""), "is_error": False}

    @staticmethod
    def _expand_braces(pattern: str) -> list[str]:
        """Expand {a,b,c} brace syntax into multiple patterns."""
        m = re.search(r'\{([^{}]+)\}', pattern)
        if not m:
            return [pattern]
        prefix = pattern[:m.start()]
        suffix = pattern[m.end():]
        results: list[str] = []
        for opt in m.group(1).split(','):
            expanded = prefix + opt.strip() + suffix
            results.extend(MiniAgent._expand_braces(expanded))
        return results

    def _tool_glob(self, inp: dict[str, Any]) -> dict[str, Any]:
        base = self._resolve(inp.get("path", str(self.workspace)))
        pattern = inp.get("pattern", "*")
        # Prevent .. traversal escaping read_root via glob pattern.
        # Resolve non-wildcard prefix of pattern + base, block if outside read_root.
        pfx = re.split(r'[*?\[\{]', pattern.replace('\\', '/'))[0]
        if pfx and '..' in pfx:
            try:
                (base / pfx).resolve().relative_to(self._read_root)
            except ValueError:
                return {"content": f"Glob {pattern} — blocked: pattern escapes read_root", "is_error": True}
        matches: list[Path] = []
        for p in self._expand_braces(pattern):
            for m in base.glob(p):
                if self._check_read_root(m.resolve(), self._read_root).name == "__denied__":
                    continue
                matches.append(m)
        matches = sorted(set(matches))
        lines: list[str] = []
        for m in matches[:200]:
            try:
                lines.append(m.resolve().relative_to(self.workspace).as_posix())
            except ValueError:
                lines.append(str(m))
        return {"content": f"Glob {pattern} — {len(matches)} matches\n" + "\n".join(lines), "is_error": False}

    def _rel_path(self, p: Path) -> str:
        """Display path relative to workspace, masking absolute paths."""
        try:
            return p.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            return p.name

    def _tool_grep(self, inp: dict[str, Any]) -> dict[str, Any]:
        pattern = inp["pattern"]
        search_path = self._resolve(inp.get("path", str(self.workspace)))
        flags = re.IGNORECASE if inp.get("-i") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {"content": f"Invalid regex: {e}", "is_error": True}
        output_mode = inp.get("output_mode", "content")
        limit = inp.get("head_limit", 250)
        file_glob = inp.get("glob")
        if search_path.is_file():
            paths = iter([search_path])
        else:
            paths = search_path.rglob("*")
        matched_files: dict[str, int] = {}
        results: list[str] = []
        content_done = False
        for p in paths:
            if content_done:
                break
            if not p.is_file():
                continue
            if file_glob and not fnmatch.fnmatch(p.name, file_glob):
                continue
            rel = self._rel_path(p)
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").split("\n"), 1):
                    if regex.search(line):
                        matched_files[rel] = matched_files.get(rel, 0) + 1
                        if output_mode == "content":
                            if len(results) < limit:
                                line_text = line.rstrip()
                                if len(line_text) > 2000:
                                    line_text = line_text[:2000] + f"... [truncated, {len(line_text)} chars]"
                                results.append(f"{rel}:{i}: {line_text}")
                            if len(results) >= limit:
                                content_done = True
                                break
                        elif output_mode == "files_with_matches":
                            break
            except Exception:
                pass
        if output_mode == "files_with_matches":
            sorted_files = sorted(matched_files.keys())
            return {"content": "\n".join(sorted_files) if sorted_files else "(no matches)", "is_error": False}
        elif output_mode == "count":
            total = sum(matched_files.values())
            lines_out = [f"{p}: {c}" for p, c in sorted(matched_files.items())]
            lines_out.append(f"\nTotal: {total}")
            return {"content": "\n".join(lines_out) if matched_files else "(no matches)", "is_error": False}
        else:
            out = "\n".join(results) if results else "(no matches)"
            if len(out) > 50000:
                path = self._save_large_output(out)
                out = self._truncate_middle(out) + f"\n\n[Full output: {path}]"
            return {"content": out, "is_error": False}

    def _tavily_keys(self) -> list[str]:
        raw = self._provider_config.get("tavily_api_key", "") if self._provider_config else ""
        return [k.strip() for k in raw.split("|") if k.strip()]

    def _tavily_post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call Tavily API. Tries each |-separated key, falls through on quota/rate errors."""
        keys = self._tavily_keys()
        if not keys:
            return {"ok": False, "error": "No Tavily API key configured"}
        from core.http_utils import build_httpx_client, resolve_proxy_url
        use_proxy = self._proxy_config.get("use_for_websearch", False) if self._proxy_config else False
        proxy_url = resolve_proxy_url({"address": self.proxy_url}) if (self.proxy_url and use_proxy) else None
        last_err = ""
        for api_key in keys:
            try:
                with build_httpx_client(proxy_url, timeout=30) as client:
                    resp = client.post(
                        f"https://api.tavily.com/{endpoint}",
                        json=payload,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code == 200:
                        return {"ok": True, "data": resp.json()}
                    err = f"Tavily {endpoint}: HTTP {resp.status_code} — {resp.text[:200]}"
                    if resp.status_code in (402, 429):
                        last_err = err
                        continue
                    return {"ok": False, "error": err}
            except Exception as e:
                last_err = f"Tavily {endpoint}: {e}"
                continue
        return {"ok": False, "error": last_err}

    def _tool_websearch(self, inp: dict[str, Any]) -> dict[str, Any]:
        query = inp["query"]
        # Try Tavily first
        payload: dict[str, Any] = {"query": query, "max_results": 8, "search_depth": "basic"}
        if inp.get("allowed_domains"):
            payload["include_domains"] = inp["allowed_domains"]
        if inp.get("blocked_domains"):
            payload["exclude_domains"] = inp["blocked_domains"]
        tr = self._tavily_post("search", payload)
        if tr.get("ok"):
            data = tr["data"]
            results: list[str] = []
            for r in data.get("results", [])[:8]:
                line = f"{r.get('title', 'Untitled')}\n  {r.get('url', '')}"
                content = r.get("content", "")
                if content:
                    line += f"\n  {content}"
                results.append(line)
            answer = data.get("answer", "")
            if answer:
                results.insert(0, f"[AI Answer]: {answer}")
            if not results:
                return {"content": f"No results for: {query}", "is_error": False}
            return {"content": f"Search results for '{query}':\n\n" + "\n\n".join(results), "is_error": False}
        # DuckDuckGo fallback
        url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
        opener = self._build_opener()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MiniAgent/1.0"})
            with opener.open(req, timeout=15) as resp:
                html_text = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return {"content": f"WebSearch failed: Tavily API error ({tr.get('error')}); DDG fallback error: {e}", "is_error": True}
        results = []
        for match in re.finditer(r"<a\b[^>]*\bhref=\"([^\"]*)\"[^>]*\bclass=[\047]result-link[\047][^>]*>(.*?)</a>", html_text, re.DOTALL):
            raw_href = match.group(1)
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            title = html.unescape(title)
            actual_url = raw_href
            parsed = urllib.parse.urlparse(raw_href if "//" in raw_href else "https:" + raw_href)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs:
                actual_url = urllib.parse.unquote(qs["uddg"][0])
            if title:
                results.append(f"{title}\n  {actual_url}")
        for idx, match in enumerate(re.finditer(r"<td\b[^>]*\bclass=[\047]result-snippet[\047][^>]*>(.*?)</td>", html_text, re.DOTALL)):
            snippet = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            snippet = html.unescape(snippet)
            if snippet and idx < len(results) and idx < 10:
                results[idx] += f"\n  {snippet}"
        if not results:
            return {"content": f"No results for: {query}", "is_error": False}
        return {"content": f"Search results for '{query}':\n\n" + "\n\n".join(results[:8]), "is_error": False}

    def _tool_webfetch(self, inp: dict[str, Any]) -> dict[str, Any]:
        url = inp["url"]
        raw = b""
        last_err = None
        opener = self._build_opener()
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; MiniAgent/1.0)",
                "Accept": "text/html,text/plain",
            })
            with opener.open(req, timeout=20) as resp:
                try:
                    raw = resp.read()
                except IncompleteRead as e:
                    raw = e.partial
        except Exception as e:
            last_err = e
        if not raw:
            return {"content": f"WebFetch failed: {last_err}", "is_error": True}
        if raw[:2] == b"\x1f\x8b":
            import gzip
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
        try:
            if b"text/html" in raw[:500] or raw[:200].lstrip().startswith(b"<"):
                html = raw.decode("utf-8", errors="ignore")
                text = re.sub(r'<[^>]+>', '', html)
            else:
                text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("utf-8", errors="ignore")
        if len(text) > 8000:
            text = text[:8000] + f"\n\n[truncated at 8000 chars, total: {len(text)}]"
        prompt = inp.get("prompt", "")
        if prompt:
            text = f"Content from {url}:\n\n{text}\n\n---\nPrompt: {prompt}\n\nExtract the requested information."
        return {"content": text, "is_error": False}

    def _tool_tavilyextract(self, inp: dict[str, Any]) -> dict[str, Any]:
        urls = inp["urls"]
        payload: dict[str, Any] = {"urls": urls}
        if inp.get("extract_depth"):
            payload["extract_depth"] = inp["extract_depth"]
        tr = self._tavily_post("extract", payload)
        if not tr.get("ok"):
            return {"content": f"TavilyExtract failed: {tr.get('error')}", "is_error": True}
        data = tr["data"]
        lines: list[str] = []
        for r in data.get("results", []):
            lines.append(f"## {r.get('title', r.get('url', 'Untitled'))}\nURL: {r.get('url', '')}\n\n{r.get('raw_content', r.get('content', ''))}")
        failed = data.get("failed_results", [])
        if failed:
            lines.append(f"\n[Failed URLs: {', '.join(failed)}]")
        text = "\n\n---\n\n".join(lines)
        if len(text) > 12000:
            text = text[:12000] + "\n\n[truncated at 12000 chars]"
        return {"content": text or "(no content extracted)", "is_error": False}

    def _tool_tavilycrawl(self, inp: dict[str, Any]) -> dict[str, Any]:
        url = inp["url"]
        payload: dict[str, Any] = {"url": url, "max_pages": inp.get("max_pages", 10)}
        if inp.get("extract_depth"):
            payload["extract_depth"] = inp["extract_depth"]
        tr = self._tavily_post("crawl", payload)
        if not tr.get("ok"):
            return {"content": f"TavilyCrawl failed: {tr.get('error')}", "is_error": True}
        data = tr["data"]
        pages: list[str] = []
        for p in data.get("pages", []):
            pages.append(f"## {p.get('title', p.get('url', 'Untitled'))}\nURL: {p.get('url', '')}\n\n{p.get('content', p.get('raw_content', ''))}")
        text = "\n\n---\n\n".join(pages)
        if len(text) > 15000:
            text = text[:15000] + "\n\n[truncated at 15000 chars]"
        return {"content": text or "(no content)", "is_error": False}

    def _tool_tavilymap(self, inp: dict[str, Any]) -> dict[str, Any]:
        url = inp["url"]
        payload: dict[str, Any] = {"url": url, "max_results": inp.get("max_links", 20)}
        tr = self._tavily_post("map", payload)
        if not tr.get("ok"):
            return {"content": f"TavilyMap failed: {tr.get('error')}", "is_error": True}
        data = tr["data"]
        links = data.get("results", data.get("links", []))
        if not links:
            return {"content": f"No pages mapped for: {url}", "is_error": False}
        lines = [f"Site map for {url}:\n"]
        for link in links[:inp.get("max_links", 20)]:
            lines.append(f"- {link}")
        return {"content": "\n".join(lines), "is_error": False}

    def _build_opener(self):
        if self._web_opener is not None:
            return self._web_opener
        from core.http_utils import resolve_proxy_url, build_urllib_opener
        use_proxy = self._proxy_config.get("use_for_websearch", False) if self._proxy_config is not None else bool(self.proxy_url)
        proxy_url = resolve_proxy_url({"address": self.proxy_url}) if (self.proxy_url and use_proxy) else None
        self._web_opener = build_urllib_opener(proxy_url)
        return self._web_opener

    # ── Task Store ─────────────────────────────────────────

    def _load_tasks(self) -> dict[str, Any]:
        if self._task_path.exists():
            try:
                return json.loads(self._task_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"tasks": {}}

    def _save_tasks(self, data: dict[str, Any]) -> None:
        self._backup_file(self._task_path)
        self._task_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._task_path.with_suffix(f".json.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._task_path)

    def _tool_taskcreate(self, inp: dict[str, Any]) -> dict[str, Any]:
        data = self._load_tasks()
        tid = uuid.uuid4().hex[:8]
        data["tasks"][tid] = {"id": tid, "subject": inp["subject"],
                               "description": inp["description"], "status": "pending",
                               "activeForm": inp.get("activeForm", "")}
        self._save_tasks(data)
        return {"content": f"Task created: {tid} — {inp['subject']}", "is_error": False}

    def _tool_taskupdate(self, inp: dict[str, Any]) -> dict[str, Any]:
        data = self._load_tasks()
        tid = inp["taskId"]
        if tid not in data["tasks"]:
            return {"content": f"Task not found: {tid}", "is_error": True}
        task = data["tasks"][tid]
        for field in ("subject", "description", "status", "activeForm"):
            if field in inp and inp[field] is not None:
                task[field] = inp[field]
        self._save_tasks(data)
        return {"content": f"Task updated: {tid} → {task['status']}", "is_error": False}

    def _tool_tasklist(self, inp: dict[str, Any]) -> dict[str, Any]:
        data = self._load_tasks()
        tasks = data["tasks"]
        if not tasks:
            return {"content": "No tasks.", "is_error": False}
        lines = [f"{t['id']}  [{t['status']}] {t['subject']}" for t in tasks.values()]
        return {"content": "\n".join(lines), "is_error": False}

    def _tool_taskget(self, inp: dict[str, Any]) -> dict[str, Any]:
        data = self._load_tasks()
        tid = inp["taskId"]
        task = data["tasks"].get(tid)
        if not task:
            return {"content": f"Task not found: {tid}", "is_error": True}
        return {"content": json.dumps(task, ensure_ascii=False, indent=2), "is_error": False}

    # ── SubAgent ──────────────────────────────────────────

    def _tool_subagent(self, inp: dict[str, Any]) -> dict[str, Any]:
        prompt = inp["prompt"]
        s_model = inp.get("model", self.model)
        s_provider = inp.get("provider", "")
        s_max_tokens = max(inp.get("max_tokens", 32000), 8192)

        # Cap thinking budget below max_tokens (API rejects thinking >= max_tokens)
        sub_thinking = self.thinking_budget
        if sub_thinking >= s_max_tokens:
            sub_thinking = 0

        sub_api_key, sub_base_url = self.api_key, self.base_url
        if s_provider and self._provider_config:
            for p in self._provider_config.get("providers", []):
                if p.get("name") == s_provider:
                    sub_api_key = p.get("api_key", "") or self.api_key
                    sub_base_url = p["base_url"]
                    break

        sub = MiniAgent(
            workspace=self.workspace,
            model=s_model,
            max_iterations=self.max_iterations,
            thinking_budget=sub_thinking,
            max_tokens=s_max_tokens,
            api_key=sub_api_key,
            base_url=sub_base_url,
            custom_md_text=self.custom_md_text,
            proxy_url=self.proxy_url,
            allow_bash=self.allow_bash,
            vision_config=self._vision_config,
            protocol=self.protocol,
        )
        sub.set_provider_config(self._provider_config)
        sub.set_proxy_config(self._proxy_config)
        sub._read_root = self._read_root
        sub.remove_tool("SubAgent")

        # Run sub-agent in a thread, polling main agent's stop signal
        result_holder: dict[str, Any] = {}
        def _run():
            try:
                result_holder["text"] = sub.handle_message(prompt)
            except Exception as e:
                result_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        deadline = bj_epoch() + 1800  # 30 min safety net
        while t.is_alive() and bj_epoch() < deadline:
            t.join(timeout=1)  # poll every 1s
            if self._stop_event.is_set():
                sub.force_stop()
                t.join(timeout=3)
                return {"content": "SubAgent interrupted by user stop", "is_error": True}

        if t.is_alive():
            sub.force_stop()
            t.join(timeout=3)
            return {"content": "SubAgent timed out after 30min", "is_error": True}

        # Merge sub-agent token usage into main agent stats
        for k in ("in", "out", "cache_read", "cache_write"):
            self.token_usage[k] = self.token_usage.get(k, 0) + sub.token_usage.get(k, 0)

        if "error" in result_holder:
            return {"content": f"SubAgent failed: {result_holder['error']}", "is_error": True}
        return {"content": result_holder.get("text", "") or "(empty)", "is_error": False}

    # ── DescribeImage ─────────────────────────────────────

    def _tool_describeimage(self, inp: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(inp["path"])
        disp = self._rel_path(path)
        if not path.exists():
            return {"content": f"File not found: {disp}", "is_error": True}
        if not path.is_file():
            return {"content": f"Not a file: {disp}", "is_error": True}

        vc = self._vision_config
        providers = vc.get("providers", [])
        if not providers and vc.get("base_url"):
            providers = [{"base_url": vc["base_url"], "api_key": vc.get("api_key", ""), "model": vc.get("model", "GLM-4.6V-Flash")}]
        if not providers:
            return {"content": "Vision provider not configured (vision.providers empty in settings.json)", "is_error": True}

        ext = path.suffix.lower()
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif",
                    ".webp": "webp", ".bmp": "bmp"}
        mime = mime_map.get(ext)
        if not mime:
            return {"content": f"Unsupported image format: {ext}. Supported: jpg, png, gif, webp, bmp", "is_error": True}

        import base64
        try:
            img_b64 = base64.b64encode(path.read_bytes()).decode()
        except Exception as e:
            return {"content": f"Failed to read image: {e}", "is_error": True}

        prompt = inp.get("prompt", "请用中文详细描述这张图片的内容。")

        import urllib.request as _ureq
        from core.http_utils import resolve_proxy_url, build_urllib_opener
        raw_proxy = self.proxy_url or self._proxy_config.get("address", "")
        proxy_url = resolve_proxy_url({"address": raw_proxy}) if raw_proxy else None
        opener = build_urllib_opener(proxy_url) if (vc.get("use_proxy") and proxy_url) else build_urllib_opener()
        max_tok = int(vc.get("max_tokens", 4096) or 4096)

        total = sum(len([m.strip() for m in (p.get("model") or "").split(",") if m.strip()]) for p in providers)
        attempted = 0
        for provider in providers:
            base_url = provider.get("base_url", "").rstrip("/")
            api_key = provider.get("api_key", "")
            if not base_url or not api_key:
                attempted += len([m.strip() for m in (provider.get("model") or "").split(",") if m.strip()])
                continue
            endpoint = base_url + "/chat/completions"
            models = [m.strip() for m in (provider.get("model") or "").split(",") if m.strip()]
            for model in models:
                body = {
                    "model": model, "max_tokens": max_tok,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                }
                try:
                    req = _ureq.Request(endpoint, data=json.dumps(body).encode("utf-8"),
                                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
                    resp = opener.open(req, timeout=60)
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"].get("content", "") or "(no description)"
                    return {"content": content, "model": model, "is_error": False}
                except _ureq.HTTPError as e:
                    attempted += 1
                    err_body = e.read().decode("utf-8", errors="ignore")
                    code = e.code
                    is_rate_limit = code == 429 or "1305" in err_body or "1302" in err_body
                    if is_rate_limit and attempted < total:
                        _time.sleep(2)
                        continue
                    if attempted >= total:
                        return {"content": f"Vision API error {code}: {err_body[:300]}", "is_error": True}
                    _time.sleep(2)
                except Exception as e:
                    attempted += 1
                    if attempted >= total:
                        return {"content": f"Vision API failed: {e}", "is_error": True}
                    _time.sleep(2)

    # ── Conversation Cleanup ───────────────────────────────

    def clean_orphaned_tool_results(self) -> None:
        with self._msg_lock:
            for i in range(len(self.conversation) - 1):
                msg = self.conversation[i]
                next_msg = self.conversation[i + 1]
                if msg.get("role") != "assistant":
                    continue
                if next_msg.get("role") != "user":
                    continue
                next_content = next_msg.get("content", [])
                if not isinstance(next_content, list):
                    continue
                has_tool_results = any(b.get("type") == "tool_result" for b in next_content)
                if not has_tool_results:
                    continue
                curr_tool_use_ids = set()
                curr_content = msg.get("content", [])
                if isinstance(curr_content, list):
                    for b in curr_content:
                        if b.get("type") == "tool_use":
                            curr_tool_use_ids.add(b.get("id", ""))
                if not curr_tool_use_ids:
                    valid = [b for b in next_content if b.get("type") != "tool_result"]
                    if len(valid) != len(next_content):
                        next_msg["content"] = valid
                else:
                    valid = [b for b in next_content
                             if b.get("type") != "tool_result" or b.get("tool_use_id") in curr_tool_use_ids]
                    if len(valid) != len(next_content):
                        next_msg["content"] = valid
            self.conversation = [m for m in self.conversation
                                 if not (m.get("role") == "user" and isinstance(m.get("content"), list)
                                         and len(m["content"]) == 0)]

    # ── File Backup ────────────────────────────────────────

    def _backup_dir(self) -> Path:
        d = _SHARED_WORKSPACE / ".backup"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _backup_file(self, path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        ts = bj_now().strftime("%Y%m%d_%H%M%S_") + f"{int(bj_epoch() * 1000) % 1000:03d}"
        try:
            rel = path.resolve().relative_to(_SHARED_WORKSPACE.resolve())
        except ValueError:
            try:
                rel = path.relative_to(path.anchor) if path.is_absolute() else path
            except ValueError:
                rel = Path(path.name)
        try:
            bak_path = self._backup_dir() / rel.parent / f"{ts}_{rel.name}.bak"
            bak_path.parent.mkdir(parents=True, exist_ok=True)
            bak_path.write_bytes(path.read_bytes())
            return str(bak_path.relative_to(self._backup_dir()))
        except (OSError, ValueError):
            return None

    # ── File change manifest (Git-like md5 tracking) ──────────

    def _manifest_path(self) -> Path:
        return self.workspace / ".file_state.json"

    def _load_manifest(self) -> dict:
        mp = self._manifest_path()
        if mp.exists():
            try:
                return json.loads(mp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_manifest(self, m: dict) -> None:
        try:
            tmp = self._manifest_path().with_suffix(f".json.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._manifest_path())
        except OSError:
            pass

    def _manifest_key(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.workspace)).replace("\\", "/")
        except ValueError:
            return path.name

    def _snapshot_file(self, path: Path, m: dict, backed: bool = False) -> str:
        """Update manifest entry. backed=False means not yet saved to .backup/."""
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        h = hashlib.md5(data).hexdigest()
        st = path.stat()
        m[self._manifest_key(path)] = {"md5": h, "mtime": st.st_mtime, "size": st.st_size, "backed": backed}
        return h

    def _scan_and_backup(self, m: dict) -> list[str]:
        """Backup files not yet in .backup/ since last snapshot. Sets backed=True."""
        backed: list[str] = []
        for f in self.workspace.rglob("*"):
            if not f.is_file() or f.name == ".file_state.json":
                continue
            if ".backup" in f.parts or ".agent_outputs" in f.parts:
                continue
            if "ua_store" in f.parts or ".done" in f.parts:
                continue
            key = self._manifest_key(f)
            entry = m.get(key)
            if entry and entry.get("backed"):
                continue  # already backed up this version
            if self._backup_file(f):
                self._snapshot_file(f, m, backed=True)
                backed.append(key)
        return backed

    # ── Helpers ────────────────────────────────────────────

    def set_system_prompt(self, text: str) -> None:
        self.system_prompt_text = text

    def force_stop(self) -> None:
        self._stop_event.set()
        try:
            driver = get_driver(self.protocol)
            driver.cancel()
        except Exception:
            pass

    def cleanup_after_stop(self) -> dict:
        """Clean up conversation after user stops generation.
        - Removes dangling user messages that lack assistant replies.
        - Injects failed tool_result blocks for orphaned tool_use.
        Returns summary dict for logging."""
        with self._msg_lock:
            while self.conversation:
                last = self.conversation[-1]
                if last.get("role") == "user" and isinstance(last.get("content"), str):
                    self.conversation.pop()
                    self._turn -= 1
                else:
                    break

            for i in range(len(self.conversation) - 1, -1, -1):
                msg = self.conversation[i]
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                orphaned = [b for b in content if b.get("type") == "tool_use"]
                if not orphaned:
                    continue
                next_msg = self.conversation[i + 1] if i + 1 < len(self.conversation) else None
                existing_results = set()
                if next_msg and next_msg.get("role") == "user":
                    nc = next_msg.get("content", [])
                    if isinstance(nc, list):
                        for b in nc:
                            if b.get("type") == "tool_result":
                                existing_results.add(b.get("tool_use_id", ""))
                failed = []
                for tu in orphaned:
                    if tu.get("id") not in existing_results:
                        failed.append({
                            "type": "tool_result",
                            "tool_use_id": tu.get("id", ""),
                            "content": f"Tool {tu.get('name', '')} interrupted by user stop",
                            "is_error": True,
                        })
                if failed:
                    turn = msg.get("turn", self._turn)
                    if next_msg and next_msg.get("role") == "user" and isinstance(next_msg.get("content"), list):
                        next_msg["content"] = next_msg.get("content", []) + failed
                    else:
                        self.conversation.insert(i + 1, {
                            "role": "user", "content": failed, "turn": turn,
                        })

        return {"turn": self._turn, "in": self.token_usage["in"],
                "out": self.token_usage["out"], "stop": True}

    def get_last_assistant_text(self) -> str:
        """Extract concatenated text from the last assistant message, for sync comparison."""
        for m in reversed(self.conversation):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, list):
                    return "".join(b.get("text", "") for b in content if b.get("type") == "text")
                return content if isinstance(content, str) else ""
        return ""
