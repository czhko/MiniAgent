"""OpenAI-compatible protocol driver — streaming Chat Completions API."""
from __future__ import annotations

import json, time
from typing import Any

import httpx

from core.infra.logger import log_model_req
from core.drivers.base import ModelRequest, ModelResponse, EventFn
from core.http_utils import build_openai_chat_url, resolve_proxy_url
from core.timeutil import bj_epoch


def _to_openai_messages(conversation: list[dict]) -> list[dict]:
    """Convert internal Anthropic-format conversation to OpenAI wire format."""
    msgs = []
    for m in conversation:
        role = m.get("role", "")
        content = m.get("content", "")

        if role == "user":
            if isinstance(content, str):
                msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                texts = []
                tool_results = []
                for b in content:
                    if b.get("type") == "text":
                        texts.append(b.get("text", ""))
                    elif b.get("type") == "tool_result":
                        tool_results.append(b)
                text_content = "".join(texts)
                if text_content:
                    msgs.append({"role": "user", "content": text_content})
                for tr in tool_results:
                    result_content = tr.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "".join(
                            b.get("text", "") for b in result_content
                            if b.get("type") == "text"
                        )
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": str(result_content),
                    })

        elif role == "assistant":
            if isinstance(content, list):
                text_parts = []
                thinking_parts = []
                tool_calls = []
                for b in content:
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "thinking":
                        thinking_parts.append(b.get("thinking", ""))
                    elif b.get("type") == "tool_use":
                        args = b.get("input", {})
                        tool_calls.append({
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        })
                text_content = "".join(text_parts) or None
                reasoning_content = "".join(thinking_parts) or None
                msg: dict[str, Any] = {"role": "assistant"}
                if text_content is not None:
                    msg["content"] = text_content
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                if "content" not in msg and "tool_calls" not in msg:
                    msg["content"] = "(thinking)"
                if reasoning_content and tool_calls:
                    msg["reasoning_content"] = reasoning_content
                msgs.append(msg)
            elif isinstance(content, str):
                msgs.append({"role": "assistant", "content": content})
    return msgs


class OpenAIDriver:
    """OpenAI-compatible Chat Completions driver. Holds no persistent state."""

    def __init__(self):
        self._active_response: Any = None

    # ── ModelDriver Protocol ──────────────────────────

    def stream(self, req: ModelRequest, on_event: EventFn | None = None) -> ModelResponse:
        return self._call(req, on_event)

    def cancel(self) -> None:
        try:
            if self._active_response is not None:
                self._active_response.close()
        except Exception:
            pass
        self._active_response = None

    # ── Internal ──────────────────────────────────────

    def _call(self, req: ModelRequest, on_event: EventFn | None) -> ModelResponse:
        url = build_openai_chat_url(req.base_url or "https://api.openai.com/v1")

        headers = {
            "Authorization": f"Bearer {req.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        api_messages = _to_openai_messages(req.messages)

        body: dict[str, Any] = {
            "model": req.model,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if any(req.model.lower().startswith(p) for p in ("o1", "o3", "o4", "gpt-5")):
            body["max_completion_tokens"] = req.max_tokens
        else:
            body["max_tokens"] = req.max_tokens

        tools = []
        for t in (req.tools or []):
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        if tools:
            body["tools"] = tools

        if req.thinking and req.thinking.enabled:
            body["extra_body"] = {"thinking": {"type": "enabled"}}
            body["reasoning_effort"] = "max" if req.thinking.budget_tokens > 16000 else "high"

        if req.system_prompt:
            body["messages"].insert(0, {"role": "system", "content": req.system_prompt})

        # Log request
        log_model_req("openai", {
            "ts": bj_epoch(), "type": "request", "model": req.model,
            "base_url": url, "max_tokens": req.max_tokens,
            "thinking_budget": req.thinking.budget_tokens if req.thinking else 0,
            "msg_count": len(body["messages"]), "has_tools": bool(tools),
            "system_len": len(req.system_prompt),
        })

        display_text = ""
        reasoning_buf = ""
        content_blocks: list[dict[str, Any]] = []
        tool_calls: dict[int, dict] = {}
        final_usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

        try:
            from core.http_utils import build_httpx_client
            pc = req.proxy_config or {}
            proxy_url = resolve_proxy_url(pc) if pc.get("use_for_api") else None
            with build_httpx_client(proxy_url) as client:
                with client.stream("POST", url, json=body, headers=headers) as resp:
                    self._active_response = resp
                    if resp.status_code != 200:
                        try:
                            detail = resp.read().decode("utf-8", errors="ignore")[:500]
                        except Exception:
                            detail = "(could not read error body)"
                        raise RuntimeError(f"OpenAI API HTTP {resp.status_code}: {detail}")
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

                        if "usage" in chunk and chunk.get("usage"):
                            u = chunk["usage"]
                            final_usage = {
                                "in": u.get("prompt_tokens", 0),
                                "out": u.get("completion_tokens", 0),
                                "cache_read": u.get("prompt_cache_hit_tokens", 0),
                                "cache_write": u.get("prompt_cache_miss_tokens", 0),
                            }

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})

                        rc = delta.get("reasoning_content", "")
                        if rc:
                            reasoning_buf += rc
                            if on_event:
                                on_event("thinking", {"delta": rc})

                        tc_deltas = delta.get("tool_calls", [])
                        for tc in tc_deltas:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_calls[idx]["function"]["name"] += func["name"]
                            if func.get("arguments"):
                                tool_calls[idx]["function"]["arguments"] += func["arguments"]

                        content = delta.get("content", "")
                        if content:
                            display_text += content
                            if on_event:
                                on_event("text", {"delta": content})

            if reasoning_buf:
                content_blocks.append({"type": "thinking", "thinking": reasoning_buf, "signature": ""})
            if display_text:
                content_blocks.append({"type": "text", "text": display_text})
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                content_blocks.append({
                    "type": "tool_use", "id": tc["id"], "name": name, "input": args,
                })

        except Exception as e:
            log_model_req("openai", {"ts": bj_epoch(), "type": "response", "status": "error",
                         "error": str(e)[:500]})
            raise
        finally:
            self._active_response = None

        if on_event:
            on_event("usage", dict(final_usage))

        log_model_req("openai", {
            "ts": bj_epoch(), "type": "response", "status": "ok",
            "content_block_types": [b["type"] for b in content_blocks],
            "usage": final_usage,
        })

        return ModelResponse(text=display_text, blocks=content_blocks, usage=final_usage)
