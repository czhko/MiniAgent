"""Anthropic protocol driver — streaming Messages API via SDK."""
from __future__ import annotations

import json, os
from typing import Any

import httpx
from anthropic import Anthropic

from core.infra.logger import log_model_req
from core.drivers.base import ModelRequest, ModelResponse, EventFn
from core.http_utils import resolve_proxy_url
from core.timeutil import bj_epoch


def _httpx_request_hook(request: httpx.Request):
    try:
        body = request.content.decode("utf-8", errors="replace")
        safe_headers = dict(request.headers)
        for sensitive in ("authorization", "x-api-key", "api-key"):
            if sensitive in safe_headers:
                safe_headers[sensitive] = "***REDACTED***"
        log_model_req("anthropic", {
            "ts": bj_epoch(), "direction": "request",
            "method": request.method, "url": str(request.url),
            "headers": safe_headers,
            "body_len": len(body),
            "body": body[:3000] if len(body) > 3000 else body,
        })
    except Exception:
        pass


def _httpx_response_hook(response: httpx.Response):
    try:
        body = response.text[:2000] if response.text else ""
        log_model_req("anthropic", {
            "ts": bj_epoch(), "direction": "response",
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        })
    except Exception:
        pass


class AnthropicDriver:
    """Anthropic Messages API driver. Holds its own SDK client cache."""

    def __init__(self):
        self._client: Anthropic | None = None
        self._client_hash: str = ""
        self._active_stream: Any = None

    # ── ModelDriver Protocol ──────────────────────────

    def stream(self, req: ModelRequest, on_event: EventFn | None = None) -> ModelResponse:
        return self._call(req, on_event)

    def cancel(self) -> None:
        try:
            if self._active_stream is not None:
                self._active_stream.close()
        except Exception:
            pass
        self._active_stream = None

    # ── Internal ──────────────────────────────────────

    def _get_client(self, req: ModelRequest) -> Anthropic:
        api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY")
        key_hash = f"{api_key}|{req.base_url or ''}"
        if self._client is None or self._client_hash != key_hash:
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                # SDK 0.86.0: api_key → X-Api-Key, auth_token → Authorization: Bearer.
                # Both are merged in auth_headers and sent on every request.
                # auth_token also overrides any ANTHROPIC_AUTH_TOKEN env var,
                # preventing dual-auth conflicts when both env and code provide credentials.
                "auth_token": api_key,
            }
            if req.base_url:
                kwargs["base_url"] = req.base_url

            pc = req.proxy_config or {}
            proxy_url = resolve_proxy_url(pc) if pc.get("use_for_api") else None
            client_kwargs: dict[str, Any] = {
                "event_hooks": {
                    "request": [_httpx_request_hook],
                    "response": [_httpx_response_hook],
                }}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
            client_kwargs["timeout"] = httpx.Timeout(600, connect=10)
            kwargs["http_client"] = httpx.Client(**client_kwargs)

            self._client = Anthropic(**kwargs)
            self._client_hash = key_hash
        return self._client

    def _call(self, req: ModelRequest, on_event: EventFn | None) -> ModelResponse:
        client = self._get_client(req)

        is_true_anthropic = (
            not req.base_url
            or "anthropic.com" in (req.base_url or "").lower()
        ) and not req.model.lower().startswith("deepseek")

        system_param: list[dict[str, Any]] = [
            {"type": "text", "text": req.system_prompt}
        ]
        if is_true_anthropic:
            system_param[0]["cache_control"] = {"type": "ephemeral"}

        if req.messages:
            last_msg = req.messages[-1]
            last_content = last_msg["content"]
            if isinstance(last_content, str):
                cached_last = [{"type": "text", "text": last_content}]
                if is_true_anthropic:
                    cached_last[0]["cache_control"] = {"type": "ephemeral"}
            elif isinstance(last_content, list):
                if last_content:
                    cached_last = [dict(b) for b in last_content]
                    if is_true_anthropic:
                        for block in reversed(cached_last):
                            if block.get("type") == "text":
                                block["cache_control"] = {"type": "ephemeral"}
                                break
                else:
                    cached_last = [{"type": "text", "text": ""}]
            else:
                cached_last = [{"type": "text", "text": str(last_content) if last_content else ""}]
            api_messages = list(req.messages[:-1]) + [{"role": last_msg["role"], "content": cached_last}]
        else:
            api_messages = []

        display_text = ""

        api_kwargs: dict[str, Any] = {
            "model": req.model, "system": system_param, "messages": api_messages,
            "max_tokens": req.max_tokens,
        }
        if req.tools:
            api_kwargs["tools"] = req.tools
        if req.thinking and req.thinking.enabled:
            api_kwargs["thinking"] = req.thinking.to_anthropic()

        # Log request
        msg_dump = []
        for m in api_messages:
            c = m["content"]
            if isinstance(c, str):
                msg_dump.append({"role": m["role"], "content": c[:300]})
            elif isinstance(c, list):
                blocks = []
                for b in c:
                    if b.get("type") == "text":
                        blocks.append({"type": "text", "text": b["text"][:200]})
                    elif b.get("type") == "tool_use":
                        blocks.append({"type": "tool_use", "id": b["id"][:20], "name": b["name"]})
                    elif b.get("type") == "tool_result":
                        blocks.append({"type": "tool_result", "id": b.get("tool_use_id", "")[:20],
                                       "content": str(b.get("content", ""))[:100]})
                    else:
                        blocks.append({"type": b.get("type", "?")})
                msg_dump.append({"role": m["role"], "blocks": blocks})
        log_model_req("anthropic", {
            "ts": bj_epoch(), "type": "request", "model": req.model,
            "base_url": req.base_url, "max_tokens": req.max_tokens,
            "thinking_budget": req.thinking.budget_tokens if req.thinking else 0,
            "msg_count": len(api_messages), "has_tools": "tools" in api_kwargs,
            "system_len": len(req.system_prompt),
            "messages": msg_dump,
        })
        log_model_req("anthropic", {
            "type": "full_dump", "model": req.model, "base_url": req.base_url,
            "max_tokens": req.max_tokens,
            "thinking_budget": req.thinking.budget_tokens if req.thinking else 0,
            "system": req.system_prompt,
            "messages": api_messages,
            "tools_count": len(req.tools or []),
        })

        self._active_stream = client.messages.stream(**api_kwargs)
        raw_events: list[dict] = []
        try:
            with self._active_stream as stream:
                for event in stream:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type") and delta.type == "text_delta":
                            display_text += delta.text
                            raw_events.append({"type": "text", "delta": delta.text})
                            if on_event:
                                on_event("text", {"delta": delta.text})
                        elif hasattr(delta, "type") and delta.type == "thinking_delta":
                            raw_events.append({"type": "thinking", "delta": delta.thinking})
                            if on_event:
                                on_event("thinking", {"delta": delta.thinking})
                        elif hasattr(delta, "type") and delta.type == "input_json_delta":
                            raw_events.append({"type": "input_json_delta", "delta": delta.partial_json})
                    elif hasattr(event, "type"):
                        raw_events.append({"type": event.type})
                final_message = stream.get_final_message()
        finally:
            self._active_stream = None

        try:
            from core.infra.raw_saver import save_raw
            save_raw(req.model, "anthropic", [json.dumps({"events": raw_events, "content_blocks": [{"type": b.type, "text": getattr(b, "text", None), "thinking": getattr(b, "thinking", None)[:100] if hasattr(b, "thinking") else None} for b in final_message.content]}, ensure_ascii=False) + "\n"])
        except Exception:
            pass

        log_model_req("anthropic", {
            "ts": bj_epoch(), "type": "response", "status": "ok",
            "content_block_types": [b.type for b in final_message.content],
            "usage": {"in": getattr(final_message.usage, "input_tokens", 0),
                      "out": getattr(final_message.usage, "output_tokens", 0)},
        })

        content_blocks: list[dict[str, Any]] = []
        for block in final_message.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "thinking":
                content_blocks.append({"type": "thinking", "thinking": block.thinking,
                                       "signature": getattr(block, "signature", "")})
            elif block.type == "redacted_thinking":
                content_blocks.append({"type": "thinking", "thinking": "[redacted]",
                                       "signature": ""})
            elif block.type == "tool_use":
                content_blocks.append({"type": "tool_use", "id": block.id,
                                       "name": block.name, "input": block.input})

        usage = {}
        if hasattr(final_message, "usage"):
            usage["in"] = getattr(final_message.usage, "input_tokens", 0) or 0
            usage["out"] = getattr(final_message.usage, "output_tokens", 0) or 0
            if is_true_anthropic:
                usage["cache_read"] = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0
                usage["cache_write"] = getattr(final_message.usage, "cache_creation_input_tokens", 0) or 0
            if on_event:
                on_event("usage", {
                    "in": usage["in"], "out": usage["out"],
                    "cache_read": usage.get("cache_read", 0),
                    "cache_write": usage.get("cache_write", 0),
                })

        return ModelResponse(text=display_text, blocks=content_blocks, usage=usage)
