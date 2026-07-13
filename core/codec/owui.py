"""OWUI message protocol codec. Layer 1 — parse + construct, never mutate.

Pure functions with no side effects and no internal state.
"""
from __future__ import annotations
from typing import TypedDict


class ContentParts(TypedDict):
    text: str
    images: list[dict]
    files: list[dict]


def extract_text(content) -> str:
    """Extract plain text from OWUI message content (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def extract_content(content) -> ContentParts:
    """Parse OWUI message content into structured parts."""
    result: ContentParts = {"text": "", "images": [], "files": []}
    if isinstance(content, str):
        result["text"] = content
        return result
    if not isinstance(content, list):
        return result
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type", "")
        if t == "text":
            result["text"] += block.get("text", "")
        elif t == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:image/"):
                header, b64 = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0].split("/")[-1]
                result["images"].append({"mime": mime, "data": b64})
        elif t == "image" and isinstance(block.get("source", {}), dict):
            src = block["source"]
            if src.get("type") == "base64":
                m = src.get("media_type", "image/png")
                result["images"].append({"mime": m.split("/")[-1], "data": src.get("data", "")})
        elif t == "file":
            f = block.get("file") or block.get("source") or {}
            result["files"].append({
                "name": f.get("filename", "unknown"),
                "mime": f.get("media_type", "application/octet-stream"),
                "data": f.get("data", ""),
            })
    return result


def parse_history_format(content: str) -> list[dict] | None:
    """Parse OWUI 'History:' compressed format into structured messages.

    Format:
        History:
        USER: \"\"\"...\"\"\"
        ASSISTANT: \"\"\"...\"\"\"
        Query: <current question>

    Returns list of {"role": ..., "content": "..."} dicts, or None if not History format.
    """
    if not isinstance(content, str) or not content.startswith("History:"):
        return None
    text = content[len("History:"):]
    messages: list[dict] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch in ('\n', '\r', ' '):
            pos += 1
            continue
        remain = text[pos:]
        if remain.startswith("USER:"):
            pos += len("USER:")
            pos = _extract_quoted(text, pos, messages, "user")
        elif remain.startswith("ASSISTANT:"):
            pos += len("ASSISTANT:")
            pos = _extract_quoted(text, pos, messages, "assistant")
        elif remain.startswith("SYSTEM:"):
            pos += len("SYSTEM:")
            pos = _extract_quoted(text, pos, messages, "system")
        elif remain.startswith("Query:"):
            q = remain[len("Query:"):].strip()
            messages.append({"role": "user", "content": q})
            break
        else:
            pos += 1
    return messages if messages else None


def _extract_quoted(text: str, pos: int, messages: list[dict], role: str) -> int:
    """Extract \"\"\"...\"\"\" content starting from pos. Returns new position."""
    q0 = text.find('"""', pos)
    if q0 == -1:
        return pos
    q1 = text.find('"""', q0 + 3)
    if q1 == -1:
        messages.append({"role": role, "content": text[q0 + 3:].strip()})
        return len(text)
    inner = text[q0 + 3:q1]
    messages.append({"role": role, "content": inner})
    return q1 + 3


def make_text_chunk(chat_id: str, model: str, text: str) -> dict:
    """Build an OpenAI-compatible SSE text chunk."""
    import time
    return {
        "id": chat_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


def make_stop_chunk(chat_id: str, model: str, usage: dict | None = None) -> dict:
    """Build an OpenAI-compatible SSE stop chunk with optional usage."""
    import time
    u = usage or {}
    return {
        "id": chat_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": u.get("input", 0),
            "completion_tokens": u.get("output", 0),
            "total_tokens": u.get("input", 0) + u.get("output", 0),
        },
    }


