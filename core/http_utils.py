"""Shared HTTP client and proxy utilities."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse
import httpx
import urllib.request


def _remove_suffix(s: str, suffix: str) -> str:
    """Python 3.8 compat: str.removesuffix is 3.9+."""
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def resolve_proxy_url(proxy_config: dict | None) -> str | None:
    """Extract proxy URL from config dict. Returns None if disabled or empty."""
    if not proxy_config or not proxy_config.get("address"):
        return None
    addr: str = proxy_config["address"]
    if not addr:
        return None
    return addr if "://" in addr else f"http://{addr}"


def preprocess_base_url(base_url: str) -> str:
    """Strip known API path suffixes, matching build_agent_from_route behaviour.

    Normalises raw base_url so URL construction starts from a clean base.
    Handles Anthropic suffixes (/v1/messages, /messages) and OpenAI suffixes
    (/v1/chat/completions, /chat/completions).  Longer prefixes are stripped
    first to avoid the shorter suffix consuming part of the longer one.
    """
    s = base_url.strip().rstrip("/")
    s = _remove_suffix(s, "/v1/messages")
    s = _remove_suffix(s, "/v1/chat/completions")
    s = _remove_suffix(s, "/messages")
    s = _remove_suffix(s, "/chat/completions")
    s = _remove_suffix(s, "/v1")
    return s


def build_anthropic_messages_url(base_url: str) -> str:
    """Build the full Anthropic Messages API URL ending in /v1/messages.

    Defensive: strips known suffixes before reconstructing so it works on
    already-normalised URLs as well as raw base_url values.
    """
    base = base_url.strip().rstrip("/")
    base = _remove_suffix(base, "/v1/messages")
    base = _remove_suffix(base, "/messages")
    parsed = urlparse(base)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return f"{parsed.scheme}://{parsed.netloc}{path}/messages"
    return f"{parsed.scheme}://{parsed.netloc}{path}/v1/messages"


def build_openai_chat_url(base_url: str) -> str:
    """Build the full OpenAI Chat Completions URL ending in /v1/chat/completions.

    Defensive: strips known suffixes before reconstructing.
    """
    base = base_url.strip().rstrip("/")
    base = _remove_suffix(base, "/v1/chat/completions")
    base = _remove_suffix(base, "/chat/completions")
    base = _remove_suffix(base, "/v1/messages")
    base = _remove_suffix(base, "/messages")
    parsed = urlparse(base)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return f"{parsed.scheme}://{parsed.netloc}{path}/chat/completions"
    return f"{parsed.scheme}://{parsed.netloc}{path}/v1/chat/completions"


def build_httpx_client(proxy_url: str | None = None, timeout: float = 600, connect_timeout: float = 10) -> httpx.Client:
    """Build an httpx.Client with optional proxy."""
    kwargs: dict[str, Any] = {"timeout": httpx.Timeout(timeout, connect=connect_timeout)}
    if proxy_url:
        kwargs["proxy"] = proxy_url
    return httpx.Client(**kwargs)


def build_urllib_opener(proxy_url: str | None = None) -> urllib.request.OpenerDirector:
    """Build a urllib OpenerDirector with optional proxy."""
    if proxy_url:
        handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()
