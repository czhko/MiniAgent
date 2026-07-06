"""Driver contracts — data classes + ModelDriver Protocol. Layer 4."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# Event callback: (etype: str, data: dict) -> None
EventFn = Callable[[str, dict], None]


@dataclass
class ThinkingConfig:
    """Normalized thinking configuration. Agent-side, protocol-agnostic."""
    enabled: bool
    budget_tokens: int

    def to_anthropic(self) -> dict:
        return {"type": "enabled", "budget_tokens": self.budget_tokens}

    def to_openai(self) -> dict | None:
        return None  # OpenAI uses max_completion_tokens, not a thinking param


@dataclass
class ModelRequest:
    """Complete model call input. No MiniAgent references."""
    model:          str
    messages:       list[dict]
    system_prompt:  str
    max_tokens:     int
    protocol:       str

    api_key:        str | None       = None
    base_url:       str | None       = None
    proxy_config:   dict | None      = None
    tools:          list[dict] | None = None
    thinking:       ThinkingConfig | None = None


@dataclass
class ModelResponse:
    """Complete model call output."""
    text:           str
    blocks:         list[dict]
    usage:          dict = field(default_factory=dict)
    # {in, out, cache_read, cache_write} — cumulative for this call only


class ModelDriver(Protocol):
    """Protocol for model drivers. Each driver is a singleton instance."""

    def stream(self, req: ModelRequest, on_event: EventFn | None) -> ModelResponse:
        """Execute one streaming model call. Returns text + blocks + usage."""
        ...

    def cancel(self) -> None:
        """Cancel an in-progress stream. Called by force_stop."""
        ...
