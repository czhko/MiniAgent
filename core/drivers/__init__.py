"""Model driver registry. Layer 4."""
from __future__ import annotations
from core.drivers.base import ModelDriver, ModelRequest, ModelResponse, ThinkingConfig
from core.drivers.anthropic import AnthropicDriver
from core.drivers.openai import OpenAIDriver

_DRIVERS: dict[str, ModelDriver] = {
    "anthropic": AnthropicDriver(),
    "openai": OpenAIDriver(),
}


def get_driver(protocol: str) -> ModelDriver:
    """Get a driver instance by protocol name."""
    driver = _DRIVERS.get(protocol)
    if driver is None:
        raise ValueError(f"Unknown protocol: {protocol}")
    return driver
