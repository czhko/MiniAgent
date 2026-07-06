"""Event constants + DebugSink Protocol. Layer 1 — zero dependencies."""


class DriverEvent:
    """drivers -> agent event stream types."""
    TEXT            = "text"
    THINK           = "think"
    TOOL_USE        = "tool_use"
    TOOL_RESULT     = "tool_result"
    RETRY           = "retry"
    STOP            = "stop"
    ERROR           = "error"
    USAGE           = "usage"
    # Driver-internal events (not emitted upward)
    _INPUT_JSON_DELTA = "input_json_delta"
    _THINKING         = "thinking"


class EmitPhase:
    """execution -> server emit_cb phase parameter."""
    MAIN            = "main"
    SIDE            = "side"
