"""Protocol codec + event constants. Layer 1."""
from core.codec.owui import (
    ContentParts, extract_text, extract_content, parse_history_format,
    make_text_chunk, make_stop_chunk,
)
from core.codec.events import DriverEvent, EmitPhase
