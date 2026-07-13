"""OWUI protocol adaptation layer — conversation rebuild, OWUI text merging, &lt;think&gt; wrapping.

Layer: between L1 (codec) and L5 (engine). Bridges OWUI text format ↔ structured conversation.
"""
from core.adapter.owui import apply_owui_text, rebuild_conversation, make_on_event
