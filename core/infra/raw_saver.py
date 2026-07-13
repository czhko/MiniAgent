"""Save raw model API responses for debugging. Layer 2 — depends on paths/timeutil."""
from __future__ import annotations

from core.paths import ROOT_DIR
from core.timeutil import bj_now, bj_epoch

_RAW_DIR = ROOT_DIR / "config" / "logs" / "raw"


def save_raw(model: str, protocol: str, raw_lines: list[str]) -> str | None:
    """Persist raw SSE lines to disk. Returns file path or None."""
    if not raw_lines:
        return None
    try:
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        ts = bj_now().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "-").replace("\\", "-")
        fname = f"{ts}_{safe_model}_{bj_epoch():.3f}.raw"
        path = _RAW_DIR / fname
        path.write_text("".join(raw_lines), encoding="utf-8")
        return str(path)
    except OSError:
        return None
