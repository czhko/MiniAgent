"""Time utilities — single canonical time source. Layer 0."""
from datetime import datetime, timezone, timedelta

BJ_TZ = timezone(timedelta(hours=8))  # exported for timestamp conversion


def bj_now() -> datetime:
    """Current Beijing time as timezone-aware datetime."""
    return datetime.now(BJ_TZ)


def bj_epoch() -> float:
    """Current epoch seconds (Beijing-time-aligned)."""
    return datetime.now(BJ_TZ).timestamp()
