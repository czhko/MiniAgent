"""UA (User-Assistant) pair store — hash-matched tool context recovery.

Stores raw conversation blocks (thinking/tool_use/tool_result/text) keyed by
SHA256 hash of the assistant text that OWUI will return. On the next request,
the same hash is computed from OWUI's assistant message text → raw blocks
are restored, giving the agent full context of previous tool calls.

v2: side path blocks stored alongside main path blocks under the same hash key.
One hash match → both main and side context restored. One hash miss → neither.

Storage: config/ua_store/{hash[:2]}/{hash[2:4]}/{hash}.json
Max entries: _MAX_FILES (oldest evicted by mtime).
"""
from __future__ import annotations

import hashlib, json, threading
from pathlib import Path

from core.timeutil import bj_epoch
from core.paths import ROOT_DIR

UA_STORE_DIR = ROOT_DIR / "config" / "ua_store"

# Safety limits
_MAX_FILES = 100_000           # max number of stored entries
_PREFIX_DEPTH = 2              # directory nesting depth (ab/cd/ef...json)
_PREFIX_LEN = 2                # chars per directory level
_VERSION = 2                   # v2: added "s" field for side path blocks

_lock = threading.Lock()
_backfill_done = False


def _hash(text: str) -> str:
    """First 16 hex chars of SHA256 — enough for collision resistance."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _path(h: str) -> Path:
    """Dir-prefixed path: ua_store/ab/cd/abcdef123456.json"""
    p = UA_STORE_DIR
    for i in range(0, _PREFIX_DEPTH):
        seg = h[i * _PREFIX_LEN:(i + 1) * _PREFIX_LEN]
        if len(seg) < _PREFIX_LEN:
            seg = seg.ljust(_PREFIX_LEN, "0")
        p = p / seg
    return p / f"{h}.json"


def _write_entry(hash_key: str, data: dict) -> int:
    """Write data to ua_store under hash_key. Returns bytes written."""
    path = _path(hash_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(raw, encoding="utf-8")
        tmp.replace(path)
        return len(raw)
    except OSError as e:
        try:
            from core.infra.logger import trace_log
            trace_log(f"UA_SAVE OSError hash={hash_key} err={e}")
        except Exception:
            pass
        return 0


def _try_load(path: Path) -> tuple:
    """Read a UA Store entry. Hash match = trust. Returns (main_blocks, side_blocks) or (None, None)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    return data.get("b"), data.get("s")


def save(assistant_text: str, main_blocks: list[dict],
         side_blocks: dict[str, list[dict]] | None = None) -> None:
    """Store main_blocks (and optional side_blocks) under text hash."""
    if not assistant_text or not main_blocks:
        return
    t0 = bj_epoch()
    data = {
        "v": _VERSION,
        "t": assistant_text,
        "b": main_blocks,
        "ts": bj_epoch(),
    }
    if side_blocks:
        data["s"] = side_blocks
    import re as _re
    hashes = [_hash(assistant_text)]
    stripped = _re.sub(r"<thinking>.*?</thinking>", "", assistant_text, flags=_re.DOTALL)
    if stripped != assistant_text:
        hashes.append(_hash(stripped))
    if len(assistant_text) > 100:
        hashes.append(_hash(assistant_text[:100]))
        hashes.append(_hash(assistant_text[-100:]))
    with _lock:
        for h in hashes:
            data["h"] = h
            _write_entry(h, data)
    _ms = int((bj_epoch() - t0) * 1000)
    if _ms > 50:
        from core.infra.logger import trace_log
        trace_log(
            f"UA_SAVE: {_ms}ms main_blocks={len(main_blocks)}"
            + (f" sides={list(side_blocks.keys())}" if side_blocks else "")
            + f" hashes={len(hashes)}")
    _evict_oldest()


def load(assistant_text: str) -> tuple[list[dict] | None, dict[str, list[dict]] | None]:
    """Look up (main_blocks, side_blocks) by hash.
    Returns (None, None) on miss.
    Tries full hash → head 100 → tail 100.
    """
    global _backfill_done
    if not _backfill_done:
        with _lock:
            if not _backfill_done:
                _backfill_done = True
                need_backfill = True
            else:
                need_backfill = False
        if need_backfill:
            backfill()
    if not assistant_text:
        return None, None
    t0 = bj_epoch()
    import re as _re
    candidates = [_hash(assistant_text)]
    stripped = _re.sub(r"<thinking>.*?</thinking>", "", assistant_text, flags=_re.DOTALL)
    if stripped != assistant_text:
        candidates.append(_hash(stripped))
    if len(assistant_text) > 100:
        candidates.append(_hash(assistant_text[:100]))
        candidates.append(_hash(assistant_text[-100:]))
    main_result = None
    side_result = None
    with _lock:
        for h in candidates:
            main_result, side_result = _try_load(_path(h))
            if main_result is not None:
                break
    _ms = int((bj_epoch() - t0) * 1000)
    if _ms > 10:
        from core.infra.logger import trace_log
        trace_log(
            f"UA_LOAD: {_ms}ms hit={main_result is not None}"
            f" sides={list(side_result.keys()) if side_result else 0}"
            f" text_len={len(assistant_text)}")
    return main_result, side_result


def _evict_oldest() -> None:
    """Remove the oldest 10% of files when over _MAX_FILES."""
    if not UA_STORE_DIR.exists():
        return
    files = sorted(UA_STORE_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if len(files) <= _MAX_FILES:
        return
    for f in files[:max(1, len(files) - _MAX_FILES)]:
        try:
            f.unlink()
        except OSError:
            pass
    # Clean up empty directories
    for d in sorted(UA_STORE_DIR.rglob("*"), key=lambda p: len(str(p)), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def backfill() -> int:
    """Migrate existing entries: add head-100 and tail-100 hash symlinks.
    Called once at startup. Returns number of entries backfilled."""
    if not UA_STORE_DIR.exists():
        return 0
    count = 0
    for p in sorted(UA_STORE_DIR.rglob("*.json")):
        rel = p.relative_to(UA_STORE_DIR)
        parts = rel.parts
        # Only process full-hash entries (depth 3: XX/YY/hash.json)
        if len(parts) != 3:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = data.get("t", "")
        if not t or len(t) < 100:
            continue
        # Check if head/tail entries already exist
        h_head = _hash(t[:100])
        h_tail = _hash(t[-100:])
        path_head = _path(h_head)
        path_tail = _path(h_tail)
        if not path_head.exists():
            data["h"] = h_head
            _write_entry(h_head, data)
            count += 1
        if not path_tail.exists():
            data["h"] = h_tail
            _write_entry(h_tail, data)
            count += 1
    return count
