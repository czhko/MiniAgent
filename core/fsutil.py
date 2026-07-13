"""File-system utilities: atomic write, backup. Layer 0 — zero internal core deps.

Imports from core.paths inside functions to avoid circular dependency at import time.
"""
from __future__ import annotations
from pathlib import Path


def _backup_dir() -> Path:
    from core.paths import BACKUP_DIR
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return BACKUP_DIR


def backup_file(path: Path) -> str | None:
    """Copy file to workspace/.backup/ preserving directory structure.
    Returns relative backup path or None on failure.
    """
    if not path.exists() or not path.is_file():
        return None
    from core.timeutil import bj_now, bj_epoch
    ts = bj_now().strftime("%Y%m%d_%H%M%S_") + f"{int(bj_epoch() * 1000) % 1000:03d}"
    try:
        try:
            from core.paths import WORKSPACE
            rel = path.resolve().relative_to(WORKSPACE.resolve())
        except ValueError:
            try:
                rel = path.relative_to(path.anchor) if path.is_absolute() else path
            except ValueError:
                rel = Path(path.name)
        bd = _backup_dir()
        bak_path = bd / rel.parent / f"{ts}_{rel.name}.bak"
        bak_path.parent.mkdir(parents=True, exist_ok=True)
        bak_path.write_bytes(path.read_bytes())
        return str(bak_path.relative_to(bd))
    except (OSError, ValueError):
        return None


def atomic_write(path: Path, data) -> dict:
    """Write JSON to path atomically via tmp + replace. Returns {"ok": True} or {"ok": False, "error": ...}."""
    import json
    backup_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return {"ok": True}
    except (OSError, TypeError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
