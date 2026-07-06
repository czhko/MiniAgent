"""Path resolution and write safety. Shared between MiniAgent and AdminBackend."""
from pathlib import Path

_UNIX_ROOT_PREFIXES = [
    "/root/workspace/", "/root/", "/workspace/", "/tmp/", "/mnt/", "/home/",
]


def resolve_path(raw: str, workspace: Path, shared_workspace: Path) -> Path:
    """Resolve a tool-provided path string safely within workspace."""
    if raw.startswith("/") and not raw.startswith("//"):
        if raw.startswith("/home/") and raw.count("/") >= 3:
            parts = raw.split("/")
            raw = "/" + "/".join(parts[3:])
        for prefix in _UNIX_ROOT_PREFIXES:
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        p = Path(raw)
        if p.parts and p.parts[0] != "..":
            candidate = (workspace / p).resolve()
            if candidate.exists():
                return candidate
            shared = (shared_workspace / p).resolve()
            if shared.exists():
                return shared
        name = Path(raw).name
        if name:
            matches = list(workspace.rglob(name))
            if len(matches) == 1:
                return matches[0].resolve()
            matches = list(shared_workspace.rglob(name))
            if len(matches) == 1:
                return matches[0].resolve()
            return (workspace / name).resolve()
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    candidate = (workspace / p).resolve()
    if candidate.exists():
        return candidate
    shared = (shared_workspace / p).resolve()
    if shared.exists():
        return shared
    return candidate


def check_write(path: Path, workspace: Path) -> dict | None:
    """Check if path is within workspace. Returns None if allowed, error dict if denied."""
    try:
        path.resolve().relative_to(workspace)
        return None
    except ValueError:
        return {"content": "Permission denied: outside workspace/", "is_error": True}
