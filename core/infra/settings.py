"""v0.4 Configuration center. Load/save settings/routes/chains/plugins, migration.

Layer 2 — depends on paths, fsutil, timeutil.
"""
from __future__ import annotations

import copy, hashlib, json, time as _time
from pathlib import Path
from typing import Any

from core.paths import ROOT_DIR, WORKSPACE, BACKUP_DIR as _BACKUP_DIR, LOG_DIR
from core.fsutil import backup_file as _backup_file, atomic_write as _atomic_write
from core.timeutil import bj_now

# ── Path constants ───────────────────────────────────────

SETTINGS_PATH = ROOT_DIR / "config" / "settings.json"
ROUTES_PATH = ROOT_DIR / "config" / "routes.json"
CHAINS_PATH = ROOT_DIR / "config" / "chains.json"
CHAINS_DIR = ROOT_DIR / "config" / "chains"
PLUGINS_DIR = ROOT_DIR / "config" / "plugins"

# ── Helpers ──────────────────────────────────────────────

_DEFAULT_SETTINGS: dict[str, Any] = {
    "_version": "0.4.3",
    "_build": "2026-06-17-fix-save",
    "system": {"system_prompt": "", "custom_md": ""},
    "proxy": {"address": "http://127.0.0.1:7897", "use_for_api": True, "use_for_websearch": False},
    "vision": {"use_proxy": False},
    "debug": {"enabled": False},
    "routes": [],
    "chains": [],
}


def get_agent_workspace(model: str = "", chain_label: str = "") -> Path:
    """Isolated workspace per agent/chain to prevent cross-contamination."""
    if chain_label:
        return WORKSPACE / chain_label
    return WORKSPACE / model.replace("/", "-")


def collect_done_tasks(workspace: Path) -> str:
    """Scan workspace/.done/ for completed background task markers.
    Returns concatenated content or empty string. Deletes markers after reading.
    """
    done_dir = workspace / ".done"
    if not done_dir.is_dir():
        return ""
    parts: list[str] = []
    try:
        for f in sorted(done_dir.iterdir()):
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except (OSError, UnicodeDecodeError):
                    pass
                try:
                    f.unlink()
                except OSError:
                    pass
    except OSError:
        pass
    return "\n".join(parts)


# ── Routes ───────────────────────────────────────────────

def load_routes() -> list[dict]:
    """Read routes.json. Returns [] if missing/corrupt."""
    if ROUTES_PATH.exists():
        try:
            data = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_routes(data: list) -> dict:
    """Atomic save to routes.json."""
    if not isinstance(data, list):
        return {"ok": False, "error": "Routes must be a JSON array"}
    return _atomic_write(ROUTES_PATH, data)


# ── Chains ───────────────────────────────────────────────

def load_chains() -> list[dict]:
    """Read chains from config/chains/ directory. One file per chain, broken files skipped.
    Falls back to chains.json for backward compat. Returns [] if nothing found."""
    if CHAINS_DIR.exists():
        chains = []
        for f in sorted(CHAINS_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("label"):
                    chains.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return chains
    if CHAINS_PATH.exists():
        try:
            data = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_chains(data: list) -> dict:
    """Atomic save each chain to config/chains/{label}.json. Removes deleted chains."""
    if not isinstance(data, list):
        return {"ok": False, "error": "Chains must be a JSON array"}
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in CHAINS_DIR.glob("*.json")}
    kept: set[str] = set()
    for chain in data:
        label = chain.get("label", "")
        if not label:
            continue
        filename = f"{label}.json"
        kept.add(filename)
        result = _atomic_write(CHAINS_DIR / filename, chain)
        if not result.get("ok"):
            return result
    for name in existing - kept:
        try:
            path = CHAINS_DIR / name
            _backup_file(path)
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return {"ok": True}


# ── Plugins ──────────────────────────────────────────────

def load_plugins() -> list[dict]:
    """Read plugins from config/plugins/ directory. One file per plugin, broken files skipped."""
    if PLUGINS_DIR.exists():
        plugins = []
        for f in sorted(PLUGINS_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("id"):
                    plugins.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return plugins
    return []


def save_plugins(data: list) -> dict:
    """Atomic save each plugin to config/plugins/{id}.json. Removes deleted plugins."""
    if not isinstance(data, list):
        return {"ok": False, "error": "Plugins must be a JSON array"}
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in PLUGINS_DIR.glob("*.json")}
    kept: set[str] = set()
    for plugin in data:
        pid = plugin.get("id", "")
        if not pid:
            continue
        filename = f"{pid}.json"
        kept.add(filename)
        result = _atomic_write(PLUGINS_DIR / filename, plugin)
        if not result.get("ok"):
            return result
    for name in existing - kept:
        try:
            path = PLUGINS_DIR / name
            _backup_file(path)
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return {"ok": True}


# ── Settings ─────────────────────────────────────────────

def load_settings() -> dict[str, Any]:
    """Load settings.json (system/proxy/vision only). Auto-migrates old formats.
    Routes and chains have their own files — use load_routes() / load_chains()."""
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if isinstance(data, dict):
            # Only migrate if old-format fields actually exist (idempotent, no version check)
            needs_migrate = "providers" in data or "da" in data
            if needs_migrate:
                data = _migrate_to_v04(data)
                routes = data.pop("routes", None)
                if routes:
                    save_routes(routes)
                chains = data.pop("chains", None)
                if chains:
                    save_chains(chains)
            else:
                data.pop("routes", None)
                data.pop("chains", None)
            return data
    return copy.deepcopy(_DEFAULT_SETTINGS)


def save_settings(data: dict) -> dict:
    """Atomic save settings.json. Strips dead fields, routes/chains (own files)."""
    if not isinstance(data, dict):
        return {"ok": False, "error": "Settings must be a JSON object"}
    copy_data = {k: v for k, v in data.items()}
    copy_data.pop("_usage", None)
    copy_data.pop("logging", None)
    copy_data.pop("routes", None)
    copy_data.pop("chains", None)
    if isinstance(copy_data.get("proxy"), dict):
        copy_data["proxy"].pop("use_for_plugins", None)
    copy_data["_version"] = "0.4.3"
    return _atomic_write(SETTINGS_PATH, copy_data)


def get_route(label: str) -> dict | None:
    """Find a route by label."""
    for r in load_routes():
        if r.get("label") == label:
            return r
    return None


# ── Migration ────────────────────────────────────────────

def _migrate_to_v04(old: dict) -> dict:
    """Migrate v0.3.x settings to v0.4 schema."""
    new = dict(_DEFAULT_SETTINGS)

    # Migrate system
    if "system_prompt" in old:
        new["system"]["system_prompt"] = old.get("system_prompt", "")
    if "custom_md" in old:
        new["system"]["custom_md"] = old.get("custom_md", "")

    # Migrate proxy
    proxy_str = old.get("proxy", "")
    if isinstance(proxy_str, str) and proxy_str:
        new["proxy"]["address"] = proxy_str
    elif isinstance(proxy_str, dict):
        new["proxy"] = proxy_str

    # Migrate providers -> routes
    for p in old.get("providers", []):
        route = {
            "label": p.get("name", ""),
            "protocol": p.get("protocol", "anthropic"),
            "api_key": p.get("api_key", ""),
            "base_url": p.get("base_url", ""),
            "thinking": "high",
            "max_iterations": old.get("max_iterations", 20),
            "context_1m": False,
            "enabled": p.get("enabled", True),
            "selected_models": [],
        }
        new["routes"].append(route)

    # Migrate vision
    if "vision" in old:
        new["vision"] = old["vision"]

    # Migrate da config -> chain
    da = old.get("da", {})
    old_providers = old.get("providers", [])
    main_route = ""
    main_model = ""
    if old_providers:
        first_protocol = old_providers[0].get("protocol", "anthropic")
        if first_protocol == "anthropic":
            default_main_model = "claude-sonnet-4-6"
            default_side_model = "claude-haiku-4-5"
        else:
            default_main_model = "deepseek-v4-pro"
            default_side_model = "deepseek-v4-flash"
        main_route = da.get("main_provider", "") or (old_providers[0].get("name", "") if old_providers else "")
        main_model = da.get("main_model", "") or old.get("model", default_main_model)

        chain = {
            "label": "da",
            "display_name": "da",
            "enabled": True,
            "side_paths": [{
                "label": "side",
                "type": "agent",
                "route": old_providers[0].get("name", ""),
                "model": default_side_model,
                "system_prompt": "{设定}",
                "user_injection": "{user_text}",
                "thinking": "low",
                "context_output": {"tool_use": True, "tool_result": True, "think": True, "text": True},
            }],
            "main_path": {
                "route": main_route,
                "model": main_model,
                "system_prompt": "{设定}",
                "user_injection": "{user_text}\n\n{side0}",
                "thinking": "high",
                "context_output": {"tool_use": True, "tool_result": True, "think": True, "text": True},
            },
            "variables": {},
        }
        new["chains"].append(chain)

    # Also create a "roleplay" chain (single agent)
    rp_chain = {
        "label": "roleplay",
        "display_name": "roleplay",
        "enabled": True,
        "side_paths": [],
        "main_path": {
            "route": main_route if old_providers else "",
            "model": main_model if old_providers else "",
            "system_prompt": "{设定}",
            "user_injection": "{user_text}",
            "thinking": "high",
            "context_output": {"tool_use": True, "tool_result": True, "think": True, "text": True},
        },
        "variables": {},
    }
    new["chains"].append(rp_chain)

    return new
