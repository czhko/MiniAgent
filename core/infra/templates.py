"""Template resolution and thinking presets. Layer 2 — depends on paths."""
from pathlib import Path
from typing import Any

THINKING_PRESETS = {"off": 0, "low": 4000, "high": 16000, "max": 32000}
THINKING_BUDGET_REVERSE = {0: "off", 4000: "low", 16000: "high", 32000: "max"}


def normalize_thinking(value) -> str:
    """Normalize a thinking value (int budget or string preset) to a string preset name."""
    if value is None or value == "":
        return "off"
    if isinstance(value, int):
        return THINKING_BUDGET_REVERSE.get(value, "off")
    if isinstance(value, str) and value.isdigit():
        budget = int(value)
        return THINKING_BUDGET_REVERSE.get(budget, "off")
    if isinstance(value, str) and value in THINKING_PRESETS:
        return value
    return "off"


def resolve_variable(name: str, variables: dict) -> str:
    """Read file content for a variable binding. Returns raw content or placeholder."""
    path = variables.get(name, "")
    if not path:
        return f"{{{name}}}"
    try:
        p = Path(path)
        if not p.is_absolute():
            from core.paths import ROOT_DIR
            p = ROOT_DIR / p
        p = p.resolve()
        # Reject paths that escape the project root
        from core.paths import ROOT_DIR
        try:
            p.relative_to(ROOT_DIR.resolve())
        except ValueError:
            return f"{{{name}}}"
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        pass
    return f"{{{name}}}"


def resolve_template(template: str, variables: dict, extra: dict | None = None) -> str:
    """Replace {variable} and {key} placeholders in a template string."""
    import re
    result = template
    # Replace {variable_name} placeholders
    for name in variables:
        placeholder = "{" + name + "}"
        if placeholder in result:
            val = resolve_variable(name, variables)
            result = result.replace(placeholder, val)
    # Replace {key} placeholders from extra dict
    if extra:
        for k, v in extra.items():
            placeholder = "{" + k + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(v))
    return result
