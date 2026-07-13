"""Global path constants. Layer 0 — zero internal core dependencies."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT_DIR / "workspace"
BACKUP_DIR = WORKSPACE / ".backup"

LOG_DIR = ROOT_DIR / "config" / "logs"
