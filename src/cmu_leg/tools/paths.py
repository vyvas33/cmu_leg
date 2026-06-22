"""Output-path helper for tool/diagnostic scripts.
"""
from datetime import datetime
from pathlib import Path

def _repo_root() -> Path:
    """Walk up from this file to the dir containing pyproject.toml."""
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("repo root (pyproject.toml) not found")


def make_run_dir(label: str = "run") -> Path:
    """Create and return output/<timestamp>_<label>/ at the repo root.

    Each call is a new directory, so runs never overwrite each other.
    """
    safe = label.replace("/", "-").replace(" ", "-")
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = _repo_root() / "output" / f"{ts}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    return out
