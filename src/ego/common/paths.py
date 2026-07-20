"""Path resolution helpers for datasets, outputs, and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

_DOTENV_LOADED = False


def project_root() -> Path:
    """Return the EGO repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def load_dotenv(path: str | Path | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables are never overridden. Missing files are
    silently ignored (``.env`` is optional; configs may hold concrete paths
    directly).
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    env_path = Path(path) if path is not None else project_root() / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _DOTENV_LOADED = True


def expand_path(value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Expand ``~`` and ``$VAR``/``${VAR}`` references and resolve relative paths.

    Relative paths are resolved against ``base_dir`` (default: repository root)
    so configs can use paths relative to the repo regardless of the caller's
    working directory.
    """
    load_dotenv()
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(base_dir) if base_dir is not None else project_root()
        path = path / expanded
    return path


def data_dir() -> Path:
    return project_root() / "data"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def checkpoints_dir() -> Path:
    return project_root() / "checkpoints"


def third_party_dir() -> Path:
    return project_root() / "third_party"
