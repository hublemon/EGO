"""Configuration loading helpers for EGO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ego.common.exceptions import EgoConfigError
from ego.common.io import read_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config as a plain nested dict."""
    p = Path(path)
    if not p.is_file():
        raise EgoConfigError(f"Config file not found: {p}")
    return read_yaml(p)


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Look up ``dotted_key`` (e.g. ``'dataset.video_root'``) in a nested dict."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require(config: dict[str, Any], dotted_key: str) -> Any:
    """Like :func:`get`, but raise ``EgoConfigError`` if the value is missing or null.

    Step 1 config templates ship with ``null`` placeholders for machine-specific
    paths and hyperparameters; this catches unfilled templates early with a
    clear error instead of failing deep inside training/inference.
    """
    value = get(config, dotted_key, default=_MISSING)
    if value is _MISSING or value is None:
        raise EgoConfigError(
            f"Required config key '{dotted_key}' is missing or null. "
            "Fill it in before running this command."
        )
    return value


_MISSING = object()
