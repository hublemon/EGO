"""Logging helpers for EGO commands and experiments."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "ego") -> logging.Logger:
    """Return a logger with a consistent console format across EGO commands."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger("ego")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logger


def step_log(step: int, phase: str, message: str) -> None:
    """Print a standardized checkpoint line, e.g. ``[Step 1][Train] Epoch 1/20``.

    Required by the Step 1 execution contract so progress is legible without
    reading source: dataset prep, model loading, training, and inference must
    each emit these lines for their key inputs/outputs.
    """
    get_logger().info(f"[Step {step}][{phase}] {message}")
