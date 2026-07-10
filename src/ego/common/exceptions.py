"""Project-specific exception types for EGO."""

from __future__ import annotations


class EgoError(Exception):
    """Base class for all EGO errors."""


class EgoConfigError(EgoError):
    """Raised when a configuration file is missing, invalid, or incomplete."""


class EgoDatasetError(EgoError):
    """Raised when dataset annotations, videos, or manifests are invalid."""


class EgoLabelMappingError(EgoError):
    """Raised when verb/noun/action label mappings are inconsistent."""


class EgoCheckpointError(EgoError):
    """Raised when a model checkpoint cannot be loaded or is incompatible."""
