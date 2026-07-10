"""Action contract scaffold for verb, noun, and action-pair labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ActionLabel:
    """A resolved verb/noun/action label, in both raw dataset-id and unified-id space.

    ``verb_id``/``noun_id``/``action_id`` are the dense, contiguous ids used as
    model output indices (see ``ego.datasets.label_mapping.LabelMapping``).
    ``verb``/``noun`` are the human-readable text labels.
    """

    verb_id: int | None
    verb: str | None
    noun_id: int | None
    noun: str | None
    action_id: int | None

    def to_dict(self) -> dict:
        return asdict(self)
