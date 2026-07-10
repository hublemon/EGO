"""Dataset label mapping scaffold.

Builds the dense/unified verb, noun, and action id spaces used as model
output indices, and guards against the failure modes that action-anticipation
pipelines are prone to: train/val mapping drift, non-deterministic id
assignment, and silent handling of unknown classes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ego.common.exceptions import EgoLabelMappingError


@dataclass
class LabelMapping:
    """Deterministic raw-id <-> unified-id mapping, fit on the train split only.

    ``verb_classes``/``noun_classes`` map raw dataset ids to dense ids
    ``0..K-1`` assigned by sorted order (never Python ``set`` iteration order,
    which is not guaranteed stable across processes/runs). ``action_classes``
    maps ``(raw_verb_id, raw_noun_id)`` to a dense joint id, also sorted.
    """

    verb_classes: dict[int, int]
    noun_classes: dict[int, int]
    action_classes: dict[tuple[int, int], int]
    verb_text: dict[int, str] = field(default_factory=dict)
    noun_text: dict[int, str] = field(default_factory=dict)

    inv_verb_classes: dict[int, int] = field(init=False)
    inv_noun_classes: dict[int, int] = field(init=False)
    inv_action_classes: dict[int, tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.inv_verb_classes = {v: k for k, v in self.verb_classes.items()}
        self.inv_noun_classes = {v: k for k, v in self.noun_classes.items()}
        self.inv_action_classes = {v: k for k, v in self.action_classes.items()}

    @property
    def num_verbs(self) -> int:
        return len(self.verb_classes)

    @property
    def num_nouns(self) -> int:
        return len(self.noun_classes)

    @property
    def num_actions(self) -> int:
        return len(self.action_classes)

    def encode_verb(self, raw_verb_id: int) -> int:
        if raw_verb_id not in self.verb_classes:
            raise EgoLabelMappingError(
                f"Unknown verb id {raw_verb_id}: not present in the train-fit label mapping."
            )
        return self.verb_classes[raw_verb_id]

    def encode_noun(self, raw_noun_id: int) -> int:
        if raw_noun_id not in self.noun_classes:
            raise EgoLabelMappingError(
                f"Unknown noun id {raw_noun_id}: not present in the train-fit label mapping."
            )
        return self.noun_classes[raw_noun_id]

    def encode_action(self, raw_verb_id: int, raw_noun_id: int) -> int:
        key = (raw_verb_id, raw_noun_id)
        if key not in self.action_classes:
            raise EgoLabelMappingError(
                f"Unknown (verb, noun) pair {key}: not present in the train-fit label mapping."
            )
        return self.action_classes[key]

    def decode_verb_text(self, unified_verb_id: int) -> str | None:
        raw_id = self.inv_verb_classes.get(unified_verb_id)
        return self.verb_text.get(raw_id) if raw_id is not None else None

    def decode_noun_text(self, unified_noun_id: int) -> str | None:
        raw_id = self.inv_noun_classes.get(unified_noun_id)
        return self.noun_text.get(raw_id) if raw_id is not None else None

    def to_dict(self) -> dict:
        return {
            "num_verbs": self.num_verbs,
            "num_nouns": self.num_nouns,
            "num_actions": self.num_actions,
            "verb_classes": self.verb_classes,
            "noun_classes": self.noun_classes,
            "action_classes": {f"{v}|{n}": a for (v, n), a in self.action_classes.items()},
        }


def build_label_mapping(
    train_verb_noun_pairs: Iterable[tuple[int, int]],
    verb_text: dict[int, str] | None = None,
    noun_text: dict[int, str] | None = None,
) -> LabelMapping:
    """Fit a :class:`LabelMapping` on the (verb_id, noun_id) pairs seen in train.

    Deterministic by construction: ids are assigned in sorted order of the
    raw dataset ids, so the same train split always yields the same mapping
    regardless of row order or process.
    """
    pairs = list(train_verb_noun_pairs)
    if not pairs:
        raise EgoLabelMappingError("Cannot build a label mapping from zero training samples.")

    verbs = sorted({v for v, _ in pairs})
    nouns = sorted({n for _, n in pairs})
    actions = sorted(set(pairs))

    verb_classes = {raw: dense for dense, raw in enumerate(verbs)}
    noun_classes = {raw: dense for dense, raw in enumerate(nouns)}
    action_classes = {raw: dense for dense, raw in enumerate(actions)}

    return LabelMapping(
        verb_classes=verb_classes,
        noun_classes=noun_classes,
        action_classes=action_classes,
        verb_text=dict(verb_text or {}),
        noun_text=dict(noun_text or {}),
    )


def filter_to_known_pairs(
    rows: Iterable[dict],
    known_pairs: set[tuple[int, int]],
    verb_key: str = "verb_id",
    noun_key: str = "noun_id",
) -> list[dict]:
    """Keep only rows whose ``(verb_id, noun_id)`` pair was seen in the train split.

    Used to filter validation/test annotations down to the label space the
    model was actually fit on, so evaluation never encounters an unknown class.
    """
    return [row for row in rows if (row[verb_key], row[noun_key]) in known_pairs]


def check_mapping_covers_split(
    mapping: LabelMapping,
    split_verb_noun_pairs: Iterable[tuple[int, int]],
    split_name: str = "validation",
) -> None:
    """Raise :class:`EgoLabelMappingError` if ``split`` contains an unmapped verb/noun/action.

    Callers should filter splits with :func:`filter_to_known_pairs` first;
    this is the explicit safety net so an unmapped class fails loudly instead
    of raising a bare ``KeyError`` deep inside training.
    """
    for verb_id, noun_id in split_verb_noun_pairs:
        if verb_id not in mapping.verb_classes:
            raise EgoLabelMappingError(
                f"{split_name} split contains verb id {verb_id}, absent from the train mapping."
            )
        if noun_id not in mapping.noun_classes:
            raise EgoLabelMappingError(
                f"{split_name} split contains noun id {noun_id}, absent from the train mapping."
            )
        if (verb_id, noun_id) not in mapping.action_classes:
            raise EgoLabelMappingError(
                f"{split_name} split contains action pair {(verb_id, noun_id)}, "
                "absent from the train mapping."
            )
