"""Compatibility loader for the pre-refactor EK100 classifier checkpoint.

``checkpoints/step1/legacy_ek100_vitl256/best_action.pt`` was trained by the
``EvE/V-JEPA2`` prototype's ``evals/action_anticipation_frozen/eval.py``
under a differently-named architecture (``AttentiveClassifier``: pooler +
verb/noun/action Linear heads, DDP-wrapped) than this repo's
``AnticipationHead`` (``probe.pooler`` + the same three heads). The classes
are structurally identical, so this module remaps checkpoint keys instead of
retraining, and reconstructs the *exact* label mapping it was trained under
(verb/noun happen to be identical to :func:`build_label_mapping`'s sorted
order for EK100, since verb/noun ids are dense integer ranges and CPython's
``set`` of small ints iterates in sorted order -- verified against this
checkpoint's stored class counts, 97/289/3568. Action ids are NOT guaranteed
sorted by a plain ``set``, so :func:`build_legacy_ek100_label_mapping`
replicates the original non-sorted ``enumerate(set(...))`` algorithm
bit-for-bit rather than assuming equivalence).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ego.datasets.label_mapping import LabelMapping


def load_legacy_head_state_dict(checkpoint_path: str | Path, classifier_index: int = 0) -> dict:
    """Load and key-remap a legacy ``AttentiveClassifier`` checkpoint for :class:`AnticipationHead`.

    Strips the ``module.`` DDP prefix and rewrites ``pooler.`` ->
    ``probe.pooler.`` (the only structural difference between the two
    classes' attribute names).
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw["classifiers"][classifier_index] if "classifiers" in raw else raw
    remapped = {}
    for k, v in state_dict.items():
        k = k.removeprefix("module.")
        if k.startswith("pooler."):
            k = "probe." + k
        remapped[k] = v
    return remapped


def build_legacy_ek100_label_mapping(
    train_annotations_path: str | Path,
    verb_classes_csv: str | Path | None = None,
    noun_classes_csv: str | Path | None = None,
) -> LabelMapping:
    """Reconstruct the exact (non-sorted) label mapping the legacy checkpoint was trained under."""
    tdf = pd.read_csv(train_annotations_path)
    pairs = list(zip(tdf["verb_class"].astype(int), tdf["noun_class"].astype(int)))
    tactions = set(pairs)
    tverbs = {v for v, _ in tactions}
    tnouns = {n for _, n in tactions}

    verb_classes = {k: i for i, k in enumerate(tverbs)}
    noun_classes = {k: i for i, k in enumerate(tnouns)}
    action_classes = {k: i for i, k in enumerate(tactions)}

    if verb_classes_csv and Path(verb_classes_csv).is_file():
        verb_text = dict(zip(pd.read_csv(verb_classes_csv)["verb_class"], pd.read_csv(verb_classes_csv)["key"]))
    else:
        verb_text = dict(zip(tdf["verb_class"], tdf["verb"]))
    if noun_classes_csv and Path(noun_classes_csv).is_file():
        noun_text = dict(zip(pd.read_csv(noun_classes_csv)["noun_class"], pd.read_csv(noun_classes_csv)["key"]))
    else:
        noun_text = dict(zip(tdf["noun_class"], tdf["noun"]))

    return LabelMapping(
        verb_classes=verb_classes,
        noun_classes=noun_classes,
        action_classes=action_classes,
        verb_text=verb_text,
        noun_text=noun_text,
    )
