"""Tests for action label mapping behavior."""

from __future__ import annotations

import pytest

from ego.common.exceptions import EgoLabelMappingError
from ego.datasets.label_mapping import (
    build_label_mapping,
    check_mapping_covers_split,
    filter_to_known_pairs,
)

TRAIN_PAIRS = [(3, 12), (3, 12), (0, 1), (5, 5), (0, 1), (2, 9)]


def test_same_verb_noun_pair_maps_to_same_action_id():
    mapping = build_label_mapping(TRAIN_PAIRS)
    first = mapping.encode_action(3, 12)
    second = mapping.encode_action(3, 12)
    assert first == second


def test_mapping_is_deterministic_regardless_of_row_order():
    shuffled = list(reversed(TRAIN_PAIRS))
    m1 = build_label_mapping(TRAIN_PAIRS)
    m2 = build_label_mapping(shuffled)
    assert m1.verb_classes == m2.verb_classes
    assert m1.noun_classes == m2.noun_classes
    assert m1.action_classes == m2.action_classes


def test_verb_and_noun_ids_are_assigned_independently_not_swapped():
    mapping = build_label_mapping(TRAIN_PAIRS)
    # verb ids {0, 2, 3, 5} and noun ids {1, 5, 9, 12} are disjoint dense
    # spaces of the same size here; make sure encode_verb/encode_noun don't
    # accidentally share one counter (which would make verb 5 == noun 5).
    assert mapping.num_verbs == len({v for v, _ in TRAIN_PAIRS})
    assert mapping.num_nouns == len({n for _, n in TRAIN_PAIRS})
    assert set(mapping.verb_classes.values()) == set(range(mapping.num_verbs))
    assert set(mapping.noun_classes.values()) == set(range(mapping.num_nouns))


def test_train_and_val_mapping_are_identical_when_shared():
    train_mapping = build_label_mapping(TRAIN_PAIRS)
    val_rows = [{"verb_id": 3, "noun_id": 12}, {"verb_id": 0, "noun_id": 1}]
    # Should not raise: both pairs were seen in train.
    check_mapping_covers_split(train_mapping, [(r["verb_id"], r["noun_id"]) for r in val_rows])


def test_unknown_class_is_rejected_explicitly():
    mapping = build_label_mapping(TRAIN_PAIRS)
    with pytest.raises(EgoLabelMappingError):
        mapping.encode_verb(999)
    with pytest.raises(EgoLabelMappingError):
        mapping.encode_noun(999)
    with pytest.raises(EgoLabelMappingError):
        mapping.encode_action(999, 999)
    with pytest.raises(EgoLabelMappingError):
        check_mapping_covers_split(mapping, [(999, 1)], split_name="test")


def test_filter_to_known_pairs_drops_unseen_val_rows():
    known = set(TRAIN_PAIRS)
    rows = [
        {"verb_id": 3, "noun_id": 12},  # seen
        {"verb_id": 42, "noun_id": 42},  # unseen
    ]
    kept = filter_to_known_pairs(rows, known)
    assert kept == [{"verb_id": 3, "noun_id": 12}]


def test_cannot_build_mapping_from_empty_train_split():
    with pytest.raises(EgoLabelMappingError):
        build_label_mapping([])
