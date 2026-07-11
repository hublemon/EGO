"""Tests for Ego4D LTA class-distribution statistics helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from ego.datasets.ego4d_stats import (
    build_pilot_taxonomy,
    class_frequency,
    gini_coefficient,
    head_mid_tail_bands,
    imbalance_ratio,
    scenario_distribution,
    verb_noun_cooccurrence,
)


def test_class_frequency_counts_correctly():
    df = pd.DataFrame({"verb_label": [0, 0, 1, 2, 2, 2]})
    freq = class_frequency(df, "verb_label")
    assert freq == {0: 2, 1: 1, 2: 3}


def test_head_mid_tail_bands_partitions_all_classes():
    freq = {i: (10 - i) for i in range(10)}  # class 0 most frequent .. class 9 least
    bands = head_mid_tail_bands(freq, head_frac=0.2, tail_frac=0.5)
    assert set(bands.keys()) == set(freq.keys())
    assert set(bands.values()) <= {"head", "mid", "tail"}
    # most frequent classes are head, least frequent are tail
    assert bands[0] == "head"
    assert bands[9] == "tail"


def test_head_mid_tail_bands_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        head_mid_tail_bands({0: 1}, head_frac=0.6, tail_frac=0.6)


def test_gini_coefficient_zero_for_uniform_distribution():
    values = [10, 10, 10, 10]
    assert gini_coefficient(values) == pytest.approx(0.0, abs=1e-9)


def test_gini_coefficient_high_for_skewed_distribution():
    values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]
    assert gini_coefficient(values) > 0.5


def test_imbalance_ratio():
    assert imbalance_ratio([5, 10, 50]) == 10.0
    assert imbalance_ratio([]) != imbalance_ratio([])  # NaN != NaN


def test_verb_noun_cooccurrence_shape():
    df = pd.DataFrame({"verb_label": [0, 0, 1], "noun_label": [0, 1, 1]})
    cooc = verb_noun_cooccurrence(df)
    assert cooc.loc[0, 0] == 1
    assert cooc.loc[0, 1] == 1
    assert cooc.loc[1, 1] == 1


def test_scenario_distribution():
    df = pd.DataFrame({"scenario": ["Cooking", "Cooking", "Cleaning"]})
    assert scenario_distribution(df) == {"Cooking": 2, "Cleaning": 1}


def test_build_pilot_taxonomy_exclude_mode_drops_rows():
    df = pd.DataFrame(
        {
            "verb_label": [0, 0, 1, 2, 2],
            "noun_label": [0, 0, 1, 2, 2],
        }
    )
    filtered, info = build_pilot_taxonomy(df, top_verb=1, top_noun=1, mode="exclude")
    assert info["rows_before"] == 5
    assert info["rows_after"] == 2  # only the two verb=0,noun=0 rows survive
    assert set(filtered["verb_label"]) == {0}


def test_build_pilot_taxonomy_other_mode_preserves_row_count():
    df = pd.DataFrame({"verb_label": [0, 1, 2], "noun_label": [0, 1, 2]})
    filtered, info = build_pilot_taxonomy(df, top_verb=1, top_noun=1, mode="other")
    assert info["rows_after"] == len(df)
    assert set(filtered["verb_label"]) == {0, -1}


def test_build_pilot_taxonomy_rejects_bad_mode():
    df = pd.DataFrame({"verb_label": [0], "noun_label": [0]})
    with pytest.raises(ValueError):
        build_pilot_taxonomy(df, top_verb=1, top_noun=1, mode="bogus")
