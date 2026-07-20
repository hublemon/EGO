"""Tests for the scenario-stratified sampler used by train_lta_z1.py."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "step1" / "ego4d_lta"))
from train_lta_z1 import ScenarioStratifiedSampler  # noqa: E402


def test_yields_exactly_one_epoch_worth_of_indices():
    scenarios = ["A"] * 10 + ["B"] * 3 + ["C"] * 1
    sampler = ScenarioStratifiedSampler(scenarios, seed=0)
    order = list(sampler)
    assert len(order) == len(scenarios)
    assert all(0 <= i < len(scenarios) for i in order)


def test_small_scenarios_are_not_starved_relative_to_large_ones():
    scenarios = ["A"] * 100 + ["B"] * 2
    sampler = ScenarioStratifiedSampler(scenarios, seed=0)
    order = list(sampler)
    drawn_scenarios = [scenarios[i] for i in order]
    counts = Counter(drawn_scenarios)
    # round-robin cycling means A and B get drawn roughly the same number of
    # times per epoch, unlike i.i.d. sampling which would draw B ~2% as often.
    assert counts["A"] > 0 and counts["B"] > 0
    assert counts["B"] / counts["A"] > 0.5


def test_is_deterministic_given_seed_and_epoch():
    scenarios = ["A", "B", "C"] * 10
    sampler1 = ScenarioStratifiedSampler(scenarios, seed=42)
    sampler2 = ScenarioStratifiedSampler(scenarios, seed=42)
    sampler1.set_epoch(3)
    sampler2.set_epoch(3)
    assert list(sampler1) == list(sampler2)


def test_different_epochs_give_different_orders():
    scenarios = ["A", "B", "C"] * 20
    sampler = ScenarioStratifiedSampler(scenarios, seed=42)
    sampler.set_epoch(0)
    order0 = list(sampler)
    sampler.set_epoch(1)
    order1 = list(sampler)
    assert order0 != order1


def test_len_matches_dataset_size():
    scenarios = ["A"] * 7
    assert len(ScenarioStratifiedSampler(scenarios)) == 7
