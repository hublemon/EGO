"""Tests for Step 1 evaluation metric functions."""

from __future__ import annotations

import math

import torch

from ego.step1_action_anticipation.metrics import (
    candidate_coverage,
    class_mean_recall,
    prediction_entropy,
    top_k_recall,
)


def test_top_k_recall_counts_hits_within_k():
    # 3 samples, 4 classes. Labels: 0, 1, 2.
    logits = torch.tensor(
        [
            [5.0, 1.0, 1.0, 1.0],  # top-1 is class 0 -> hit for label 0
            [1.0, 5.0, 4.0, 1.0],  # top-2 are {1, 2} -> hit for label 1 (top1)
            [1.0, 5.0, 4.0, 1.0],  # label 2 is rank-2 -> miss at k=1, hit at k=2
        ]
    )
    labels = torch.tensor([0, 1, 2])
    assert abs(top_k_recall(logits, labels, k=1) - 200 / 3) < 1e-4
    assert top_k_recall(logits, labels, k=2) == 100.0


def test_class_mean_recall_averages_per_class_not_per_instance():
    # Class 0 has 3 samples (all hit), class 1 has 1 sample (miss).
    # Instance-level recall would be 3/4 = 75%; class-mean should be
    # (100 + 0) / 2 = 50%, since class 1 gets equal weight despite fewer samples.
    logits = torch.tensor(
        [
            [5.0, 1.0],
            [5.0, 1.0],
            [5.0, 1.0],
            [5.0, 1.0],  # true label is 1, but class 0 wins -> miss
        ]
    )
    labels = torch.tensor([0, 0, 0, 1])
    cmr = class_mean_recall(logits, labels, num_classes=2, k=1)
    assert abs(cmr - 50.0) < 1e-6


def test_candidate_coverage_counts_unique_classes_in_topk():
    logits = torch.tensor(
        [
            [5.0, 1.0, 1.0, 1.0],
            [1.0, 5.0, 1.0, 1.0],
        ]
    )
    # top-1 covers classes {0, 1} out of 4 -> 50% coverage
    coverage = candidate_coverage(logits, num_classes=4, k=1)
    assert abs(coverage - 50.0) < 1e-6


def test_prediction_entropy_uniform_distribution_is_log_c():
    logits = torch.zeros(1, 4)  # uniform softmax over 4 classes
    entropy = prediction_entropy(logits)
    assert abs(entropy.item() - math.log(4)) < 1e-5


def test_prediction_entropy_peaked_distribution_is_near_zero():
    logits = torch.tensor([[50.0, 0.0, 0.0, 0.0]])
    entropy = prediction_entropy(logits)
    assert entropy.item() < 1e-3
