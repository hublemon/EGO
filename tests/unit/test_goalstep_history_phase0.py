"""Stdlib tests for the pure history Phase-0 selection code.

Run without pytest:
    python -m unittest tests.unit.test_goalstep_history_phase0
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "step1" / "goalstep" / "run_history_phase0.py"
SPEC = importlib.util.spec_from_file_location("goalstep_history_phase0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
phase0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase0
SPEC.loader.exec_module(phase0)


class HistoryPhase0Test(unittest.TestCase):
    def test_video_folds_are_stable_and_disjoint(self) -> None:
        videos = ["a", "a", "b", "c", "c", "d"]
        first = phase0._stable_video_folds(videos, seed=42)
        second = phase0._stable_video_folds(videos, seed=42)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first[0].item(), first[1].item())
        self.assertEqual(first[3].item(), first[4].item())

    def test_transition_matrix_uses_counts_and_global_backoff(self) -> None:
        train = pd.DataFrame(
            {
                "observed_action_label": [0, 0, 1, 1],
                "action_label": [1, 1, 0, 2],
            }
        )
        counts, row_counts, prior = phase0._transition_components(train, 3)
        matrix = phase0._transition_matrix(counts, row_counts, prior, alpha=0.0)
        self.assertTrue(torch.allclose(matrix.sum(1), torch.ones(3)))
        self.assertEqual(matrix[0].argmax().item(), 1)
        self.assertTrue(torch.allclose(matrix[2], prior.float()))

    def test_caruana_prefers_perfect_candidate(self) -> None:
        labels = torch.tensor([0, 1, 2, 3])
        good = torch.full((4, 6), 0.1)
        good[torch.arange(4), labels] = 0.5
        bad = torch.full((4, 6), 0.2)
        bad[torch.arange(4), labels] = 0.0
        probabilities = torch.stack([bad, good])
        selected, _ = phase0._caruana_select(
            probabilities,
            labels,
            ["bad", "good"],
            num_classes=6,
            rounds=3,
            objective="top5",
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], 1)
        self.assertEqual(phase0.top_k_recall(good, labels, k=1), 100.0)

    def test_paired_delta_counts_discordant_samples(self) -> None:
        labels = torch.tensor([0, 0, 0, 0])
        correct = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 0.0])
        incorrect = torch.tensor([0.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        challenger = torch.stack([correct, correct, incorrect, incorrect])
        baseline = torch.stack([correct, incorrect, correct, incorrect])
        result = phase0._paired_delta(
            challenger, baseline, labels, ["a", "b", "c", "d"], seed=1, bootstrap_samples=20
        )
        self.assertEqual(result["challenger_only_correct"], 1)
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["delta_top5_pp"], 0.0)


if __name__ == "__main__":
    unittest.main()
