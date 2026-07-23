"""Synthetic tests for leakage-conscious Phase-1/P0-a evaluation.

Run without pytest:
    python -m unittest tests.unit.test_goalstep_history_vs_p0a
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "step1" / "goalstep" / "evaluate_history_context_vs_p0a.py"
SPEC = importlib.util.spec_from_file_location("goalstep_history_vs_p0a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def _rank_logits(labels: torch.Tensor, classes: int, good_mask: torch.Tensor) -> torch.Tensor:
    """Make labels rank first on ``good_mask`` and last elsewhere."""
    logits = torch.arange(classes, dtype=torch.float32).repeat(len(labels), 1)
    rows = torch.arange(len(labels))
    logits[rows[good_mask], labels[good_mask]] = float(classes + 2)
    logits[rows[~good_mask], labels[~good_mask]] = float(-classes - 2)
    return logits


class HistoryVsP0ATest(unittest.TestCase):
    def _write_fixture(self, root: Path, *, permute_p0a: bool = False) -> tuple[Path, Path, Path]:
        samples = [f"sample_{index}" for index in range(8)]
        videos = [f"video_{index}" for index in range(8)]
        folds = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        labels = torch.zeros(8, dtype=torch.int64)
        classes = {head: 7 for head in evaluation.HEADS}
        all_good = torch.ones(8, dtype=torch.bool)
        raw_logits = _rank_logits(labels, 7, all_good)
        visual_logits = _rank_logits(labels, 7, torch.zeros(8, dtype=torch.bool))
        endpoint = {
            "logical_sample_ids": samples,
            "source_cache_sample_ids": samples,
            "video_uids": videos,
            "labels": {head: labels.clone() for head in evaluation.HEADS},
            "num_classes": classes,
            "candidates": {
                "next_ep01": {
                    "logits": {head: raw_logits.clone() for head in evaluation.HEADS}
                },
                "next_ep02": {
                    "logits": {head: raw_logits.clone() for head in evaluation.HEADS}
                },
                "next_ep03": {
                    "logits": {head: visual_logits.clone() for head in evaluation.HEADS}
                },
            },
        }
        endpoint_path = root / "endpoint.pt"
        torch.save(endpoint, endpoint_path)

        probabilities = torch.softmax(raw_logits, dim=-1)
        selections = {
            head: [
                {
                    "test_fold": 0,
                    "tune_fold": 1,
                    "selected_with_replacement": ["next_ep01"],
                },
                {
                    "test_fold": 1,
                    "tune_fold": 0,
                    "selected_with_replacement": ["next_ep02"],
                },
            ]
            for head in evaluation.HEADS
        }
        p0a_samples = list(reversed(samples)) if permute_p0a else samples
        p0a = {
            "format_version": 1,
            "deployable_at_A2_boundary": True,
            "sample_ids": p0a_samples,
            "video_uids": videos,
            "folds": folds,
            "labels": {head: labels.clone() for head in evaluation.HEADS},
            "oof_scores": {
                head: probabilities.clone() for head in evaluation.HEADS
            },
            "selections": selections,
        }
        p0a_path = root / "p0a.pt"
        torch.save(p0a, p0a_path)

        predictions_dir = root / "predictions"
        predictions_dir.mkdir()
        # Epoch 1 is good on fold 0 and bad on fold 1. Epoch 2 is the reverse.
        # Cross-fit must therefore select epoch 2 for test fold 0 (tuned on 1)
        # and epoch 1 for test fold 1 (tuned on 0).
        epoch_good_masks = {
            0: all_good,
            1: folds == 0,
            2: folds == 1,
        }
        for epoch, good_mask in epoch_good_masks.items():
            fused = visual_logits.clone() if epoch == 0 else _rank_logits(labels, 7, good_mask)
            logits = {
                "visual": {head: visual_logits.clone() for head in evaluation.HEADS},
                "history": {head: fused.clone() for head in evaluation.HEADS},
                "current_only": {head: visual_logits.clone() for head in evaluation.HEADS},
                "fused": {head: fused.clone() for head in evaluation.HEADS},
            }
            artifact = {
                "format_version": 1,
                "kind": evaluation.PREDICTION_KIND,
                "epoch": epoch,
                "contract": evaluation.CONTRACT,
                "sample_ids": samples,
                "video_uids": videos,
                "history_lengths": torch.arange(8, dtype=torch.int64),
                "labels": {head: labels.clone() for head in evaluation.HEADS},
                "logits": logits,
                "num_classes": classes,
                "gate_values": {},
            }
            torch.save(artifact, predictions_dir / f"epoch_{epoch:02d}.pt")
        return endpoint_path, p0a_path, predictions_dir

    def test_crossfit_uses_opposite_fold_for_epoch_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, predictions = self._write_fixture(root)
            results = evaluation.evaluate_artifacts(
                endpoint_path=endpoint,
                p0a_path=p0a,
                predictions_dir=predictions,
                output_json=root / "results.json",
                output_scores=root / "scores.pt",
                expected_last_epoch=2,
                alpha_step=0.5,
                bootstrap_samples=40,
                seed=7,
            )
            action = results["selection_protocol"]["fieldwise"]["action"]
            self.assertEqual(action[0]["test_fold"], 0)
            self.assertEqual(action[0]["phase1_epoch"], 2)
            self.assertEqual(action[1]["test_fold"], 1)
            self.assertEqual(action[1]["phase1_epoch"], 1)
            self.assertEqual(
                results["audit"]["p0a_raw_reconstruction_max_abs_error"]["action"],
                0.0,
            )
            self.assertTrue((root / "results.json").is_file())
            self.assertTrue((root / "scores.pt").is_file())

    def test_alignment_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, predictions = self._write_fixture(root, permute_p0a=True)
            with self.assertRaisesRegex(ValueError, "sample order differ"):
                evaluation.evaluate_artifacts(
                    endpoint_path=endpoint,
                    p0a_path=p0a,
                    predictions_dir=predictions,
                    output_json=root / "results.json",
                    output_scores=root / "scores.pt",
                    expected_last_epoch=2,
                    alpha_step=0.5,
                    bootstrap_samples=10,
                )

    def test_clustered_bootstrap_counts_discordant_samples(self) -> None:
        labels = torch.tensor([0, 0, 0, 0])
        correct = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 0.0])
        incorrect = torch.tensor([0.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        challenger = torch.stack([correct, correct, incorrect, incorrect])
        baseline = torch.stack([correct, incorrect, correct, incorrect])
        result = evaluation._paired_video_bootstrap(
            challenger,
            baseline,
            labels,
            ["a", "b", "c", "d"],
            seed=3,
            bootstrap_samples=50,
        )
        self.assertEqual(result["challenger_only_correct"], 1)
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["delta_top5_pp"], 0.0)


if __name__ == "__main__":
    unittest.main()
