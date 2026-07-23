"""Synthetic tests for the leakage-safe Phase-2 outer-fold evaluator.

Run without pytest:
    python -m unittest tests.unit.test_goalstep_history_probe_zoo_evaluator
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts"
    / "step1"
    / "goalstep"
    / "evaluate_history_probe_zoo_vs_p0a.py"
)
SPEC = importlib.util.spec_from_file_location("goalstep_history_probe_zoo_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def _rank_logits(
    labels: torch.Tensor, classes: int, good_mask: torch.Tensor
) -> torch.Tensor:
    logits = torch.arange(classes, dtype=torch.float32).repeat(len(labels), 1)
    rows = torch.arange(len(labels))
    logits[rows[good_mask], labels[good_mask]] = float(classes + 2)
    logits[rows[~good_mask], labels[~good_mask]] = float(-classes - 2)
    return logits


class Phase2ZooEvaluatorTest(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
        *,
        expected_epochs: int = 3,
        strong_phase1_incumbent: bool = True,
    ) -> tuple[Path, Path, Path, Path, Path]:
        sample_ids = [f"sample_{index}" for index in range(12)]
        video_uids = [f"video_{index}" for index in range(12)]
        folds = torch.tensor([0] * 6 + [1] * 6, dtype=torch.int64)
        labels = torch.zeros(12, dtype=torch.int64)
        num_classes = {head: 7 for head in evaluation.HEADS}
        all_bad = torch.zeros(12, dtype=torch.bool)
        p0a_good = torch.zeros(12, dtype=torch.bool)
        p0a_good[[0, 6]] = True

        raw_logits = _rank_logits(labels, 7, p0a_good)
        visual_logits = _rank_logits(labels, 7, all_bad)
        endpoint = {
            "format_version": 1,
            "contract": evaluation.ENDPOINT_CONTRACT,
            "logical_sample_ids": sample_ids,
            "source_cache_sample_ids": sample_ids,
            "video_uids": video_uids,
            "labels": {head: labels.clone() for head in evaluation.HEADS},
            "num_classes": num_classes,
            "candidates": {
                name: {
                    "logits": {
                        head: raw_logits.clone() for head in evaluation.HEADS
                    }
                }
                for name in evaluation.P0A_CANDIDATES
            },
        }
        # The visual residual source is endpoint epoch 3 and stays frozen.
        endpoint["candidates"]["next_ep03"] = {
            "logits": {
                head: visual_logits.clone() for head in evaluation.HEADS
            }
        }
        endpoint_path = root / "endpoint.pt"
        torch.save(endpoint, endpoint_path)

        p0a_scores = {
            head: torch.empty(12, 7, dtype=torch.float32)
            for head in evaluation.HEADS
        }
        selections = {head: [] for head in evaluation.HEADS}
        for head in evaluation.HEADS:
            probabilities = torch.stack(
                [
                    torch.softmax(endpoint["candidates"][name]["logits"][head], dim=-1)
                    for name in evaluation.P0A_CANDIDATES
                ]
            )
            for test_fold in (0, 1):
                tune_fold = 1 - test_fold
                tune_mask = folds == tune_fold
                test_mask = folds == test_fold
                selected, _ = evaluation._caruana_select(
                    probabilities[:, tune_mask],
                    labels[tune_mask],
                    evaluation.P0A_CANDIDATES,
                    num_classes=7,
                )
                p0a_scores[head][test_mask] = probabilities[selected][:, test_mask].mean(0)
                selections[head].append(
                    {
                        "test_fold": test_fold,
                        "tune_fold": tune_fold,
                        "selected_with_replacement": [
                            evaluation.P0A_CANDIDATES[index] for index in selected
                        ],
                    }
                )
        p0a = {
            "format_version": 1,
            "deployable_at_A2_boundary": True,
            "sample_ids": sample_ids,
            "video_uids": video_uids,
            "folds": folds,
            "labels": {head: labels.clone() for head in evaluation.HEADS},
            "oof_scores": p0a_scores,
            "selections": selections,
        }
        p0a_path = root / "p0a.pt"
        torch.save(p0a, p0a_path)

        incumbent_good = (
            torch.ones(12, dtype=torch.bool)
            if strong_phase1_incumbent
            else torch.zeros(12, dtype=torch.bool)
        )
        incumbent_probability = torch.softmax(
            _rank_logits(labels, 7, incumbent_good), dim=-1
        )
        phase1_scores = {
            "p0a": {head: p0a_scores[head].clone() for head in evaluation.HEADS},
            "phase1": {
                head: incumbent_probability.clone() for head in evaluation.HEADS
            },
            "final_blend": {
                head: incumbent_probability.clone() for head in evaluation.HEADS
            },
            "visual_same_epoch": {
                head: torch.softmax(visual_logits, dim=-1).clone()
                for head in evaluation.HEADS
            },
            "history_same_epoch": {
                head: incumbent_probability.clone() for head in evaluation.HEADS
            },
            "current_only_same_epoch": {
                head: torch.softmax(visual_logits, dim=-1).clone()
                for head in evaluation.HEADS
            },
        }
        phase1_run_dir = root / "phase1"
        phase1_run_dir.mkdir(parents=True)
        phase1_oof_path = phase1_run_dir / evaluation.PHASE1_OOF_FILENAME
        torch.save(
            {
                "format_version": 1,
                "kind": evaluation.PHASE1_OOF_KIND,
                "contract": evaluation.CONTRACT,
                "sample_ids": sample_ids,
                "video_uids": video_uids,
                "folds": folds,
                "labels": {head: labels.clone() for head in evaluation.HEADS},
                "num_classes": num_classes,
                "scores": phase1_scores,
                "selections": {},
                "metrics_percent": {},
            },
            phase1_oof_path,
        )

        default_dir = phase1_run_dir / "val_predictions"
        zoo_dir = root / "zoo"
        default_dir.mkdir(parents=True)
        (zoo_dir / "arms").mkdir(parents=True)
        specs = []
        for grid_index, (learning_rate, weight_decay) in enumerate(
            evaluation.REGISTERED_GRID
        ):
            specs.append(
                {
                    "arm_id": evaluation._arm_id(learning_rate, weight_decay),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "grid_index": grid_index,
                }
            )
        default_id = evaluation._arm_id(*evaluation.REGISTERED_DEFAULT)
        fingerprint = "pending"

        def write_prediction(
            path: Path,
            *,
            epoch: int,
            arm: dict,
            is_default: bool,
        ) -> None:
            # Tune fold 0 ranks epoch 1 first; tune fold 1 ranks epoch 2 first.
            if epoch == 1:
                fused_good = (folds == 0) | torch.tensor(
                    [False] * 6 + [True, True, True, False, False, False]
                )
            elif epoch == 2:
                fused_good = (folds == 1) | torch.tensor(
                    [True, True, True, False, False, False] + [False] * 6
                )
            else:
                fused_good = torch.arange(12) % 2 == 0
            current_good = ~fused_good
            fused = _rank_logits(labels, 7, fused_good)
            current = _rank_logits(labels, 7, current_good)
            logits = {
                "visual": {
                    head: visual_logits.clone() for head in evaluation.HEADS
                },
                "history": {head: fused.clone() for head in evaluation.HEADS},
                "current_only": {
                    head: current.clone() for head in evaluation.HEADS
                },
                "fused": {head: fused.clone() for head in evaluation.HEADS},
            }
            artifact = {
                "format_version": 1,
                "kind": evaluation.PREDICTION_KIND,
                "epoch": epoch,
                "contract": evaluation.CONTRACT,
                "sample_ids": sample_ids,
                "video_uids": video_uids,
                "history_lengths": torch.arange(12, dtype=torch.int64) % 9,
                "labels": {
                    head: labels.clone() for head in evaluation.HEADS
                },
                "logits": logits,
                "num_classes": num_classes,
                "gate_values": {},
            }
            if not is_default:
                artifact.update(
                    {
                        "phase": "P2",
                        "arm_id": arm["arm_id"],
                        "learning_rate": arm["learning_rate"],
                        "weight_decay": arm["weight_decay"],
                        "provenance_fingerprint": fingerprint,
                    }
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, path)

        default_spec = next(spec for spec in specs if spec["arm_id"] == default_id)
        # Phase 1 retains the mandatory epoch-0 fallback as well.
        for epoch in range(expected_epochs + 1):
            write_prediction(
                default_dir / f"epoch_{epoch:02d}.pt",
                epoch=epoch,
                arm=default_spec,
                is_default=True,
            )
        default_inventory = {
            "run_dir": str(default_dir.parent),
            "files": [
                {
                    "path": str(phase1_oof_path),
                    "sha256": evaluation._sha256(phase1_oof_path),
                    "bytes": phase1_oof_path.stat().st_size,
                },
                *[
                    {
                        "path": str(default_dir / f"epoch_{epoch:02d}.pt"),
                        "sha256": evaluation._sha256(
                            default_dir / f"epoch_{epoch:02d}.pt"
                        ),
                        "bytes": (
                            default_dir / f"epoch_{epoch:02d}.pt"
                        ).stat().st_size,
                    }
                    for epoch in range(expected_epochs + 1)
                ],
            ],
        }
        provenance = {
            "format_version": 1,
            "kind": "goalstep_history_probe_zoo_provenance",
            "contract": "synthetic strict-future history contract",
            "feature_reextraction": False,
            "config": {"path": "synthetic-zoo.yaml", "sha256": "config-hash"},
            "phase1_config": {"path": "synthetic-phase1.yaml", "sha256": "phase1-hash"},
            "store_manifest": {"path": "synthetic-store.json", "sha256": "store-hash"},
            "indices": {
                "train": {"path": "synthetic-train.csv", "sha256": "train-hash"},
                "val": {"path": "synthetic-val.csv", "sha256": "val-hash"},
            },
            "default_phase1": default_inventory,
            "train_rows": 20,
            "val_rows": len(sample_ids),
            "num_classes": num_classes,
            "summary_shape": [3, 8],
            "max_history": 8,
            "seed": 42,
            "epochs": expected_epochs,
            "registered_grid": specs,
            "skipped_default_arm": default_spec,
        }
        fingerprint = evaluation._fingerprint(provenance)
        for arm in specs:
            if arm["arm_id"] == default_id:
                continue
            directory = zoo_dir / "arms" / arm["arm_id"] / "val_predictions"
            for epoch in range(1, expected_epochs + 1):
                write_prediction(
                    directory / f"epoch_{epoch:02d}.pt",
                    epoch=epoch,
                    arm=arm,
                    is_default=False,
                )

        manifest = {
            **provenance,
            "trained_arm_count": 11,
            "total_grid_arm_count": 12,
            "provenance_fingerprint": fingerprint,
        }
        (zoo_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (zoo_dir / "final_metrics.json").write_text(
            json.dumps(
                {
                    "kind": "goalstep_history_probe_zoo_result",
                    "status": "complete",
                    "completed_epoch": expected_epochs,
                    "epochs": expected_epochs,
                    "provenance_fingerprint": fingerprint,
                }
            ),
            encoding="utf-8",
        )
        return endpoint_path, p0a_path, phase1_oof_path, default_dir, zoo_dir

    def test_outer_fold_prefilter_caruana_and_blend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, phase1_oof, default_dir, zoo_dir = self._write_fixture(root)
            result = evaluation.evaluate_phase2(
                endpoint_path=endpoint,
                p0a_path=p0a,
                phase1_oof_path=phase1_oof,
                default_predictions_dir=default_dir,
                zoo_run_dir=zoo_dir,
                output_json=root / "result.json",
                output_scores=root / "scores.pt",
                expected_epochs=3,
                alpha_step=0.5,
                bootstrap_samples=50,
                seed=9,
            )
            action_folds = result["selection_protocol"][
                "fieldwise_outer_fold_selections"
            ]["action"]
            default_id = evaluation._arm_id(*evaluation.REGISTERED_DEFAULT)
            # test fold 0 tunes on fold 1, where fused epoch 2 is best.
            self.assertEqual(
                action_folds[0]["phase2_fused"]["per_arm_top2_prefilter"][
                    default_id
                ]["selected_top2_epochs"][0],
                2,
            )
            # current_only is selected independently and prefers epoch 1 there.
            self.assertEqual(
                action_folds[0]["phase2_current_only"]["per_arm_top2_prefilter"][
                    default_id
                ]["selected_top2_epochs"][0],
                1,
            )
            self.assertEqual(
                action_folds[0]["phase2_fused"]["prefilter_candidate_count"], 24
            )
            self.assertEqual(
                result["audit"]["p0a_raw_reconstruction_max_abs_error"]["action"],
                0.0,
            )
            self.assertTrue(result["audit"]["inherited_validation_adaptivity"]["present"])
            self.assertTrue(
                result["inputs"]["phase2"]["frozen_phase1_oof"][
                    "canonical_path_exact"
                ]
            )
            self.assertTrue(
                result["inputs"]["phase2"]["frozen_phase1_oof"]["sha256_exact"]
            )
            self.assertFalse(result["decisions"]["confirmatory_claim_allowed"])
            self.assertGreater(
                result["metrics_percent"]["final_blend"]["action"]["top5"],
                result["metrics_percent"]["p0a"]["action"]["top5"],
            )
            self.assertLess(
                result["metrics_percent"]["final_blend"]["action"]["top5"],
                result["metrics_percent"]["phase1_incumbent"]["action"]["top5"],
            )
            self.assertFalse(result["phase2_promotion"]["promoted"])
            self.assertEqual(
                result["phase2_promotion"]["champion_after_phase2"],
                "phase1_final_blend_incumbent_retained",
            )
            self.assertTrue((root / "result.json").is_file())
            self.assertTrue((root / "scores.pt").is_file())
            scores = torch.load(root / "scores.pt", map_location="cpu", weights_only=True)
            self.assertEqual(
                set(scores["scores"]),
                {
                    "p0a",
                    "phase1_incumbent",
                    "phase2_selected",
                    "current_only_control",
                    "final_blend",
                },
            )

    def test_phase2_promotes_only_when_it_beats_phase1_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, phase1_oof, default_dir, zoo_dir = self._write_fixture(
                root, strong_phase1_incumbent=False
            )
            result = evaluation.evaluate_phase2(
                endpoint_path=endpoint,
                p0a_path=p0a,
                phase1_oof_path=phase1_oof,
                default_predictions_dir=default_dir,
                zoo_run_dir=zoo_dir,
                output_json=root / "result.json",
                output_scores=root / "scores.pt",
                expected_epochs=3,
                alpha_step=0.5,
                bootstrap_samples=500,
                seed=17,
            )
            self.assertTrue(result["phase2_promotion"]["promoted"])
            self.assertEqual(
                result["phase2_promotion"]["champion_after_phase2"],
                "phase2_final_blend_promoted_provisionally",
            )
            self.assertTrue(
                result["decisions"][
                    "phase2_final_blend_over_phase1_incumbent"
                ]["outer_fold_rule_passed"]
            )

    def test_arm_alignment_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, phase1_oof, default_dir, zoo_dir = self._write_fixture(root)
            arm_dir = next((zoo_dir / "arms").iterdir())
            path = arm_dir / "val_predictions" / "epoch_01.pt"
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            artifact["sample_ids"] = list(reversed(artifact["sample_ids"]))
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "sample order mismatch"):
                evaluation.evaluate_phase2(
                    endpoint_path=endpoint,
                    p0a_path=p0a,
                    phase1_oof_path=phase1_oof,
                    default_predictions_dir=default_dir,
                    zoo_run_dir=zoo_dir,
                    output_json=root / "result.json",
                    output_scores=root / "scores.pt",
                    expected_epochs=3,
                    alpha_step=0.5,
                    bootstrap_samples=10,
                )

    def test_exact_phase1_oof_matches_frozen_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, phase1_oof, _, zoo_dir = self._write_fixture(root)
            manifest = json.loads(
                (zoo_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            audit = evaluation._validate_frozen_phase1_oof_inventory(
                manifest, phase1_oof
            )
            self.assertTrue(audit["inventory_entry_unique"])
            self.assertTrue(audit["canonical_path_exact"])
            self.assertTrue(audit["sha256_exact"])
            self.assertTrue(audit["bytes_exact"])

    def test_stored_p0a_recipe_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a_path, phase1_oof, default_dir, zoo_dir = self._write_fixture(root)
            p0a = torch.load(p0a_path, map_location="cpu", weights_only=True)
            p0a["selections"]["action"][0]["selected_with_replacement"] = [
                "next_ep08"
            ]
            torch.save(p0a, p0a_path)
            with self.assertRaisesRegex(ValueError, "Relearned P0-a recipe differs"):
                evaluation.evaluate_phase2(
                    endpoint_path=endpoint,
                    p0a_path=p0a_path,
                    phase1_oof_path=phase1_oof,
                    default_predictions_dir=default_dir,
                    zoo_run_dir=zoo_dir,
                    output_json=root / "result.json",
                    output_scores=root / "scores.pt",
                    expected_epochs=3,
                    alpha_step=0.5,
                    bootstrap_samples=10,
                )

    def test_tampered_phase1_oof_fails_frozen_sha_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, phase1_oof, default_dir, zoo_dir = self._write_fixture(root)
            artifact = torch.load(phase1_oof, map_location="cpu", weights_only=True)
            artifact["metrics_percent"] = {"tampered": True}
            torch.save(artifact, phase1_oof)
            with self.assertRaisesRegex(ValueError, "SHA-256 differs from the frozen"):
                evaluation.evaluate_phase2(
                    endpoint_path=endpoint,
                    p0a_path=p0a,
                    phase1_oof_path=phase1_oof,
                    default_predictions_dir=default_dir,
                    zoo_run_dir=zoo_dir,
                    output_json=root / "result.json",
                    output_scores=root / "scores.pt",
                    expected_epochs=3,
                    alpha_step=0.5,
                    bootstrap_samples=10,
                )

    def test_replaced_phase1_oof_path_fails_frozen_path_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint, p0a, phase1_oof, default_dir, zoo_dir = self._write_fixture(root)
            replacement_dir = root / "replacement"
            replacement_dir.mkdir()
            replacement = replacement_dir / evaluation.PHASE1_OOF_FILENAME
            shutil.copyfile(phase1_oof, replacement)
            self.assertEqual(
                evaluation._sha256(phase1_oof), evaluation._sha256(replacement)
            )
            with self.assertRaisesRegex(ValueError, "input path differs from the frozen"):
                evaluation.evaluate_phase2(
                    endpoint_path=endpoint,
                    p0a_path=p0a,
                    phase1_oof_path=replacement,
                    default_predictions_dir=default_dir,
                    zoo_run_dir=zoo_dir,
                    output_json=root / "result.json",
                    output_scores=root / "scores.pt",
                    expected_epochs=3,
                    alpha_step=0.5,
                    bootstrap_samples=10,
                )


if __name__ == "__main__":
    unittest.main()
