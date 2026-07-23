"""CPU synthetic smoke test for the GoalStep Phase-1 history trainer.

Runnable without pytest:

    PYTHONPATH=src python tests/smoke/test_goalstep_history_context_phase1.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from ego.step1_action_anticipation.goalstep.train_goalstep_history_context import (
    _load_phase0_diagnostic,
    run_training,
)
from ego.step1_action_anticipation.models.history_context_head import HistoryContextResidualHead


HEADS = ("verb", "noun", "action")
NUM_CLASSES = {"verb": 4, "noun": 6, "action": 9}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(root: Path, index_root: Path, split: str, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    store_count = 9 if split == "train" else 7
    target_count = 6 if split == "train" else 4
    ids = [f"{split}_segment_{index}" for index in range(store_count)]
    summaries = torch.randn(store_count, 5, 32, generator=generator).half()
    visual_logits = {
        head: torch.randn(store_count, classes, generator=generator).float()
        for head, classes in NUM_CLASSES.items()
    }
    recognition_logits = {
        head: torch.randn(store_count, classes, generator=generator).float()
        for head, classes in NUM_CLASSES.items()
    }
    split_dir = root / split
    split_dir.mkdir(parents=True)
    shard_path = split_dir / "shard_00000.pt"
    torch.save(
        {
            "schema_version": 1,
            "provenance_fingerprint": f"synthetic-{split}-shard",
            "sample_ids": ids,
            "summaries": summaries,
            "visual_logits": visual_logits,
            "recognition_logits": recognition_logits,
        },
        shard_path,
    )

    rows = []
    for target_index in range(target_count):
        current_position = target_index
        history_positions = list(range(max(0, current_position - 3), current_position))
        padding = 3 - len(history_positions)
        row = {
            "video_uid": f"{split}_video_{target_index // 2}",
            "sample_id": f"{split}_target_{target_index}",
            "current_cache_sample_id": ids[current_position],
            "verb_id": target_index % NUM_CLASSES["verb"],
            "noun_id": target_index % NUM_CLASSES["noun"],
            "action_id": target_index % NUM_CLASSES["action"],
            "scenario": "synthetic",
            "annotation_level": "step",
            "history_length": len(history_positions),
            "audit_current_observation_end_sec": float(10 + target_index),
            "audit_target_start_sec": float(11 + target_index),
        }
        for zero_slot in range(3):
            slot = zero_slot + 1
            history_offset = zero_slot - padding
            if history_offset < 0:
                row[f"history_{slot}_cache_sample_id"] = ""
                row[f"history_{slot}_mask"] = False
                row[f"history_{slot}_delta_t_sec"] = 0.0
                row[f"history_{slot}_level_id"] = -1
            else:
                history_position = history_positions[history_offset]
                row[f"history_{slot}_cache_sample_id"] = ids[history_position]
                row[f"history_{slot}_mask"] = True
                row[f"history_{slot}_delta_t_sec"] = float(
                    2 * (current_position - history_position)
                )
                row[f"history_{slot}_level_id"] = 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(index_root / f"{split}.csv", index=False)
    return {
        "rows": store_count,
        "shards": [
            {
                "path": str(shard_path.relative_to(root)),
                "rows": store_count,
                "provenance_fingerprint": f"synthetic-{split}-shard",
                "sha256": _sha256(shard_path),
            }
        ],
        "provenance_base_fingerprint": "synthetic-store-base",
    }


def test_goalstep_history_context_cpu_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="goalstep_history_smoke_") as temporary:
        root = Path(temporary)
        store_root = root / "store"
        index_root = root / "index"
        output_root = root / "run"
        gate_path = root / "p0b_results.json"
        store_root.mkdir()
        index_root.mkdir()
        train_manifest = _write_split(store_root, index_root, "train", seed=1)
        val_manifest = _write_split(store_root, index_root, "val", seed=2)
        (store_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "goalstep_history_context_derived_store",
                    "backbone_reextraction": False,
                    "source_cache_dir": "/synthetic/existing_cache",
                    "summary_shape": [5, 32],
                    "num_classes": NUM_CLASSES,
                    "visual_checkpoint": "/synthetic/visual.pt",
                    "visual_checkpoint_sha256": "synthetic-visual-hash",
                    "recognition_checkpoint": "/synthetic/recognition.pt",
                    "recognition_checkpoint_sha256": "synthetic-recognition-hash",
                    "provenance_base_fingerprint": "synthetic-store-base",
                    "splits": {"train": train_manifest, "val": val_manifest},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        gate_path.write_text(
            json.dumps(
                {
                    "phase": "P0-b",
                    "gate": {
                        "metric": "Action OOF instance Top-5 accuracy",
                        "threshold_percent": 27.7,
                        "observed_percent": 30.0,
                        "passed": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        config = {
            "experiment": {"seed": 7, "device": "cpu", "output_dir": str(output_root)},
            "phase0": {
                "gate_results_path": str(gate_path),
                "policy": "diagnostic_only",
                "historical_gate_threshold_action_top5": 27.7,
            },
            "champion": {"material_gain_pp_descriptive": 1.0},
            "dataset": {
                "history_index_dir": str(index_root),
                "derived_store_dir": str(store_root),
                "max_history": 3,
                "expected_summary_tokens": 5,
                "expected_embed_dim": 32,
                "num_workers": 0,
                "verify_shard_hashes": True,
            },
            "model": {
                "history": {
                    "segment_pooler_heads": 4,
                    "transformer_heads": 4,
                    "transformer_layers": 2,
                    "transformer_mlp_ratio": 2.0,
                    "transformer_dropout": 0.0,
                    "segment_dropout": 0.3,
                    "recency_scale_sec": 30.0,
                }
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "eval_batch_size": 2,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "warmup_epochs": 0,
                "final_lr": 0.0,
                "focal_gamma": 2.0,
                "focal_alpha": 0.25,
                "history_aux_weight": 0.25,
                "gradient_clip_norm": 1.0,
                "precision": "fp32",
            },
        }
        final = run_training(config)
        assert final["best_epoch"] in (0, 1)
        assert final["lower_bound_preserved"] is True
        assert (output_root / "checkpoints" / "epoch_00_visual_fallback.pt").is_file()
        assert (output_root / "checkpoints" / "epoch_01.pt").is_file()
        assert (output_root / "best_action_top5.pt").is_file()
        assert (output_root / "best_fullval_exploratory.pt").is_file()
        assert (output_root / "final_metrics.json").is_file()
        for epoch in (0, 1):
            artifact_path = output_root / "val_predictions" / f"epoch_{epoch:02d}.pt"
            assert artifact_path.is_file()
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
            assert artifact["kind"] == "goalstep_history_context_val_predictions"
            assert artifact["epoch"] == epoch
            assert len(artifact["sample_ids"]) == 4
            assert len(artifact["video_uids"]) == 4
            assert tuple(artifact["history_lengths"].shape) == (4,)
            assert set(artifact["labels"]) == set(HEADS)
            assert set(artifact["logits"]) == {
                "visual",
                "history",
                "current_only",
                "fused",
            }
        epoch_zero = final["epoch_0_visual_fallback"]
        for head in HEADS:
            assert epoch_zero["overall"]["visual"][head] == epoch_zero["overall"]["fused"][head]
        assert set(final["best_val"]["overall"]) == {
            "visual",
            "history",
            "current_only",
            "fused",
        }


def test_history_only_does_not_see_current_summary() -> None:
    torch.manual_seed(5)
    model = HistoryContextResidualHead(
        num_classes=NUM_CLASSES,
        embed_dim=32,
        max_history=3,
        segment_pooler_heads=4,
        transformer_heads=4,
        transformer_layers=2,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        segment_dropout=0.0,
        recency_scale_sec=30.0,
    ).eval()
    summaries = torch.randn(2, 4, 5, 32)
    changed_current = summaries.clone()
    changed_current[:, 0] = torch.randn_like(changed_current[:, 0]) * 100.0
    mask = torch.tensor([[False, True, True], [True, True, True]])
    delta = torch.tensor([[0.0, 4.0, 2.0], [6.0, 4.0, 2.0]])
    levels = torch.tensor([[-1, 0, 0], [1, 1, 1]])
    visual = {head: torch.randn(2, classes) for head, classes in NUM_CLASSES.items()}
    with torch.inference_mode():
        first = model(summaries, mask, delta, levels, visual)
        second = model(changed_current, mask, delta, levels, visual)
    for head in HEADS:
        assert torch.equal(first["history"][head], second["history"][head])
        assert torch.equal(first["current_only"][head], visual[head])
        assert torch.equal(second["current_only"][head], visual[head])
        assert torch.equal(first["fused"][head], visual[head])
        assert torch.equal(second["fused"][head], visual[head])

    # Once opened, the residual gate uses the distinct current+history pass.
    with torch.no_grad():
        for gate in model.field_gates.values():
            gate.fill_(1.0)
    with torch.inference_mode():
        first_open = model(summaries, mask, delta, levels, visual)
        second_open = model(changed_current, mask, delta, levels, visual)
    assert any(
        not torch.equal(first_open["fused"][head], second_open["fused"][head])
        for head in HEADS
    )
    for head in HEADS:
        assert not torch.equal(first_open["current_only"][head], second_open["current_only"][head])


def test_phase0_failure_is_diagnostic_only() -> None:
    with tempfile.TemporaryDirectory(prefix="goalstep_history_gate_") as temporary:
        gate_path = Path(temporary) / "p0b_results.json"
        gate_path.write_text(
            json.dumps(
                {
                    "phase": "P0-b",
                    "gate": {
                        "threshold_percent": 27.7,
                        "observed_percent": 27.6,
                        "passed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        diagnostic = _load_phase0_diagnostic(
            {
                "phase0": {
                    "gate_results_path": str(gate_path),
                    "policy": "diagnostic_only",
                    "historical_gate_threshold_action_top5": 27.7,
                }
            }
        )
        assert diagnostic["blocks_phase1"] is False
        assert diagnostic["historical_gate_passed"] is False
        assert diagnostic["observed_oof_action_top5"] == 27.6


if __name__ == "__main__":
    test_phase0_failure_is_diagnostic_only()
    test_history_only_does_not_see_current_summary()
    test_goalstep_history_context_cpu_smoke()
    print("GoalStep history-context Phase-1 CPU smoke: PASS")
