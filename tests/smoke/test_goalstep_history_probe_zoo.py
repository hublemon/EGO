"""CPU synthetic smoke for the Phase-2 shared-store probe zoo.

This exercises the closed 12-arm grid, default-arm reuse, eleven-arm shared
training/evaluation, Phase-1-compatible prediction artifacts, model-only
checkpoints, and atomic resume state without requiring pytest or CUDA.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml

from ego.step1_action_anticipation.goalstep.train_goalstep_history_probe_zoo import (
    build_registered_grid,
    run_zoo,
)
from ego.step1_action_anticipation.models.history_context_head import (
    HistoryContextResidualHead,
)


HEADS = ("verb", "noun", "action")
MODES = ("visual", "history", "current_only", "fused")
NUM_CLASSES = {"verb": 4, "noun": 5, "action": 7}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_store_and_index(root: Path, index_root: Path, split: str, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    store_count = 7 if split == "train" else 5
    target_count = 4 if split == "train" else 3
    sample_ids = [f"{split}_segment_{index}" for index in range(store_count)]
    summaries = torch.randn(store_count, 3, 8, generator=generator).half()
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
            "sample_ids": sample_ids,
            "summaries": summaries,
            "visual_logits": visual_logits,
            "recognition_logits": recognition_logits,
        },
        shard_path,
    )

    rows = []
    for target_index in range(target_count):
        history_positions = list(range(max(0, target_index - 2), target_index))
        padding = 2 - len(history_positions)
        row = {
            "video_uid": f"{split}_video_{target_index // 2}",
            "sample_id": f"{split}_target_{target_index}",
            "current_cache_sample_id": sample_ids[target_index],
            "verb_id": target_index % NUM_CLASSES["verb"],
            "noun_id": target_index % NUM_CLASSES["noun"],
            "action_id": target_index % NUM_CLASSES["action"],
            "annotation_level": "step",
            "history_length": len(history_positions),
            "audit_current_observation_end_sec": float(10 + target_index),
            "audit_target_start_sec": float(11 + target_index),
        }
        for zero_slot in range(2):
            slot = zero_slot + 1
            offset = zero_slot - padding
            if offset < 0:
                row[f"history_{slot}_cache_sample_id"] = ""
                row[f"history_{slot}_mask"] = False
                row[f"history_{slot}_delta_t_sec"] = 0.0
                row[f"history_{slot}_level_id"] = -1
            else:
                position = history_positions[offset]
                row[f"history_{slot}_cache_sample_id"] = sample_ids[position]
                row[f"history_{slot}_mask"] = True
                row[f"history_{slot}_delta_t_sec"] = float(2 * (target_index - position))
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


def _write_default_phase1(
    run_dir: Path,
    val_frame: pd.DataFrame,
    initial_model_state: dict[str, torch.Tensor],
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "val_predictions").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "final_metrics.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (run_dir / "run_metadata.json").write_text('{"synthetic":true}\n', encoding="utf-8")
    torch.save(
        {"epoch": 0, "model_state": initial_model_state},
        run_dir / "checkpoints" / "epoch_00_visual_fallback.pt",
    )
    sample_ids = val_frame["sample_id"].astype(str).tolist()
    video_uids = val_frame["video_uid"].astype(str).tolist()
    labels = {
        head: torch.tensor(val_frame[f"{head}_id"].to_numpy(), dtype=torch.long)
        for head in HEADS
    }
    (run_dir / "history_context_vs_p0a_results.json").write_text(
        json.dumps(
            {
                "phase": "Phase-1 crossfit selection and P0-a-aware final ensemble",
                "sample_count": len(val_frame),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "format_version": 1,
            "kind": "goalstep_history_context_crossfit_oof_scores",
            "sample_ids": sample_ids,
            "video_uids": video_uids,
            "labels": labels,
        },
        run_dir / "history_context_vs_p0a_oof_scores.pt",
    )
    generator = torch.Generator().manual_seed(99)
    for epoch in range(11):
        logits = {
            mode: {
                head: torch.randn(len(val_frame), classes, generator=generator).float()
                for head, classes in NUM_CLASSES.items()
            }
            for mode in MODES
        }
        torch.save(
            {
                "format_version": 1,
                "kind": "goalstep_history_context_val_predictions",
                "epoch": epoch,
                "contract": "A2.end-1s -> strict same-level A3",
                "sample_ids": sample_ids,
                "video_uids": video_uids,
                "history_lengths": torch.zeros(len(val_frame), dtype=torch.long),
                "labels": labels,
                "logits": logits,
                "num_classes": NUM_CLASSES,
                "gate_values": {head: {"raw": 0.0, "tanh": 0.0} for head in HEADS},
            },
            run_dir / "val_predictions" / f"epoch_{epoch:02d}.pt",
        )


def test_goalstep_history_probe_zoo_cpu_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="goalstep_history_zoo_smoke_") as temporary:
        root = Path(temporary)
        store_root = root / "store"
        index_root = root / "index"
        default_run = root / "phase1"
        output_dir = root / "zoo"
        base_config_path = root / "phase1.yaml"
        zoo_config_path = root / "zoo.yaml"
        store_root.mkdir()
        index_root.mkdir()
        train_manifest = _write_store_and_index(store_root, index_root, "train", seed=1)
        val_manifest = _write_store_and_index(store_root, index_root, "val", seed=2)
        (store_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "goalstep_history_context_derived_store",
                    "backbone_reextraction": False,
                    "source_cache_dir": "/synthetic/cache",
                    "summary_shape": [3, 8],
                    "num_classes": NUM_CLASSES,
                    "visual_checkpoint": "/synthetic/visual.pt",
                    "visual_checkpoint_sha256": "synthetic-visual",
                    "recognition_checkpoint": "/synthetic/recognition.pt",
                    "recognition_checkpoint_sha256": "synthetic-recognition",
                    "provenance_base_fingerprint": "synthetic-store-base",
                    "splits": {"train": train_manifest, "val": val_manifest},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        base_config = {
            "experiment": {"seed": 42, "device": "cpu"},
            "dataset": {
                "history_index_dir": str(index_root),
                "derived_store_dir": str(store_root),
                "max_history": 2,
                "num_workers": 0,
                "verify_shard_hashes": True,
            },
            "model": {
                "history": {
                    "segment_pooler_heads": 2,
                    "transformer_heads": 2,
                    "transformer_layers": 1,
                    "transformer_mlp_ratio": 1.0,
                    "transformer_dropout": 0.0,
                    "segment_dropout": 0.0,
                    "recency_scale_sec": 30.0,
                }
            },
            "training": {
                "epochs": 10,
                "batch_size": 2,
                "eval_batch_size": 2,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "warmup_epochs": 1,
                "final_lr": 0.0,
                "focal_gamma": 2.0,
                "focal_alpha": 0.25,
                "history_aux_weight": 0.25,
                "gradient_clip_norm": 1.0,
                "precision": "fp32",
            },
        }
        base_config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
        # Phase-1 and every Phase-2 arm must begin from this exact seed-42
        # state; the production trainer audits the actual Phase-1 epoch-0
        # checkpoint the same way.
        torch.manual_seed(42)
        default_model = HistoryContextResidualHead(
            num_classes=NUM_CLASSES,
            embed_dim=8,
            max_history=2,
            segment_pooler_heads=2,
            transformer_heads=2,
            transformer_layers=1,
            transformer_mlp_ratio=1.0,
            transformer_dropout=0.0,
            segment_dropout=0.0,
            recency_scale_sec=30.0,
        )
        val_frame = pd.read_csv(index_root / "val.csv")
        _write_default_phase1(default_run, val_frame, default_model.state_dict())
        zoo_config = {
            "experiment": {"name": "synthetic_zoo", "device": "cpu", "output_dir": str(output_dir)},
            "source": {
                "phase1_config": str(base_config_path),
                "default_phase1_run_dir": str(default_run),
            },
            "dataset": {"expected_train_rows": 4, "expected_val_rows": 3},
            "zoo": {
                "learning_rates": [0.0001, 0.0003, 0.001],
                "weight_decays": [0.00001, 0.0001, 0.001, 0.01],
                "default_phase1_learning_rate": 0.0003,
                "default_phase1_weight_decay": 0.0001,
                "epochs": 10,
                "seed": 42,
            },
        }
        zoo_config_path.write_text(yaml.safe_dump(zoo_config, sort_keys=False), encoding="utf-8")
        specs, default = build_registered_grid(zoo_config)
        assert len(specs) == 11
        assert (default.learning_rate, default.weight_decay) == (0.0003, 0.0001)

        first = run_zoo(
            zoo_config,
            config_path=zoo_config_path,
            allow_resume=True,
            _test_stop_after_epoch=1,
        )
        assert first["status"] == "stopped_for_smoke"
        assert first["completed_epoch"] == 1
        latest = torch.load(output_dir / "latest_resume.pt", map_location="cpu", weights_only=True)
        assert latest["epoch"] == 1
        assert len(latest["arms"]) == 11
        initialization = json.loads(
            (output_dir / "initialization_audit.json").read_text(encoding="utf-8")
        )
        assert initialization["identical"] is True
        assert initialization["matches_default_phase1_epoch0"] is True
        assert initialization["arm_count"] == 11
        assert initialization["confounded_variables"] == []
        for spec in specs:
            arm_dir = output_dir / "arms" / spec.arm_id
            checkpoint = torch.load(
                arm_dir / "checkpoints" / "epoch_01.pt",
                map_location="cpu",
                weights_only=True,
            )
            prediction = torch.load(
                arm_dir / "val_predictions" / "epoch_01.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert checkpoint["optimizer_state_included"] is False
            assert checkpoint["arm_id"] == spec.arm_id
            assert prediction["arm_id"] == spec.arm_id
            assert set(prediction["logits"]) == set(MODES)

        resumed = run_zoo(
            zoo_config,
            config_path=zoo_config_path,
            allow_resume=True,
            _test_stop_after_epoch=2,
        )
        assert resumed["completed_epoch"] == 2
        rows = json.loads((output_dir / "metrics_per_arm_epoch.json").read_text(encoding="utf-8"))
        assert len(rows) == 22
        assert {int(row["epoch"]) for row in rows} == {1, 2}


if __name__ == "__main__":
    test_goalstep_history_probe_zoo_cpu_smoke()
    print("GoalStep history probe-zoo CPU smoke: PASS")
