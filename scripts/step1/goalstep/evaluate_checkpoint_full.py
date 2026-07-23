#!/usr/bin/env python3
"""Evaluate one GoalStep checkpoint on the complete validation cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "step1" / "ego4d_lta"))

import torch  # noqa: E402

import train_lta_z1 as tz1  # noqa: E402
from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.io import write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ego4d import index_scenario_lookup  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = int(checkpoint["epoch"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_dir = expand_path(require(config, "dataset.index_dir"))
    mapping = tz1._load_registry(index_dir / "action_registry.json")
    num_classes = {"verb": mapping.num_verbs, "noun": mapping.num_nouns, "action": mapping.num_actions}
    val_index = tz1._read_index(tz1._find_index_file(index_dir, "val"))
    val_scenario_lookup = index_scenario_lookup(val_index)

    cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
    batch_size = require(config, "training.batch_size")
    num_workers = get(config, "dataset.num_workers", 0)
    val_dataset, val_loader, val_scenarios = tz1._build_eval_loader(
        cache_dir / "val", batch_size, val_scenario_lookup, num_workers=num_workers
    )

    train_heads = get(config, "training.train_heads", list(tz1.HEADS))
    action_only = train_heads == ["action"]
    train_index = tz1._read_index(tz1._find_index_file(index_dir, "train"))
    bands = {
        head: tz1.head_mid_tail_bands(tz1._unified_class_frequency(train_index, mapping, head))
        for head in train_heads
    }
    embed_dim = val_dataset[0]["video"].shape[-1]
    classifier = get(config, "model.classifier", {})
    model = AnticipationHead(
        num_verb_classes=0 if action_only else num_classes["verb"],
        num_noun_classes=0 if action_only else num_classes["noun"],
        num_action_classes=num_classes["action"],
        embed_dim=embed_dim,
        num_heads=classifier.get("num_heads", 16),
        depth=classifier.get("num_probe_blocks", 4),
        repository_dir=get(config, "model.repository_dir"),
        use_temporal_metadata=bool(classifier.get("use_temporal_metadata", False)),
        temporal_duration_scale_sec=float(classifier.get("temporal_duration_scale_sec", 32.0)),
    ).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    step_log(1, "EvaluateGoalStepFull", f"Checkpoint epoch {epoch}; full val n={len(val_dataset)}")
    result = tz1.evaluate(model, val_loader, device, num_classes, bands, val_scenarios, heads=train_heads)
    metrics = {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metric_name": checkpoint.get("metric_name", "legacy_action_cmr5"),
        "checkpoint_metric": checkpoint.get("metric"),
        "val_size": len(val_dataset),
        "overall_cmr5": result["overall"],
        "accuracy_top1": result["accuracy_top1"],
        "accuracy_top5": result["accuracy_top5"],
        "accuracy_top10": result["accuracy_top10"],
        "accuracy_top15": result["accuracy_top15"],
        "band": result["band"],
        "scenario": result["scenario"],
    }

    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    output = Path(args.output).resolve() if args.output else run_dir / f"full_val_epoch_{epoch:02d}.json"
    predictions = (
        Path(args.predictions_output).resolve()
        if args.predictions_output
        else run_dir / f"likelihood_entropy_full_val_epoch_{epoch:02d}.jsonl"
    )
    write_json(output, metrics)
    tz1.save_likelihood_entropy(result["_preds"], val_scenarios, predictions, heads=train_heads)

    for head in train_heads:
        step_log(
            1,
            "EvaluateGoalStepFull",
            f"{head}: CMR@5={result['overall'][head]:.3f} "
            f"Top-1={result['accuracy_top1'][head]:.3f} "
            f"Top-5={result['accuracy_top5'][head]:.3f} "
            f"Top-10={result['accuracy_top10'][head]:.3f} "
            f"Top-15={result['accuracy_top15'][head]:.3f}",
        )
    step_log(1, "EvaluateGoalStepFull", f"Wrote {output} and {predictions}")


if __name__ == "__main__":
    main()
