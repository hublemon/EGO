"""Final held-out evaluation for a trained Ego4D LTA Z=1 checkpoint.

``train_lta_z1.py`` only ever evaluates against the internal ``dev`` split
(used epoch-to-epoch for checkpoint selection). ``heldout`` is the slice of
val never seen during training/checkpoint-selection, reserved for one
unbiased final readout -- this script runs that readout, reusing every
metric/formatting helper from ``train_lta_z1`` unchanged (same class-mean
Recall@5, head/mid/tail bands computed from the *train* distribution, per-
scenario breakdown, likelihood/entropy dump) so the numbers are directly
comparable to what's already logged for dev.

Usage:
    python scripts/step1/ego4d_lta/evaluate_heldout.py \
        --config configs/step1/ego4d_lta/full.yaml \
        --checkpoint outputs/ego4d_lta/runs/full/best_action.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.io import ensure_dir, write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ego4d import index_scenario_lookup  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402
import train_lta_z1 as tz1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="e.g. outputs/ego4d_lta/runs/full/best_action.pt")
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_dir = expand_path(require(config, "dataset.index_dir"))
    mapping = tz1._load_registry(index_dir / "action_registry.json")
    num_classes = {"verb": mapping.num_verbs, "noun": mapping.num_nouns, "action": mapping.num_actions}
    step_log(1, "EvaluateHeldout", f"Taxonomy: verb={num_classes['verb']} noun={num_classes['noun']} action={num_classes['action']}")

    train_index = tz1._read_index(tz1._find_index_file(index_dir, "train"))
    eval_index = tz1._read_index(tz1._find_index_file(index_dir, args.split))
    eval_scenario_lookup = index_scenario_lookup(eval_index)
    bands = {h: tz1.head_mid_tail_bands(tz1._unified_class_frequency(train_index, mapping, h)) for h in tz1.HEADS}

    cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
    batch_size = require(config, "training.batch_size")
    num_workers = get(config, "dataset.num_workers", 0)
    eval_dataset, eval_loader, eval_scenarios = tz1._build_eval_loader(
        cache_dir / args.split, batch_size, eval_scenario_lookup, num_workers=num_workers
    )
    step_log(1, "EvaluateHeldout", f"{args.split} samples: {len(eval_dataset)}")

    embed_dim = eval_dataset[0]["video"].shape[-1]
    classifier_cfg = get(config, "model.classifier", {})
    head_model = AnticipationHead(
        num_verb_classes=num_classes["verb"],
        num_noun_classes=num_classes["noun"],
        num_action_classes=num_classes["action"],
        embed_dim=embed_dim,
        num_heads=classifier_cfg.get("num_heads", 16),
        depth=classifier_cfg.get("num_probe_blocks", 4),
        repository_dir=get(config, "model.repository_dir"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    head_model.load_state_dict(checkpoint["model_state"])
    step_log(1, "EvaluateHeldout", f"Loaded checkpoint {args.checkpoint} (epoch={checkpoint.get('epoch')})")

    result = tz1.evaluate(head_model, eval_loader, device, num_classes, bands, eval_scenarios)
    for h in tz1.HEADS:
        step_log(1, "EvaluateHeldout", f"{args.split} {h} class-mean Recall@5: {result['overall'][h]:.2f}")
        step_log(1, "EvaluateHeldout", f"{args.split} {h} band breakdown: {result['band'][h]}")

    run_dir = ensure_dir(expand_path(require(config, "experiment.output_dir")))
    write_json(
        run_dir / f"{args.split}_metrics.json",
        {"overall": result["overall"], "band": result["band"], "scenario": result["scenario"],
         "checkpoint": args.checkpoint, "checkpoint_epoch": checkpoint.get("epoch")},
    )
    tz1.save_likelihood_entropy(result["_preds"], eval_scenarios, run_dir / f"{args.split}_likelihood_entropy.jsonl")
    step_log(1, "EvaluateHeldout", f"Wrote {run_dir / f'{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
