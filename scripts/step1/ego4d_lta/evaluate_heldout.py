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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.io import ensure_dir, write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ego4d import index_scenario_lookup, load_lta_taxonomy  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402
import train_lta_z1 as tz1  # noqa: E402


def _action_text(taxonomy, mapping, unified_action_id: int | None) -> str | None:
    if unified_action_id is None:
        return None
    raw = mapping.inv_action_classes.get(int(unified_action_id))
    if raw is None:
        return None
    raw_verb, raw_noun = raw
    return f"{taxonomy.verb_text(raw_verb)} {taxonomy.noun_text(raw_noun)}"


def _text_for(head: str, taxonomy, mapping, unified_id: int | None) -> str | None:
    if unified_id is None:
        return None
    if head == "verb":
        raw = mapping.inv_verb_classes.get(int(unified_id))
        return taxonomy.verb_text(raw) if raw is not None else None
    if head == "noun":
        raw = mapping.inv_noun_classes.get(int(unified_id))
        return taxonomy.noun_text(raw) if raw is not None else None
    return _action_text(taxonomy, mapping, unified_id)


def build_predictions_dataframe(
    preds: dict, mapping, taxonomy, bands: dict, scenarios: list[str], k: int = 5
) -> pd.DataFrame:
    """One row per sample: GT/top-1/top-k predictions, likelihood, entropy,
    correctness, and head/mid/tail band -- for every head (verb/noun/action).
    Text labels are decoded via ``taxonomy`` where a mapping exists."""
    sample_ids = preds["sample_ids"]
    rows = [{"sample_id": sid, "scenario": scenarios[i] if i < len(scenarios) else "unknown"} for i, sid in enumerate(sample_ids)]

    for h in tz1.HEADS:
        logits = preds["logits"][h]
        labels = preds["labels"][h]
        probs = torch.softmax(logits, dim=-1)
        top1_prob, top1_id = probs.max(dim=-1)
        gt_likelihood = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        kk = min(k, logits.size(-1))
        topk_ids = logits.topk(kk, dim=-1).indices
        entropy = tz1.prediction_entropy(logits)
        hit_topk = (topk_ids == labels.unsqueeze(1)).any(dim=1)
        band_map = bands[h]

        for i in range(len(sample_ids)):
            gt_id = int(labels[i])
            pred_id = int(top1_id[i])
            rows[i][f"{h}_gt_id"] = gt_id
            rows[i][f"{h}_gt_text"] = _text_for(h, taxonomy, mapping, gt_id)
            rows[i][f"{h}_gt_likelihood"] = float(gt_likelihood[i])
            rows[i][f"{h}_pred_top1_id"] = pred_id
            rows[i][f"{h}_pred_top1_text"] = _text_for(h, taxonomy, mapping, pred_id)
            rows[i][f"{h}_pred_top1_likelihood"] = float(top1_prob[i])
            rows[i][f"{h}_topk_ids"] = ";".join(str(int(x)) for x in topk_ids[i])
            rows[i][f"{h}_correct_top1"] = pred_id == gt_id
            rows[i][f"{h}_correct_top{k}"] = bool(hit_topk[i])
            rows[i][f"{h}_entropy"] = float(entropy[i])
            rows[i][f"{h}_gt_band"] = band_map.get(gt_id, "unknown")

    return pd.DataFrame(rows)


def build_per_class_recall_dataframe(preds: dict, num_classes: dict, taxonomy, mapping, k: int = 5) -> pd.DataFrame:
    """Long-format per-class Recall@k for every head -- which specific classes
    are driving (or dragging down) the class-mean Recall@k reported in
    ``{split}_metrics.json``."""
    rows = []
    for h in tz1.HEADS:
        recall = tz1.per_class_recall(preds["logits"][h], preds["labels"][h], num_classes[h], k=k)
        support = torch.zeros(num_classes[h], dtype=torch.long)
        support.scatter_add_(0, preds["labels"][h], torch.ones_like(preds["labels"][h]))
        for cls in range(num_classes[h]):
            if support[cls] == 0:
                continue
            rows.append(
                {
                    "head": h,
                    "class_id": cls,
                    "class_text": _text_for(h, taxonomy, mapping, cls),
                    "support": int(support[cls]),
                    f"recall_at_{k}": float(recall[cls]) if not torch.isnan(recall[cls]) else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="e.g. outputs/ego4d_lta/runs/full/best_action.pt")
    parser.add_argument("--taxonomy", required=True, help="Path to fho_lta_taxonomy.json (for text-label decoding)")
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_dir = expand_path(require(config, "dataset.index_dir"))
    mapping = tz1._load_registry(index_dir / "action_registry.json")
    taxonomy = load_lta_taxonomy(expand_path(args.taxonomy))
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

    predictions_df = build_predictions_dataframe(result["_preds"], mapping, taxonomy, bands, eval_scenarios)
    predictions_path = run_dir / f"{args.split}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    step_log(1, "EvaluateHeldout", f"Wrote {predictions_path} ({len(predictions_df)} rows)")

    per_class_df = build_per_class_recall_dataframe(result["_preds"], num_classes, taxonomy, mapping)
    per_class_path = run_dir / f"{args.split}_per_class_recall.csv"
    per_class_df.to_csv(per_class_path, index=False)
    step_log(1, "EvaluateHeldout", f"Wrote {per_class_path} ({len(per_class_df)} rows)")

    step_log(1, "EvaluateHeldout", f"Wrote {run_dir / f'{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
