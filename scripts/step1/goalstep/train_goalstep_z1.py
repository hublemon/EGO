"""Task 6 -- train Step 1 verb/noun/action heads on GoalStep Z=1 samples.

Everything structural is imported from ``scripts/step1/ego4d_lta/train_lta_z1.py``
and left untouched: ``AnticipationHead`` (frozen V-JEPA2 encoder+predictor
features -> attentive probe, 3 query tokens -> 3 linear heads),
``sigmoid_focal_loss``, the warmup-cosine LR/WD schedules,
``ScenarioStratifiedSampler``, ``train_one_epoch``/``evaluate``, and
``class_mean_recall``. Only the head ``out_features`` change, and they change
by themselves -- they are read from the GoalStep ``action_registry.json``.

GoalStep-specific behaviour (the whole delta):
  * evaluates against ``val`` (goalstep_val.json, 134 videos) -- there is no
    dev/heldout re-split, and train is never used for evaluation;
  * per-epoch validation runs on a **fixed, seeded subsample** of val
    (``training.val_subset_size``, default 500) so every epoch is comparable;
    the same subset is reused for all epochs and its sample_ids are written to
    ``val_subset_sample_ids.json``;
  * **every** epoch is checkpointed (``epoch_01.pt`` ... ``epoch_NN.pt``) next
    to ``best.pt`` (best val action class-mean Recall@5) and ``latest.pt``;
  * after the last epoch, ``best.pt`` is re-evaluated once on the **full** val
    split, and both readouts are written to ``final_metrics.json`` for the
    subset-vs-full comparison in the report.

Prerequisites: ``build_goalstep_z1_index.py`` then ``extract_features.py``
(the FHO one, reused) for ``--split train`` and ``--split val``.

Usage:
    python scripts/step1/goalstep/train_goalstep_z1.py --config configs/step1/goalstep/z1.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego4d_lta"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_lta_z1 as tz1  # noqa: E402
from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.exceptions import EgoConfigError  # noqa: E402
from ego.common.io import ensure_dir, write_json, write_yaml  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.common.seed import set_seed  # noqa: E402
from ego.datasets.ego4d import index_scenario_lookup  # noqa: E402
from ego.step1_action_anticipation.data.collator import anticipation_collate  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402

PHASE = "TrainGoalStepZ1"
HEADS = tz1.HEADS

HISTORY_COLUMNS = [
    "epoch", "train_loss",
    "verb_cmr@5", "verb_top1", "verb_top5",
    "noun_cmr@5", "noun_top1", "noun_top5",
    "action_cmr@5", "action_top1", "action_top5",
    "seconds",
]


def _subset_loader(cache_dir: Path, sample_ids: list[str], batch_size: int, num_workers: int):
    """Sequential loader over an explicit sample_id list.

    Sequential (``shuffle=False``, no sampler) for the same reason
    ``train_lta_z1._build_eval_loader`` is: ``compute_predictions`` pairs its
    collected logits back to the caller's ``scenarios`` list positionally.
    """
    dataset = FeatureCacheDataset(sample_ids, cache_dir)
    if len(dataset) == 0:
        raise EgoConfigError(f"No cached features found under {cache_dir}. Run extract_features.py --split val first.")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=anticipation_collate,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return dataset, loader


def _log_eval(prefix: str, result: dict) -> None:
    for h in HEADS:
        step_log(1, PHASE, f"{prefix} {h}: class-mean Recall@5={result['overall'][h]:.2f}  "
                           f"top1={result['accuracy_top1'][h]:.2f}  top5={result['accuracy_top5'][h]:.2f}")
        step_log(1, PHASE, f"{prefix} {h} band breakdown: {result['band'][h]}")


def _metrics_dict(result: dict, train_loss: float | None = None, epoch: int | None = None) -> dict:
    out = {
        "overall_cmr5": result["overall"],
        "accuracy_top1": result["accuracy_top1"],
        "accuracy_top5": result["accuracy_top5"],
        "band": result["band"],
        "scenario": result["scenario"],
    }
    if epoch is not None:
        out["epoch"] = epoch
    if train_loss is not None:
        out["train_loss"] = train_loss
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-final-full-val", action="store_true",
                        help="Skip the one-off full-val readout of best.pt (smoke tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = get(config, "experiment.seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_dir = expand_path(require(config, "dataset.index_dir"))
    mapping = tz1._load_registry(index_dir / "action_registry.json")
    num_classes = {"verb": mapping.num_verbs, "noun": mapping.num_nouns, "action": mapping.num_actions}
    step_log(1, PHASE, f"Taxonomy (head out_features): verb={num_classes['verb']} "
                       f"noun={num_classes['noun']} action={num_classes['action']}")

    train_index = tz1._read_index(tz1._find_index_file(index_dir, "train"))
    val_index = tz1._read_index(tz1._find_index_file(index_dir, "val"))
    train_scenario_lookup = index_scenario_lookup(train_index)
    val_scenario_lookup = index_scenario_lookup(val_index)

    cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
    batch_size = require(config, "training.batch_size")
    sampler_name = get(config, "training.sampler", "random")
    num_workers = get(config, "dataset.num_workers", 0)

    train_dataset, train_loader, _ = tz1._build_train_loader(
        cache_dir / "train", batch_size, sampler_name, train_scenario_lookup, seed, num_workers=num_workers
    )
    full_val_dataset, full_val_loader, full_val_scenarios = tz1._build_eval_loader(
        cache_dir / "val", batch_size, val_scenario_lookup, num_workers=num_workers
    )
    step_log(1, PHASE, f"Train samples: {len(train_dataset)}  Val samples (full): {len(full_val_dataset)}")
    step_log(1, PHASE, f"Sampler: {sampler_name}")

    # Fixed, seeded val subsample -- identical across every epoch so the
    # per-epoch curve is comparable; best.pt is re-scored on full val at the end.
    subset_size = get(config, "training.val_subset_size", 500)
    subset_seed = get(config, "training.val_subset_seed", seed)
    all_val_ids = list(full_val_dataset.sample_ids)
    if subset_size and len(all_val_ids) > subset_size:
        subset_ids = sorted(random.Random(subset_seed).sample(all_val_ids, subset_size))
        step_log(1, PHASE, f"Per-epoch validation on a fixed {len(subset_ids)}-sample val subset "
                           f"(seed={subset_seed}); full val ({len(all_val_ids)}) reserved for the final readout")
    else:
        subset_ids = all_val_ids
        step_log(1, PHASE, f"Per-epoch validation on all {len(subset_ids)} val samples "
                           f"(<= val_subset_size={subset_size})")
    subset_dataset, subset_loader = _subset_loader(cache_dir / "val", subset_ids, batch_size, num_workers)
    subset_scenarios = [val_scenario_lookup.get(sid, "unknown") for sid in subset_dataset.sample_ids]

    bands = {h: tz1.head_mid_tail_bands(tz1._unified_class_frequency(train_index, mapping, h)) for h in HEADS}

    embed_dim = train_dataset[0]["video"].shape[-1]
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

    num_epochs = require(config, "training.epochs")
    iterations_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * iterations_per_epoch
    lr = require(config, "training.learning_rate")
    wd = get(config, "training.weight_decay", 0.0001)
    optimizer = torch.optim.AdamW(head_model.parameters(), lr=lr, weight_decay=wd)
    lr_sched = tz1._WarmupCosineLR(
        optimizer, ref_lr=lr, start_lr=get(config, "training.start_lr", 0.0),
        final_lr=get(config, "training.final_lr", 0.0),
        warmup_steps=int(get(config, "training.warmup_epochs", 0) * iterations_per_epoch),
        total_steps=total_steps,
    )
    wd_sched = tz1._CosineWD(
        optimizer, ref_wd=wd, final_wd=get(config, "training.final_weight_decay", wd), total_steps=total_steps
    )
    focal_gamma = get(config, "training.focal_gamma", 2.0)
    focal_alpha = get(config, "training.focal_alpha", 0.25)

    run_dir = ensure_dir(expand_path(require(config, "experiment.output_dir")))
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_json(run_dir / "run_metadata.json", {
        "dataset": "ego4d_goalstep",
        "focal_gamma": focal_gamma, "focal_alpha": focal_alpha, "sampler": sampler_name,
        "seed": seed, "epochs": num_epochs, "batch_size": batch_size, "learning_rate": lr,
        "tau_a": get(config, "dataset.tau_a", 1.0), "l_obs": get(config, "dataset.l_obs", 3.5),
        "taxonomy": num_classes, "index_dir": str(index_dir),
        "train_samples": len(train_dataset), "val_samples_full": len(full_val_dataset),
        "val_subset_size": len(subset_dataset), "val_subset_seed": subset_seed,
    })
    write_json(run_dir / "val_subset_sample_ids.json",
               {"seed": subset_seed, "size": len(subset_dataset), "sample_ids": subset_dataset.sample_ids})

    history_path = run_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HISTORY_COLUMNS)

    best_metric, best_epoch = float("-inf"), None
    per_epoch: list[dict] = []

    for epoch in range(1, num_epochs + 1):
        if isinstance(train_loader.sampler, tz1.ScenarioStratifiedSampler):
            train_loader.sampler.set_epoch(epoch)
        step_log(1, PHASE, f"Epoch {epoch}/{num_epochs}")
        t0 = time.time()
        train_loss = tz1.train_one_epoch(
            head_model, train_loader, optimizer, lr_sched, wd_sched, device, focal_gamma, focal_alpha
        )
        step_log(1, PHASE, f"Train loss: {train_loss:.4f}")

        result = tz1.evaluate(head_model, subset_loader, device, num_classes, bands, subset_scenarios)
        _log_eval(f"Val[subset n={len(subset_dataset)}] epoch {epoch}", result)
        elapsed = time.time() - t0

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.4f}",
                *(x for h in HEADS for x in (
                    f"{result['overall'][h]:.4f}",
                    f"{result['accuracy_top1'][h]:.4f}",
                    f"{result['accuracy_top5'][h]:.4f}",
                )),
                f"{elapsed:.1f}",
            ])
        per_epoch.append(_metrics_dict(result, train_loss=train_loss, epoch=epoch))
        write_json(run_dir / "metrics_per_epoch.json", per_epoch)

        state = {"epoch": epoch, "model_state": head_model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "metric": result["overall"]["action"], "num_classes": num_classes}
        torch.save(state, ckpt_dir / f"epoch_{epoch:02d}.pt")
        torch.save(state, run_dir / "latest.pt")
        if result["overall"]["action"] > best_metric:
            best_metric, best_epoch = result["overall"]["action"], epoch
            torch.save(state, run_dir / "best.pt")
            step_log(1, PHASE, f"Best updated -> epoch {epoch} (val-subset action cmr@5={best_metric:.2f})")

        tz1.save_likelihood_entropy(result["_preds"], subset_scenarios,
                                    run_dir / f"likelihood_entropy_epoch_{epoch:02d}.jsonl")

    step_log(1, PHASE, f"Training done. Best epoch={best_epoch} (val-subset action cmr@5={best_metric:.2f})")

    final = {
        "best_epoch": best_epoch,
        "val_subset": {"size": len(subset_dataset), "seed": subset_seed,
                       "metrics": per_epoch[best_epoch - 1] if best_epoch else None},
        "per_epoch": per_epoch,
    }
    if not args.skip_final_full_val and best_epoch is not None:
        step_log(1, PHASE, f"Final readout: best.pt (epoch {best_epoch}) on FULL val ({len(full_val_dataset)} samples)")
        head_model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model_state"])
        full_result = tz1.evaluate(head_model, full_val_loader, device, num_classes, bands, full_val_scenarios)
        _log_eval(f"Val[FULL n={len(full_val_dataset)}] best epoch {best_epoch}", full_result)
        step_log(1, PHASE, f"Val[FULL] action scenario breakdown: {full_result['scenario']['action']}")
        final["val_full"] = {"size": len(full_val_dataset), "metrics": _metrics_dict(full_result, epoch=best_epoch)}
        tz1.save_likelihood_entropy(full_result["_preds"], full_val_scenarios,
                                    run_dir / "likelihood_entropy_full_val_best.jsonl")

    write_json(run_dir / "final_metrics.json", final)
    step_log(1, PHASE, f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
