"""Task 6 -- train a Step 1 action head on GoalStep Z=1 samples.

The frozen V-JEPA2 features feed an ``AnticipationHead`` configured for either
an action-only classifier or verb/noun/action classifiers, as selected by
``training.train_heads``.

GoalStep-specific behaviour (the whole delta):
  * evaluates against ``val`` (goalstep_val.json, 134 videos) -- there is no
    dev/heldout re-split, and train is never used for evaluation;
  * per-epoch validation runs on a **fixed, seeded subsample** of val
    (``training.val_subset_size``, default 500) so every epoch is comparable;
    the same subset is reused for all epochs and its sample_ids are written to
    ``val_subset_sample_ids.json``;
  * **every** epoch is checkpointed (``epoch_01.pt`` ... ``epoch_NN.pt``) next
    to ``best.pt`` (best val Action Top-5 accuracy) and ``latest.pt``;
  * after the last epoch, ``best.pt`` is re-evaluated once on the **full** val
    split, and both readouts are written to ``final_metrics.json`` for the
    subset-vs-full comparison in the report.

Prerequisites: ``build_goalstep_z1_index.py`` then ``extract_features.py``
(the FHO one, reused) for ``--split train`` and ``--split val``.

Usage:
    python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py --config configs/step1/goalstep/z1.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

# the FHO trainer this one reuses still lives under scripts/ (parents[4] == repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "step1" / "ego4d_lta"))
# parents[3] is <repo>/src (this file lives at src/ego/step1_action_anticipation/goalstep/)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_lta_z1 as tz1  # noqa: E402
from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.exceptions import EgoConfigError  # noqa: E402
from ego.common.io import ensure_dir, write_json, write_yaml  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.common.seed import set_seed  # noqa: E402
from ego.datasets.ego4d import index_scenario_lookup, z1_sample_id  # noqa: E402
from ego.step1_action_anticipation.data.collator import anticipation_collate  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402

PHASE = "TrainGoalStepZ1"
HEADS = tz1.HEADS
BEST_METRIC_NAME = "action_top5"

def _history_columns(heads: list[str]) -> list[str]:
    return ["epoch", "train_loss", *(column for h in heads for column in (
        f"{h}_cmr@5", f"{h}_top1", f"{h}_top5", f"{h}_top10", f"{h}_top15",
    )), "seconds"]


def _subset_loader(
    cache_dir: Path,
    sample_ids: list[str],
    batch_size: int,
    num_workers: int,
    label_overrides: dict[str, dict[str, int]] | None = None,
):
    """Sequential loader over an explicit sample_id list.

    Sequential (``shuffle=False``, no sampler) for the same reason
    ``train_lta_z1._build_eval_loader`` is: ``compute_predictions`` pairs its
    collected logits back to the caller's ``scenarios`` list positionally.
    """
    dataset = FeatureCacheDataset(sample_ids, cache_dir, label_overrides=label_overrides)
    if len(dataset) == 0:
        raise EgoConfigError(f"No cached features found under {cache_dir}. Run extract_features.py --split val first.")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=anticipation_collate,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return dataset, loader


def _sample_ids_from_index(index_df):
    """Return cache identities, preserving source-row ids for relabel protocols."""
    if "cache_sample_id" in index_df.columns:
        ids = index_df["cache_sample_id"].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise EgoConfigError("dataset index contains duplicate cache_sample_id values")
        return ids
    frame = index_df.reset_index(drop=True)
    return [z1_sample_id(str(row["clip_uid"]), i) for i, row in frame.iterrows()]


def _scenario_lookup(index_df) -> dict[str, str]:
    if "cache_sample_id" not in index_df.columns:
        return index_scenario_lookup(index_df)
    return {
        str(row["cache_sample_id"]): str(row["scenario"])
        for _, row in index_df.iterrows()
    }


def _label_overrides(index_df, mapping) -> dict[str, dict[str, int]]:
    if "cache_sample_id" not in index_df.columns:
        raise EgoConfigError("dataset.labels_from_index=true requires cache_sample_id in the index")
    overrides = {}
    for _, row in index_df.iterrows():
        verb_raw, noun_raw = int(row["verb_label"]), int(row["noun_label"])
        overrides[str(row["cache_sample_id"])] = {
            "verb_id": mapping.encode_verb(verb_raw),
            "noun_id": mapping.encode_noun(noun_raw),
            "action_id": mapping.encode_action(verb_raw, noun_raw),
        }
    return overrides


def _indexed_loader(
    cache_dir: Path,
    index_df,
    mapping,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    sampler_name: str | None = None,
    seed: int = 42,
):
    sample_ids = _sample_ids_from_index(index_df)
    overrides = _label_overrides(index_df, mapping)
    dataset = FeatureCacheDataset(sample_ids, cache_dir, label_overrides=overrides)
    if len(dataset) != len(sample_ids):
        raise EgoConfigError(
            f"Relabel index/cache mismatch under {cache_dir}: "
            f"index={len(sample_ids)} cached={len(dataset)} missing={len(sample_ids) - len(dataset)}"
        )
    scenarios_by_id = _scenario_lookup(index_df)
    scenarios = [scenarios_by_id[sid] for sid in dataset.sample_ids]
    sampler = None
    if sampler_name == "scenario_stratified":
        sampler = tz1.ScenarioStratifiedSampler(scenarios, seed=seed)
        shuffle = False
    elif sampler_name not in (None, "random"):
        raise EgoConfigError(f"Unknown sampler {sampler_name!r}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        collate_fn=anticipation_collate, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader, scenarios, overrides


def _log_eval(prefix: str, result: dict, heads: list[str]) -> None:
    for h in heads:
        step_log(1, PHASE, f"{prefix} {h}: class-mean Recall@5={result['overall'][h]:.2f}  "
                           f"top1={result['accuracy_top1'][h]:.2f}  "
                           f"top5={result['accuracy_top5'][h]:.2f}  "
                           f"top10={result['accuracy_top10'][h]:.2f}  "
                           f"top15={result['accuracy_top15'][h]:.2f}")
        step_log(1, PHASE, f"{prefix} {h} band breakdown: {result['band'][h]}")


def _metrics_dict(result: dict, train_loss: float | None = None, epoch: int | None = None) -> dict:
    out = {
        "overall_cmr5": result["overall"],
        "accuracy_top1": result["accuracy_top1"],
        "accuracy_top5": result["accuracy_top5"],
        "accuracy_top10": result["accuracy_top10"],
        "accuracy_top15": result["accuracy_top15"],
        "band": result["band"],
        "scenario": result["scenario"],
    }
    if "stratified" in result:
        out["stratified"] = result["stratified"]
    if epoch is not None:
        out["epoch"] = epoch
    if train_loss is not None:
        out["train_loss"] = train_loss
    return out


def _adaptive_group_lookup(index_df) -> dict[str, dict[str, str]] | None:
    """Build audit-only adaptive cohort groups keyed by cache sample id.

    These fields are joined to predictions only after inference.  In
    particular, the future inter-action gap never enters the feature cache or
    model forward path.
    """
    required = {
        "annotation_level",
        "inter_action_gap_sec",
        "observed_action_duration_sec",
        "observed_action_label",
        "action_label",
    }
    if not required.issubset(index_df.columns):
        return None

    sample_ids = _sample_ids_from_index(index_df)
    lookup: dict[str, dict[str, str]] = {}
    for sample_id, (_, row) in zip(sample_ids, index_df.reset_index(drop=True).iterrows()):
        gap = float(row["inter_action_gap_sec"])
        if gap <= 0.5 + 1e-9:
            gap_bin = "0-0.5s"
        elif gap <= 1.0 + 1e-9:
            gap_bin = "0.5-1s"
        else:
            gap_bin = "1-2s"

        duration = float(row["observed_action_duration_sec"])
        if duration < 8.0:
            duration_bin = "1-8s"
        elif duration < 16.0:
            duration_bin = "8-16s"
        elif duration <= 32.0:
            duration_bin = "16-32s"
        else:
            duration_bin = ">32s"

        lookup[sample_id] = {
            "gap": gap_bin,
            "level": str(row["annotation_level"]),
            "transition": (
                "same_class"
                if int(row["observed_action_label"]) == int(row["action_label"])
                else "different_class"
            ),
            "observed_action_duration": duration_bin,
        }
    return lookup


def _attach_adaptive_stratified_metrics(
    result: dict,
    index_df,
    num_classes: dict[str, int],
    heads: list[str],
) -> None:
    lookup = _adaptive_group_lookup(index_df)
    if lookup is None:
        return

    sample_ids = result["_preds"]["sample_ids"]
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise EgoConfigError(
            f"Adaptive evaluation metadata is missing {len(missing)} prediction ids; "
            f"first={missing[0]}"
        )

    stratified: dict[str, dict[str, dict]] = {}
    for dimension in ("gap", "level", "transition", "observed_action_duration"):
        groups: dict[str, list[int]] = {}
        for position, sample_id in enumerate(sample_ids):
            groups.setdefault(lookup[sample_id][dimension], []).append(position)
        stratified[dimension] = {}
        for group, positions in groups.items():
            index_tensor = torch.tensor(positions, dtype=torch.long)
            group_metrics = {"size": len(positions), "heads": {}}
            for head in heads:
                logits = result["_preds"]["logits"][head][index_tensor]
                labels = result["_preds"]["labels"][head][index_tensor]
                group_metrics["heads"][head] = {
                    "cmr5": tz1.class_mean_recall(
                        logits, labels, num_classes[head], k=5
                    ),
                    "top1": tz1.top_k_recall(logits, labels, k=1),
                    "top5": tz1.top_k_recall(logits, labels, k=5),
                    "top10": tz1.top_k_recall(logits, labels, k=10),
                    "top15": tz1.top_k_recall(logits, labels, k=15),
                }
            stratified[dimension][group] = group_metrics
    result["stratified"] = stratified


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
    train_scenario_lookup = _scenario_lookup(train_index)
    val_scenario_lookup = _scenario_lookup(val_index)

    cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
    batch_size = require(config, "training.batch_size")
    sampler_name = get(config, "training.sampler", "random")
    num_workers = get(config, "dataset.num_workers", 0)

    # Training-loop precision only; eval/likelihood/entropy stay fp32 (see
    # tz1.train_one_epoch docstring for the measured fp32-vs-bf16 deltas).
    precision = str(get(config, "training.precision", "fp32")).lower()
    amp_dtypes = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in amp_dtypes:
        raise EgoConfigError(f"training.precision must be one of {list(amp_dtypes)}; got {precision!r}")
    amp_dtype = amp_dtypes[precision]
    if precision == "fp16":
        raise EgoConfigError("training.precision='fp16' needs a GradScaler; use 'bf16' on H100/H200.")
    step_log(1, PHASE, f"Train-loop precision: {precision} (eval + exported probabilities: fp32)")

    # Video-axis thinning (both default to None = use every cached sample).
    max_per_video = get(config, "training.max_samples_per_video", None)
    max_videos = get(config, "training.max_train_videos", None)
    max_samples = get(config, "training.max_train_samples", None)
    if max_per_video is not None or max_videos is not None or max_samples is not None:
        step_log(1, PHASE, f"Train pool thinned: max_samples_per_video={max_per_video} "
                           f"max_train_videos={max_videos} max_train_samples={max_samples}")

    labels_from_index = bool(get(config, "dataset.labels_from_index", False))
    train_label_overrides = val_label_overrides = None
    if labels_from_index:
        if max_per_video is not None or max_videos is not None or max_samples is not None:
            raise EgoConfigError("Index-label overlay currently requires the complete relabel index")
        train_dataset, train_loader, _, train_label_overrides = _indexed_loader(
            cache_dir / "train", train_index, mapping, batch_size, num_workers,
            shuffle=True, sampler_name=sampler_name, seed=seed,
        )
        full_val_dataset, full_val_loader, full_val_scenarios, val_label_overrides = _indexed_loader(
            cache_dir / "val", val_index, mapping, batch_size, num_workers,
            shuffle=False,
        )
        step_log(1, PHASE, "Labels: index overlay enabled; cached visual features reused unchanged")
    else:
        train_dataset, train_loader, _ = tz1._build_train_loader(
            cache_dir / "train", batch_size, sampler_name, train_scenario_lookup, seed, num_workers=num_workers,
            max_per_video=max_per_video, max_videos=max_videos, max_samples=max_samples
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
    subset_dataset, subset_loader = _subset_loader(
        cache_dir / "val", subset_ids, batch_size, num_workers, label_overrides=val_label_overrides
    )
    subset_scenarios = [val_scenario_lookup.get(sid, "unknown") for sid in subset_dataset.sample_ids]

    num_epochs = require(config, "training.epochs")
    iterations_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * iterations_per_epoch
    lr = require(config, "training.learning_rate")
    wd = get(config, "training.weight_decay", 0.0001)
    focal_gamma = get(config, "training.focal_gamma", 2.0)
    focal_alpha = get(config, "training.focal_alpha", 0.25)
    train_heads = get(config, "training.train_heads", list(HEADS))
    supported_head_modes = [list(HEADS), ["action"]]
    if train_heads not in supported_head_modes:
        raise EgoConfigError(
            f"training.train_heads must be either {list(HEADS)} or ['action']; got {train_heads}"
        )
    action_only = train_heads == ["action"]
    step_log(1, PHASE, f"Supervised/emitted/evaluated heads: {train_heads}")

    bands = {
        h: tz1.head_mid_tail_bands(tz1._unified_class_frequency(train_index, mapping, h))
        for h in train_heads
    }
    embed_dim = train_dataset[0]["video"].shape[-1]
    classifier_cfg = get(config, "model.classifier", {})
    head_model = AnticipationHead(
        num_verb_classes=0 if action_only else num_classes["verb"],
        num_noun_classes=0 if action_only else num_classes["noun"],
        num_action_classes=num_classes["action"],
        embed_dim=embed_dim,
        num_heads=classifier_cfg.get("num_heads", 16),
        depth=classifier_cfg.get("num_probe_blocks", 4),
        repository_dir=get(config, "model.repository_dir"),
        use_temporal_metadata=bool(classifier_cfg.get("use_temporal_metadata", False)),
        temporal_duration_scale_sec=float(classifier_cfg.get("temporal_duration_scale_sec", 32.0)),
    ).to(device)
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

    run_dir = ensure_dir(expand_path(require(config, "experiment.output_dir")))
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_json(run_dir / "run_metadata.json", {
        "dataset": "ego4d_goalstep",
        "focal_gamma": focal_gamma, "focal_alpha": focal_alpha, "sampler": sampler_name,
        "train_heads": train_heads, "emitted_heads": train_heads, "evaluated_heads": train_heads,
        "seed": seed, "epochs": num_epochs, "batch_size": batch_size, "learning_rate": lr,
        "train_precision": precision, "eval_precision": "fp32",
        "labels_from_index": labels_from_index,
        "tau_a": get(config, "dataset.tau_a", 1.0), "l_obs": get(config, "dataset.l_obs", 3.5),
        "predictor_grid_fps": get(config, "dataset.frames_per_second", None),
        "frame_sampling": get(config, "dataset.frame_sampling", {"strategy": "uniform"}),
        "task_contract": get(config, "dataset.task_contract", None),
        "use_temporal_metadata": bool(classifier_cfg.get("use_temporal_metadata", False)),
        "taxonomy": num_classes, "index_dir": str(index_dir),
        "train_samples": len(train_dataset), "val_samples_full": len(full_val_dataset),
        "val_subset_size": len(subset_dataset), "val_subset_seed": subset_seed,
        "checkpoint_selection_metric": BEST_METRIC_NAME,
    })
    write_json(run_dir / "val_subset_sample_ids.json",
               {"seed": subset_seed, "size": len(subset_dataset), "sample_ids": subset_dataset.sample_ids})

    history_path = run_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_history_columns(train_heads))

    best_metric, best_epoch = float("-inf"), None
    per_epoch: list[dict] = []

    for epoch in range(1, num_epochs + 1):
        if isinstance(train_loader.sampler, tz1.ScenarioStratifiedSampler):
            train_loader.sampler.set_epoch(epoch)
        step_log(1, PHASE, f"Epoch {epoch}/{num_epochs}")
        t0 = time.time()
        train_loss = tz1.train_one_epoch(
            head_model, train_loader, optimizer, lr_sched, wd_sched, device, focal_gamma, focal_alpha,
            loss_heads=train_heads, amp_dtype=amp_dtype,
        )
        step_log(1, PHASE, f"Train loss: {train_loss:.4f}")

        result = tz1.evaluate(
            head_model, subset_loader, device, num_classes, bands, subset_scenarios, heads=train_heads
        )
        _attach_adaptive_stratified_metrics(result, val_index, num_classes, train_heads)
        _log_eval(f"Val[subset n={len(subset_dataset)}] epoch {epoch}", result, train_heads)
        elapsed = time.time() - t0

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.4f}",
                *(x for h in train_heads for x in (
                    f"{result['overall'][h]:.4f}",
                    f"{result['accuracy_top1'][h]:.4f}",
                    f"{result['accuracy_top5'][h]:.4f}",
                    f"{result['accuracy_top10'][h]:.4f}",
                    f"{result['accuracy_top15'][h]:.4f}",
                )),
                f"{elapsed:.1f}",
            ])
        per_epoch.append(_metrics_dict(result, train_loss=train_loss, epoch=epoch))
        write_json(run_dir / "metrics_per_epoch.json", per_epoch)

        selection_metric = result["accuracy_top5"]["action"]
        state = {"epoch": epoch, "model_state": head_model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "metric": selection_metric, "metric_name": BEST_METRIC_NAME,
                 "num_classes": num_classes}
        torch.save(state, ckpt_dir / f"epoch_{epoch:02d}.pt")
        torch.save(state, run_dir / "latest.pt")
        if selection_metric > best_metric:
            best_metric, best_epoch = selection_metric, epoch
            torch.save(state, run_dir / "best.pt")
            torch.save(state, run_dir / "best_action_top5.pt")
            step_log(1, PHASE, f"Best updated -> epoch {epoch} (val-subset action top5={best_metric:.2f})")

        tz1.save_likelihood_entropy(
            result["_preds"], subset_scenarios,
            run_dir / f"likelihood_entropy_epoch_{epoch:02d}.jsonl", heads=train_heads,
        )

    step_log(1, PHASE, f"Training done. Best epoch={best_epoch} (val-subset action top5={best_metric:.2f})")

    final = {
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": BEST_METRIC_NAME,
        "best_metric": best_metric,
        "val_subset": {"size": len(subset_dataset), "seed": subset_seed,
                       "metrics": per_epoch[best_epoch - 1] if best_epoch else None},
        "per_epoch": per_epoch,
    }
    if not args.skip_final_full_val and best_epoch is not None:
        step_log(1, PHASE, f"Final readout: best.pt (epoch {best_epoch}) on FULL val ({len(full_val_dataset)} samples)")
        head_model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model_state"])
        full_result = tz1.evaluate(
            head_model, full_val_loader, device, num_classes, bands, full_val_scenarios, heads=train_heads
        )
        _attach_adaptive_stratified_metrics(full_result, val_index, num_classes, train_heads)
        _log_eval(f"Val[FULL n={len(full_val_dataset)}] best epoch {best_epoch}", full_result, train_heads)
        step_log(1, PHASE, f"Val[FULL] action scenario breakdown: {full_result['scenario']['action']}")
        final["val_full"] = {"size": len(full_val_dataset), "metrics": _metrics_dict(full_result, epoch=best_epoch)}
        tz1.save_likelihood_entropy(
            full_result["_preds"], full_val_scenarios,
            run_dir / "likelihood_entropy_full_val_best.jsonl", heads=train_heads,
        )

    write_json(run_dir / "final_metrics.json", final)
    step_log(1, PHASE, f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
