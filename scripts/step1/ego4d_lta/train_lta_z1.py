"""Train Step 1 verb/noun/action heads on Ego4D LTA Z=1 samples.

Architecture is untouched: reuses ``AnticipationHead`` (frozen V-JEPA2
backbone -> attentive probe, 3 query tokens -> independent verb/noun/action
linear heads), the same ``sigmoid_focal_loss`` and warmup-cosine
LR/weight-decay schedules, and ``class_mean_recall`` from
``ego.step1_action_anticipation.{train,metrics}`` -- only the data (Ego4D LTA
feature cache instead of EK100 video), output dimensions (taxonomy size),
and focal-loss gamma/alpha come from config. Trains against a pre-built
feature cache (run ``extract_features.py`` first).

Usage:
    python scripts/step1/ego4d_lta/train_lta_z1.py --config configs/step1/ego4d_lta/pilot.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.exceptions import EgoConfigError  # noqa: E402
from ego.common.io import ensure_dir, write_json, write_jsonl, write_yaml  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.common.seed import set_seed  # noqa: E402
from ego.datasets.ego4d import index_scenario_lookup  # noqa: E402
from ego.datasets.ego4d_stats import class_frequency, head_mid_tail_bands  # noqa: E402
from ego.datasets.label_mapping import LabelMapping  # noqa: E402
from ego.step1_action_anticipation.data.collator import anticipation_collate  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset  # noqa: E402
from ego.step1_action_anticipation.metrics import (  # noqa: E402
    class_mean_recall,
    per_class_recall,
    prediction_entropy,
    top_k_recall,
)
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402
from ego.step1_action_anticipation.train import _CosineWD, _WarmupCosineLR, sigmoid_focal_loss  # noqa: E402

HEADS = ("verb", "noun", "action")


class ScenarioStratifiedSampler(Sampler):
    """Round-robin over scenarios so large scenarios don't dominate a batch/epoch.

    Cycles through scenarios in a per-epoch-shuffled order, popping one
    sample index from each scenario's own (separately shuffled, recycled on
    exhaustion) pool, until an epoch's worth of indices is produced. This is
    scenario-balanced draw order, not scenario-balanced *frequency*: small
    scenarios get reused (with reshuffling) more often per epoch than large
    ones, rather than being proportionally under-represented.
    """

    def __init__(self, scenarios: list[str], seed: int = 42) -> None:
        self.scenarios = scenarios
        self.seed = seed
        self.by_scenario: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(scenarios):
            self.by_scenario[s].append(i)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        pools = {s: idxs.copy() for s, idxs in self.by_scenario.items()}
        for idxs in pools.values():
            rng.shuffle(idxs)
        cursors = dict.fromkeys(pools, 0)
        scenario_names = list(pools.keys())

        order: list[int] = []
        total = len(self.scenarios)
        while len(order) < total:
            rng.shuffle(scenario_names)
            for s in scenario_names:
                if cursors[s] >= len(pools[s]):
                    rng.shuffle(pools[s])
                    cursors[s] = 0
                order.append(pools[s][cursors[s]])
                cursors[s] += 1
                if len(order) >= total:
                    break
        yield from order


def _read_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _find_index_file(index_dir: Path, split: str) -> Path:
    for ext in (".parquet", ".csv"):
        p = index_dir / f"{split}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No {split}.parquet or {split}.csv found under {index_dir}")


def _load_registry(path: Path) -> LabelMapping:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    verb_classes = {int(k): v for k, v in data["verb_classes"].items()}
    noun_classes = {int(k): v for k, v in data["noun_classes"].items()}
    action_classes = {}
    for key, action_id in data["action_classes"].items():
        v, n = key.split("|")
        action_classes[(int(v), int(n))] = action_id
    return LabelMapping(verb_classes=verb_classes, noun_classes=noun_classes, action_classes=action_classes)


def _unified_class_frequency(index_df: pd.DataFrame, mapping: LabelMapping, head: str) -> dict[int, int]:
    if head == "verb":
        ids = [mapping.verb_classes[v] for v in index_df["verb_label"] if v in mapping.verb_classes]
    elif head == "noun":
        ids = [mapping.noun_classes[n] for n in index_df["noun_label"] if n in mapping.noun_classes]
    else:
        ids = [
            mapping.action_classes[(v, n)]
            for v, n in zip(index_df["verb_label"], index_df["noun_label"])
            if (v, n) in mapping.action_classes
        ]
    return dict(class_frequency(pd.DataFrame({"x": ids}), "x"))


def _video_of(sample_id: str) -> str:
    """``{video_uuid}_{row_index}`` -> ``{video_uuid}``."""
    return sample_id.rsplit("_", 1)[0]


def subsample_by_video(
    sample_ids: list[str], max_per_video: int | None, max_videos: int | None, seed: int
) -> list[str]:
    """Thin the training pool along the *video* axis rather than the sample axis.

    GoalStep yields ~53 samples per video from only 570 train videos, so samples
    within a video share almost all of their visual context. ``max_per_video``
    caps samples while keeping every video; ``max_videos`` keeps every sample of
    a video subset. Holding the resulting sample count equal between the two
    separates "how many samples" from "how many distinct videos".
    """
    if max_per_video is None and max_videos is None:
        return sample_ids

    by_video: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(sample_ids):
        by_video[_video_of(sid)].append(sid)

    rng = random.Random(seed)
    videos = sorted(by_video)
    if max_videos is not None:
        rng.shuffle(videos)
        videos = sorted(videos[:max_videos])

    kept: list[str] = []
    for video in videos:
        ids = by_video[video]
        if max_per_video is not None and len(ids) > max_per_video:
            ids = sorted(rng.sample(ids, max_per_video))
        kept.extend(ids)
    return sorted(kept)


def _build_train_loader(
    cache_dir: Path, batch_size: int, sampler_name: str, scenario_lookup: dict, seed: int, num_workers: int = 0,
    max_per_video: int | None = None, max_videos: int | None = None, max_samples: int | None = None,
):
    dataset = FeatureCacheDataset.from_cache_dir(cache_dir)
    if max_per_video is not None or max_videos is not None:
        dataset = FeatureCacheDataset(
            subsample_by_video(dataset.sample_ids, max_per_video, max_videos, seed), cache_dir
        )
    if max_samples is not None and max_samples < len(dataset.sample_ids):
        # Plain random subset -- the data-scaling axis, orthogonal to the video-axis
        # thinning above. Seeded so a given (seed, max_samples) is reproducible and
        # nested subsets stay comparable across runs.
        ids = sorted(random.Random(seed).sample(sorted(dataset.sample_ids), max_samples))
        dataset = FeatureCacheDataset(ids, cache_dir)
    if len(dataset) == 0:
        raise EgoConfigError(
            f"No cached features found under {cache_dir}. Run extract_features.py for this split first."
        )
    scenarios = [scenario_lookup.get(sid, "unknown") for sid in dataset.sample_ids]

    sampler = None
    shuffle = True
    if sampler_name == "scenario_stratified":
        sampler = ScenarioStratifiedSampler(scenarios, seed=seed)
        shuffle = False
    elif sampler_name != "random":
        raise EgoConfigError(f"Unknown sampler {sampler_name!r}; expected 'random' or 'scenario_stratified'")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=anticipation_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader, scenarios


def _build_eval_loader(cache_dir: Path, batch_size: int, scenario_lookup: dict, num_workers: int = 0):
    """Always sequential (``shuffle=False``, no sampler): evaluation code pairs
    ``compute_predictions``' collected ``sample_ids``/logits back to this
    function's ``scenarios`` list positionally, so iteration order must be
    reproducible and match ``dataset.sample_ids``'s order exactly."""
    dataset = FeatureCacheDataset.from_cache_dir(cache_dir)
    if len(dataset) == 0:
        raise EgoConfigError(
            f"No cached features found under {cache_dir}. Run extract_features.py for this split first."
        )
    scenarios = [scenario_lookup.get(sid, "unknown") for sid in dataset.sample_ids]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=anticipation_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader, scenarios


def train_one_epoch(
    head_model, loader, optimizer, lr_sched, wd_sched, device, gamma, alpha,
    loss_heads=HEADS, amp_dtype=None,
) -> float:
    """Train one epoch using supervision from only ``loss_heads``.

    The model may still emit verb/noun/action logits for evaluation even when
    only the action head contributes to the optimization objective.

    ``amp_dtype`` (e.g. ``torch.bfloat16``) runs the probe forward + loss under
    autocast. Training only -- evaluation and the exported likelihood/entropy
    artifacts stay fp32 on purpose, so Step 2/3 consume full-precision
    probabilities. Measured on this machine (H200, 40 real GoalStep batches,
    identical seed/order): 6.4x faster, final-loss delta 0.02%, top-1 agreement
    100%, max softmax delta 1.9e-4, entropy delta <= 5e-4 nats. Safe because
    ``binary_cross_entropy_with_logits`` autocast-promotes to fp32 and AdamW
    keeps fp32 master weights (so bf16 needs no GradScaler).
    """
    invalid_heads = set(loss_heads) - set(HEADS)
    if not loss_heads or invalid_heads:
        raise EgoConfigError(
            f"loss_heads must be a non-empty subset of {HEADS}; got {list(loss_heads)}"
        )
    head_model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        features = batch["video"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = _forward_head(head_model, features, batch, device)
            loss = sum(
                sigmoid_focal_loss(logits[h], batch[f"{h}_id"].to(device), alpha=alpha, gamma=gamma)
                for h in loss_heads
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()
        wd_sched.step()
        total_loss += float(loss.detach().cpu())
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def compute_predictions(head_model, loader, device, heads=HEADS) -> dict:
    head_model.eval()
    logits_all = {h: [] for h in heads}
    labels_all = {h: [] for h in heads}
    sample_ids: list[str] = []
    for batch in loader:
        features = batch["video"].to(device)
        logits = _forward_head(head_model, features, batch, device)
        for h in heads:
            logits_all[h].append(logits[h].cpu())
            labels_all[h].append(batch[f"{h}_id"].cpu())
        sample_ids.extend(batch["sample_id"])
    for h in heads:
        logits_all[h] = torch.cat(logits_all[h])
        labels_all[h] = torch.cat(labels_all[h])
    return {"logits": logits_all, "labels": labels_all, "sample_ids": sample_ids}


def _forward_head(head_model, features, batch: dict, device):
    """Forward cached features with optional adaptive-window time metadata."""
    if not getattr(head_model, "use_temporal_metadata", False):
        return head_model(features)
    required = (
        "observation_duration_sec",
        "observed_action_duration_sec",
        "frame_time_positions",
        "frame_terminal_mask",
        "annotation_level_id",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise EgoConfigError(f"Temporal-metadata head is missing cached fields: {missing}")
    return head_model(
        features,
        observation_duration_sec=batch["observation_duration_sec"].to(device),
        observed_action_duration_sec=batch["observed_action_duration_sec"].to(device),
        frame_time_positions=batch["frame_time_positions"].to(device),
        frame_terminal_mask=batch["frame_terminal_mask"].to(device),
        annotation_level_id=batch["annotation_level_id"].to(device),
    )


def band_breakdown(logits, labels, num_classes: int, bands: dict[int, str], k: int = 5) -> dict[str, float]:
    recall = per_class_recall(logits, labels, num_classes, k=k)
    grouped: dict[str, list[float]] = defaultdict(list)
    for cls, band in bands.items():
        if cls < num_classes and not torch.isnan(recall[cls]):
            grouped[band].append(float(recall[cls]))
    return {band: (100.0 * sum(v) / len(v) if v else float("nan")) for band, v in grouped.items()}


def scenario_breakdown(logits, labels, num_classes: int, scenarios: list[str], k: int = 5) -> dict[str, float]:
    by_scenario: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scenarios):
        by_scenario[s].append(i)
    out = {}
    for s, idxs in by_scenario.items():
        idx_t = torch.tensor(idxs, dtype=torch.long)
        out[s] = class_mean_recall(logits[idx_t], labels[idx_t], num_classes, k=k)
    return out


def evaluate(
    head_model, loader, device, num_classes: dict, bands: dict, scenarios: list[str],
    k: int = 5, heads=HEADS,
) -> dict:
    preds = compute_predictions(head_model, loader, device, heads=heads)
    result: dict = {
        "overall": {}, "band": {}, "scenario": {},
        "accuracy_top1": {}, "accuracy_top5": {},
        "accuracy_top10": {}, "accuracy_top15": {},
    }
    for h in heads:
        logits, labels = preds["logits"][h], preds["labels"][h]
        result["overall"][h] = class_mean_recall(logits, labels, num_classes[h], k=k)
        result["band"][h] = band_breakdown(logits, labels, num_classes[h], bands[h], k=k)
        # Simple (micro, instance-level) accuracy alongside the class-mean recall above --
        # distinct metric (not weighted equally per class), always logged together per user request.
        result["accuracy_top1"][h] = top_k_recall(logits, labels, k=1)
        result["accuracy_top5"][h] = top_k_recall(logits, labels, k=5)
        result["accuracy_top10"][h] = top_k_recall(logits, labels, k=10)
        result["accuracy_top15"][h] = top_k_recall(logits, labels, k=15)
        result["scenario"][h] = scenario_breakdown(logits, labels, num_classes[h], scenarios, k=k)
    result["_preds"] = preds
    return result


def save_likelihood_entropy(preds: dict, scenarios: list[str], path: Path, heads=HEADS) -> None:
    sample_ids = preds["sample_ids"]
    records = []
    per_head_probs = {}
    per_head_entropy = {}
    for h in heads:
        probs = torch.softmax(preds["logits"][h], dim=-1)
        per_head_probs[h] = probs.max(dim=-1).values.tolist()
        per_head_entropy[h] = prediction_entropy(preds["logits"][h]).tolist()
    for i, sid in enumerate(sample_ids):
        record = {"sample_id": sid, "scenario": scenarios[i] if i < len(scenarios) else "unknown"}
        for h in heads:
            record[f"{h}_likelihood"] = per_head_probs[h][i]
            record[f"{h}_entropy"] = per_head_entropy[h][i]
        records.append(record)
    write_jsonl(path, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        help="Path to a previous run's latest.pt (e.g. from a shorter training.epochs run). "
        "Continues from checkpoint['epoch']+1 up to this config's training.epochs, restoring "
        "model/optimizer state and fast-forwarding the LR/WD schedules so the cosine curve is "
        "continuous across the two runs (both runs must share the same training.epochs, "
        "batch_size, and dataset for the schedule/steps-per-epoch math to line up).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = get(config, "experiment.seed", 42)
    set_seed(seed)
    step_log(1, "TrainLTAZ1", "Config loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index_dir = expand_path(require(config, "dataset.index_dir"))
    mapping = _load_registry(index_dir / "action_registry.json")
    num_classes = {"verb": mapping.num_verbs, "noun": mapping.num_nouns, "action": mapping.num_actions}
    step_log(1, "TrainLTAZ1", f"Taxonomy: verb={num_classes['verb']} noun={num_classes['noun']} action={num_classes['action']}")

    train_index = _read_index(_find_index_file(index_dir, "train"))
    dev_index = _read_index(_find_index_file(index_dir, "dev"))
    train_scenario_lookup = index_scenario_lookup(train_index)
    dev_scenario_lookup = index_scenario_lookup(dev_index)

    cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
    batch_size = require(config, "training.batch_size")
    sampler_name = get(config, "training.sampler", "random")
    num_workers = get(config, "dataset.num_workers", 0)
    train_dataset, train_loader, train_scenarios = _build_train_loader(
        cache_dir / "train", batch_size, sampler_name, train_scenario_lookup, seed, num_workers=num_workers
    )
    dev_dataset, dev_loader, dev_scenarios = _build_eval_loader(
        cache_dir / "dev", batch_size, dev_scenario_lookup, num_workers=num_workers
    )
    step_log(1, "TrainLTAZ1", f"Train samples: {len(train_dataset)}  Dev samples: {len(dev_dataset)}")
    step_log(1, "TrainLTAZ1", f"Sampler: {sampler_name}")

    bands = {h: head_mid_tail_bands(_unified_class_frequency(train_index, mapping, h)) for h in HEADS}

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
        use_temporal_metadata=bool(classifier_cfg.get("use_temporal_metadata", False)),
        temporal_duration_scale_sec=float(classifier_cfg.get("temporal_duration_scale_sec", 32.0)),
    ).to(device)

    num_epochs = require(config, "training.epochs")
    iterations_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * iterations_per_epoch
    warmup_epochs = get(config, "training.warmup_epochs", 0)
    lr = require(config, "training.learning_rate")
    wd = get(config, "training.weight_decay", 0.0001)

    optimizer = torch.optim.AdamW(head_model.parameters(), lr=lr, weight_decay=wd)
    lr_sched = _WarmupCosineLR(
        optimizer, ref_lr=lr, start_lr=get(config, "training.start_lr", 0.0),
        final_lr=get(config, "training.final_lr", 0.0),
        warmup_steps=int(warmup_epochs * iterations_per_epoch), total_steps=total_steps,
    )
    wd_sched = _CosineWD(
        optimizer, ref_wd=wd, final_wd=get(config, "training.final_weight_decay", wd), total_steps=total_steps
    )
    focal_gamma = get(config, "training.focal_gamma", 2.0)
    focal_alpha = get(config, "training.focal_alpha", 0.25)

    run_dir = ensure_dir(expand_path(require(config, "experiment.output_dir")))
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_json(
        run_dir / "run_metadata.json",
        {
            "focal_gamma": focal_gamma, "focal_alpha": focal_alpha, "sampler": sampler_name,
            "seed": seed, "taxonomy": {"num_verbs": num_classes["verb"], "num_nouns": num_classes["noun"],
            "num_actions": num_classes["action"]}, "index_dir": str(index_dir),
        },
    )

    start_epoch = 1
    best_metric = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        head_model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        completed_steps = checkpoint["epoch"] * iterations_per_epoch
        lr_sched._step = completed_steps
        wd_sched._step = completed_steps
        best_ckpt_path = run_dir / "best_action.pt"
        if best_ckpt_path.exists():
            best_metric = torch.load(best_ckpt_path, map_location="cpu").get("metric", float("-inf"))
        step_log(
            1, "TrainLTAZ1",
            f"Resumed from {args.resume}: epoch {checkpoint['epoch']} -> continuing at epoch {start_epoch} "
            f"(target {num_epochs}, best_action so far={best_metric:.2f})",
        )
        if start_epoch > num_epochs:
            raise EgoConfigError(
                f"--resume checkpoint is already at epoch {checkpoint['epoch']} >= this config's "
                f"training.epochs={num_epochs}; raise training.epochs to continue."
            )

    history_path = run_dir / "training_history.csv"
    if not (args.resume and history_path.exists()):
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "verb_cmr@5", "noun_cmr@5", "action_cmr@5", "seconds"])

    for epoch in range(start_epoch, num_epochs + 1):
        if isinstance(train_loader.sampler, ScenarioStratifiedSampler):
            train_loader.sampler.set_epoch(epoch)
        step_log(1, "TrainLTAZ1", f"Epoch {epoch}/{num_epochs}")
        t0 = time.time()
        train_loss = train_one_epoch(head_model, train_loader, optimizer, lr_sched, wd_sched, device, focal_gamma, focal_alpha)
        step_log(1, "TrainLTAZ1", f"Train loss: {train_loss:.4f}")

        eval_result = evaluate(head_model, dev_loader, device, num_classes, bands, dev_scenarios)
        for h in HEADS:
            step_log(1, "TrainLTAZ1", f"Dev {h} class-mean Recall@5: {eval_result['overall'][h]:.2f}")
            step_log(1, "TrainLTAZ1", f"Dev {h} band breakdown: {eval_result['band'][h]}")
            step_log(
                1, "TrainLTAZ1",
                f"Dev {h} simple accuracy: top1={eval_result['accuracy_top1'][h]:.2f}  "
                f"top5={eval_result['accuracy_top5'][h]:.2f}",
            )
        step_log(1, "TrainLTAZ1", f"Dev action scenario breakdown: {eval_result['scenario']['action']}")

        elapsed = time.time() - t0
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", eval_result["overall"]["verb"], eval_result["overall"]["noun"],
                 eval_result["overall"]["action"],
                 eval_result["accuracy_top1"]["verb"], eval_result["accuracy_top5"]["verb"],
                 eval_result["accuracy_top1"]["noun"], eval_result["accuracy_top5"]["noun"],
                 eval_result["accuracy_top1"]["action"], eval_result["accuracy_top5"]["action"],
                 f"{elapsed:.1f}"]
            )

        action_metric = eval_result["overall"]["action"]
        torch.save(
            {"epoch": epoch, "model_state": head_model.state_dict(), "optimizer_state": optimizer.state_dict()},
            run_dir / "latest.pt",
        )
        if action_metric > best_metric:
            best_metric = action_metric
            torch.save({"epoch": epoch, "model_state": head_model.state_dict(), "metric": action_metric}, run_dir / "best_action.pt")
            step_log(1, "TrainLTAZ1", f"Best checkpoint updated: {run_dir / 'best_action.pt'} (action_cmr@5={action_metric:.2f})")

        write_json(
            run_dir / "metrics.json",
            {"epoch": epoch, "train_loss": train_loss, "overall": eval_result["overall"],
             "band": eval_result["band"], "scenario": eval_result["scenario"],
             "accuracy_top1": eval_result["accuracy_top1"], "accuracy_top5": eval_result["accuracy_top5"]},
        )
        save_likelihood_entropy(eval_result["_preds"], dev_scenarios, run_dir / "likelihood_entropy.jsonl")

    step_log(1, "TrainLTAZ1", f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
