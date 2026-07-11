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
from ego.step1_action_anticipation.metrics import class_mean_recall, per_class_recall, prediction_entropy  # noqa: E402
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


def _build_train_loader(cache_dir: Path, batch_size: int, sampler_name: str, scenario_lookup: dict, seed: int):
    dataset = FeatureCacheDataset.from_cache_dir(cache_dir)
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
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=anticipation_collate
    )
    return dataset, loader, scenarios


def _build_eval_loader(cache_dir: Path, batch_size: int, scenario_lookup: dict):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=anticipation_collate)
    return dataset, loader, scenarios


def train_one_epoch(head_model, loader, optimizer, lr_sched, wd_sched, device, gamma, alpha) -> float:
    head_model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        features = batch["video"].to(device)
        logits = head_model(features)
        loss = sum(
            sigmoid_focal_loss(logits[h], batch[f"{h}_id"].to(device), alpha=alpha, gamma=gamma) for h in HEADS
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
def compute_predictions(head_model, loader, device) -> dict:
    head_model.eval()
    logits_all = {h: [] for h in HEADS}
    labels_all = {h: [] for h in HEADS}
    sample_ids: list[str] = []
    for batch in loader:
        features = batch["video"].to(device)
        logits = head_model(features)
        for h in HEADS:
            logits_all[h].append(logits[h].cpu())
            labels_all[h].append(batch[f"{h}_id"].cpu())
        sample_ids.extend(batch["sample_id"])
    for h in HEADS:
        logits_all[h] = torch.cat(logits_all[h])
        labels_all[h] = torch.cat(labels_all[h])
    return {"logits": logits_all, "labels": labels_all, "sample_ids": sample_ids}


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


def evaluate(head_model, loader, device, num_classes: dict, bands: dict, scenarios: list[str], k: int = 5) -> dict:
    preds = compute_predictions(head_model, loader, device)
    result: dict = {"overall": {}, "band": {}, "scenario": {}}
    for h in HEADS:
        logits, labels = preds["logits"][h], preds["labels"][h]
        result["overall"][h] = class_mean_recall(logits, labels, num_classes[h], k=k)
        result["band"][h] = band_breakdown(logits, labels, num_classes[h], bands[h], k=k)
        result["scenario"][h] = scenario_breakdown(logits, labels, num_classes[h], scenarios, k=k)
    result["_preds"] = preds
    return result


def save_likelihood_entropy(preds: dict, scenarios: list[str], path: Path) -> None:
    sample_ids = preds["sample_ids"]
    records = []
    per_head_probs = {}
    per_head_entropy = {}
    for h in HEADS:
        probs = torch.softmax(preds["logits"][h], dim=-1)
        per_head_probs[h] = probs.max(dim=-1).values.tolist()
        per_head_entropy[h] = prediction_entropy(preds["logits"][h]).tolist()
    for i, sid in enumerate(sample_ids):
        record = {"sample_id": sid, "scenario": scenarios[i] if i < len(scenarios) else "unknown"}
        for h in HEADS:
            record[f"{h}_likelihood"] = per_head_probs[h][i]
            record[f"{h}_entropy"] = per_head_entropy[h][i]
        records.append(record)
    write_jsonl(path, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
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
    train_dataset, train_loader, train_scenarios = _build_train_loader(
        cache_dir / "train", batch_size, sampler_name, train_scenario_lookup, seed
    )
    dev_dataset, dev_loader, dev_scenarios = _build_eval_loader(
        cache_dir / "dev", batch_size, dev_scenario_lookup
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

    history_path = run_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "verb_cmr@5", "noun_cmr@5", "action_cmr@5", "seconds"])

    best_metric = float("-inf")
    for epoch in range(1, num_epochs + 1):
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
        step_log(1, "TrainLTAZ1", f"Dev action scenario breakdown: {eval_result['scenario']['action']}")

        elapsed = time.time() - t0
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", eval_result["overall"]["verb"], eval_result["overall"]["noun"],
                 eval_result["overall"]["action"], f"{elapsed:.1f}"]
            )

        action_metric = eval_result["overall"]["action"]
        torch.save({"epoch": epoch, "model_state": head_model.state_dict()}, run_dir / "latest.pt")
        if action_metric > best_metric:
            best_metric = action_metric
            torch.save({"epoch": epoch, "model_state": head_model.state_dict(), "metric": action_metric}, run_dir / "best_action.pt")
            step_log(1, "TrainLTAZ1", f"Best checkpoint updated: {run_dir / 'best_action.pt'} (action_cmr@5={action_metric:.2f})")

        write_json(
            run_dir / "metrics.json",
            {"epoch": epoch, "train_loss": train_loss, "overall": eval_result["overall"],
             "band": eval_result["band"], "scenario": eval_result["scenario"]},
        )
        save_likelihood_entropy(eval_result["_preds"], dev_scenarios, run_dir / "likelihood_entropy.jsonl")

    step_log(1, "TrainLTAZ1", f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
