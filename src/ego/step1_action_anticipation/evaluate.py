"""Evaluation scaffold for Step 1 action anticipation.

Reads an exported ``action_candidates.jsonl`` (produced by ``infer.py``) and
computes the metric suite required for a Step 1 baseline report: Top-K
recall, class-mean recall, joint verb+noun rate, head/tail class breakdown,
and prediction class distribution.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ego.common.config import get, load_config, require
from ego.common.io import ensure_dir, read_jsonl, write_json
from ego.common.logging import step_log
from ego.common.paths import expand_path


def _hit_at_k(candidates: list[dict], id_field: str, gt_id, k: int) -> bool:
    if gt_id is None:
        return False
    return any(c.get(id_field) == gt_id for c in candidates if c.get("rank", k + 1) <= k)


def _instance_recall_and_class_mean(
    records: list[dict], candidates_field: str, gt_id_field: str, id_field: str, k: int
) -> tuple[float, float, dict]:
    hits = []
    support: Counter = Counter()
    per_class_hits: Counter = Counter()
    for r in records:
        gt_id = r.get("gt", {}).get(gt_id_field) if r.get("gt") else None
        if gt_id is None:
            continue
        hit = _hit_at_k(r.get(candidates_field, []), id_field, gt_id, k)
        hits.append(hit)
        support[gt_id] += 1
        if hit:
            per_class_hits[gt_id] += 1

    instance_recall = 100.0 * sum(hits) / len(hits) if hits else float("nan")
    per_class_recall = {cid: 100.0 * per_class_hits[cid] / n for cid, n in support.items()}
    class_mean = sum(per_class_recall.values()) / len(per_class_recall) if per_class_recall else float("nan")
    return instance_recall, class_mean, {"support": dict(support), "per_class_recall": per_class_recall}


def _joint_recall(records: list[dict], k: int) -> float:
    hits = []
    for r in records:
        gt = r.get("gt") or {}
        if gt.get("verb_id") is None or gt.get("noun_id") is None:
            continue
        verb_hit = _hit_at_k(r.get("verb_candidates", []), "verb_id", gt.get("verb_id"), k)
        noun_hit = _hit_at_k(r.get("noun_candidates", []), "noun_id", gt.get("noun_id"), k)
        hits.append(verb_hit and noun_hit)
    return 100.0 * sum(hits) / len(hits) if hits else float("nan")


def _head_tail_split(per_class_recall: dict, support: dict, head_fraction: float = 0.2) -> dict:
    ordered = sorted(support.items(), key=lambda kv: -kv[1])
    n_head = max(1, int(len(ordered) * head_fraction))
    head_ids = {cid for cid, _ in ordered[:n_head]}
    head_scores = [per_class_recall[c] for c in head_ids if c in per_class_recall]
    tail_scores = [v for c, v in per_class_recall.items() if c not in head_ids]
    return {
        "head_class_mean_recall": sum(head_scores) / len(head_scores) if head_scores else float("nan"),
        "tail_class_mean_recall": sum(tail_scores) / len(tail_scores) if tail_scores else float("nan"),
        "num_head_classes": len(head_ids),
        "num_tail_classes": len(per_class_recall) - len(head_ids),
    }


def _prediction_distribution(records: list[dict], candidates_field: str, label_field: str) -> dict:
    counts: Counter = Counter()
    for r in records:
        top1 = next((c for c in r.get(candidates_field, []) if c.get("rank") == 1), None)
        if top1 is not None and top1.get(label_field) is not None:
            counts[top1[label_field]] += 1
    return dict(counts)


def _write_distribution_csv(path: Path, verb_dist: dict, noun_dist: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "label", "count"])
        for label, count in sorted(verb_dist.items(), key=lambda kv: -kv[1]):
            writer.writerow(["verb", label, count])
        for label, count in sorted(noun_dist.items(), key=lambda kv: -kv[1]):
            writer.writerow(["noun", label, count])


def evaluate(config_path: str) -> dict:
    config = load_config(config_path)
    step_log(1, "Evaluate", "Config loaded")

    candidates_path = expand_path(require(config, "inference.output_path"))
    k = get(config, "inference.top_k", 5)
    records = list(read_jsonl(candidates_path))
    step_log(1, "Evaluate", f"Loaded {len(records)} candidate records from {candidates_path}")

    verb_recall, verb_cmr, verb_detail = _instance_recall_and_class_mean(
        records, "verb_candidates", "verb_id", "verb_id", k
    )
    noun_recall, noun_cmr, noun_detail = _instance_recall_and_class_mean(
        records, "noun_candidates", "noun_id", "noun_id", k
    )
    action_recall, action_cmr, action_detail = _instance_recall_and_class_mean(
        records, "candidates", "action_id", "action_id", k
    )
    joint_recall = _joint_recall(records, k)

    metrics = {
        "num_samples": len(records),
        "k": k,
        f"verb_recall@{k}": verb_recall,
        f"noun_recall@{k}": noun_recall,
        f"action_recall@{k}": action_recall,
        f"verb_class_mean_recall@{k}": verb_cmr,
        f"noun_class_mean_recall@{k}": noun_cmr,
        f"action_class_mean_recall@{k}": action_cmr,
        f"verb_noun_joint_recall@{k}": joint_recall,
        "verb_head_tail": _head_tail_split(verb_detail["per_class_recall"], verb_detail["support"]),
        "noun_head_tail": _head_tail_split(noun_detail["per_class_recall"], noun_detail["support"]),
        "action_head_tail": _head_tail_split(action_detail["per_class_recall"], action_detail["support"]),
    }

    step_log(1, "Evaluate", f"Verb Recall@{k}: {verb_recall:.2f}  Class-Mean: {verb_cmr:.2f}")
    step_log(1, "Evaluate", f"Noun Recall@{k}: {noun_recall:.2f}  Class-Mean: {noun_cmr:.2f}")
    step_log(1, "Evaluate", f"Action Recall@{k}: {action_recall:.2f}  Class-Mean: {action_cmr:.2f}")
    step_log(1, "Evaluate", f"Verb+Noun joint Recall@{k}: {joint_recall:.2f}")

    output_dir = expand_path(get(config, "experiment.output_dir", str(candidates_path.parent)))
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    step_log(1, "Evaluate", f"Metrics written: {metrics_path}")

    dist_path = output_dir / "class_distribution.csv"
    _write_distribution_csv(
        dist_path,
        _prediction_distribution(records, "verb_candidates", "verb"),
        _prediction_distribution(records, "noun_candidates", "noun"),
    )
    step_log(1, "Evaluate", f"Prediction class distribution written: {dist_path}")

    return metrics
