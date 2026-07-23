#!/usr/bin/env python3
"""Run the leakage-safe GoalStep history Phase-0 gates and ensembles.

This runner deliberately separates two contracts:

* ``endpoint`` (primary/deployable): observe A2 through A2.end-1s and predict
  the next strict same-level action A3.  The recognition epoch-15 probe and
  all eight direct-next checkpoints share one ordered cache pass.
* ``later-anchor`` (optional benchmark only): evaluate start-1s probes on the
  reconstructed A3 start-window.  This uses knowledge of the future A3
  boundary and must never be reported as an A2-boundary deployable ensemble.

P0-b is evaluated before any P0-a result.  Its transition matrix is built
only from the 29,293-row training index.  Alpha and recognition temperature
are selected out-of-fold with video-disjoint two-fold validation, and the
pre-registered gate is exactly OOF Action Top-5 >= 27.7.

Examples (run from the repository root)::

    python scripts/step1/goalstep/run_history_phase0.py --stage gate
    python scripts/step1/goalstep/run_history_phase0.py --stage primary
    python scripts/step1/goalstep/run_history_phase0.py --stage mixed

The first command writes endpoint logits.  Later commands validate and reuse
that artifact, so the 60 GB endpoint validation cache is not reread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "step1" / "ego4d_lta"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_lta_z1 as tz1  # noqa: E402
from ego.common.config import get, load_config  # noqa: E402
from ego.step1_action_anticipation.data.collator import anticipation_collate  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset  # noqa: E402
from ego.step1_action_anticipation.metrics import class_mean_recall, top_k_recall  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402


FORMAT_VERSION = 1
HEADS = ("verb", "noun", "action")
DEFAULT_NEXT_INDEX = "src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8_next_action"
DEFAULT_ENDPOINT_INDEX = "src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8"
DEFAULT_ENDPOINT_CACHE = "../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna"
DEFAULT_ENDPOINT_CONFIG = "configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10.yaml"
DEFAULT_RECOGNITION_CONFIG = "configs/step1/goalstep/z1_end_m1_lobs8_vna.yaml"
DEFAULT_RECOGNITION_CHECKPOINT = "outputs/goalstep/runs/z1_end_m1_lobs8_vna/best.pt"
DEFAULT_NEXT_CHECKPOINT_DIR = "outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10/checkpoints"
DEFAULT_NEXT_BEST_CHECKPOINT = "outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10/best.pt"
DEFAULT_START8_INDEX = "src/ego/step1_action_anticipation/goalstep/index_start_m1_lobs8"
DEFAULT_START8_CACHE = "../datasets/Ego4D/goalstep_feature_cache_start_m1_lobs8_vna"
DEFAULT_START8_CONFIG = "configs/step1/goalstep/z1_start_m1_lobs8_vna.yaml"
DEFAULT_START8_CHECKPOINT_DIR = "outputs/goalstep/runs/z1_start_m1_lobs8_vna/checkpoints"
DEFAULT_START16_INDEX = "src/ego/step1_action_anticipation/goalstep/index_start_m1_lobs16"
DEFAULT_START16_CACHE = "../datasets/Ego4D/goalstep_feature_cache_start_m1_lobs16_vna"
DEFAULT_START16_CONFIG = "configs/step1/goalstep/z1_start_m1_lobs16_vna.yaml"
DEFAULT_START16_CHECKPOINT_DIR = "outputs/goalstep/runs/z1_start_m1_lobs16_vna/checkpoints"
DEFAULT_OUTPUT = "outputs/goalstep/runs/history_context_phase0"


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path
    expected_epoch: int
    config_path: Path


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _read_index(index_dir: Path, split: str) -> pd.DataFrame:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = index_dir / f"{split}{suffix}"
        if path.is_file():
            return reader(path).reset_index(drop=True)
    raise FileNotFoundError(f"No {split}.parquet or {split}.csv under {index_dir}")


def _parse_grid(value: str, *, name: str, positive: bool = False) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name} grid: {value!r}") from exc
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{name} grid must contain finite values")
    if positive and any(item <= 0 for item in values):
        raise ValueError(f"{name} values must be > 0")
    if not positive and any(item < 0 for item in values):
        raise ValueError(f"{name} values must be >= 0")
    return values


def _checkpoint_signature(spec: CheckpointSpec) -> dict[str, Any]:
    stat = spec.path.stat()
    return {
        "name": spec.name,
        "path": str(spec.path),
        "expected_epoch": spec.expected_epoch,
        "config_path": str(spec.config_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temp)
    os.replace(temp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _load_torch(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def _checkpoint_model_states_equal(left: Path, right: Path) -> bool:
    """Exact checkpoint alias check used for the audited epoch-3 best copy."""
    if left.resolve() == right.resolve():
        return True
    left_record = _load_torch(left, mmap=True)
    right_record = _load_torch(right, mmap=True)
    left_state = left_record.get("model_state", {})
    right_state = right_record.get("model_state", {})
    return (
        int(left_record.get("epoch", -1)) == int(right_record.get("epoch", -2))
        and left_state.keys() == right_state.keys()
        and all(torch.equal(left_state[key], right_state[key]) for key in left_state)
    )


def _stable_video_folds(video_uids: Sequence[str], seed: int) -> torch.Tensor:
    folds = []
    for video_uid in video_uids:
        digest = hashlib.sha256(f"{seed}|{video_uid}".encode("utf-8")).digest()
        folds.append(int.from_bytes(digest[:8], "big") % 2)
    result = torch.tensor(folds, dtype=torch.long)
    if set(result.tolist()) != {0, 1}:
        raise RuntimeError("Video-disjoint fold assignment produced an empty fold")
    by_video: dict[str, int] = {}
    for video_uid, fold in zip(video_uids, result.tolist()):
        previous = by_video.setdefault(video_uid, fold)
        if previous != fold:
            raise RuntimeError(f"Video {video_uid} leaked across folds")
    return result


def _dense_labels(index: pd.DataFrame, mapping: Any) -> dict[str, torch.Tensor]:
    verb = torch.tensor(
        [mapping.encode_verb(int(value)) for value in index["verb_label"]], dtype=torch.long
    )
    noun = torch.tensor(
        [mapping.encode_noun(int(value)) for value in index["noun_label"]], dtype=torch.long
    )
    action = torch.tensor(
        [
            mapping.encode_action(int(verb_raw), int(noun_raw))
            for verb_raw, noun_raw in zip(index["verb_label"], index["noun_label"])
        ],
        dtype=torch.long,
    )
    if "action_label" in index and not torch.equal(
        action, torch.tensor(index["action_label"].astype(int).tolist(), dtype=torch.long)
    ):
        raise RuntimeError("Index action_label disagrees with action_registry.json")
    return {"verb": verb, "noun": noun, "action": action}


def _metric_bundle(scores: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    return {
        "top1": top_k_recall(scores, labels, k=1),
        "top5": top_k_recall(scores, labels, k=5),
        "top10": top_k_recall(scores, labels, k=10),
        "top15": top_k_recall(scores, labels, k=15),
        "top20": top_k_recall(scores, labels, k=20),
        "cmr5": class_mean_recall(scores, labels, num_classes, k=5),
    }


def _topk_hits(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    k = min(k, scores.shape[-1])
    return (scores.topk(k, dim=-1).indices == labels[:, None]).any(dim=1)


def _paired_delta(
    challenger: torch.Tensor,
    baseline: torch.Tensor,
    labels: torch.Tensor,
    video_uids: Sequence[str],
    *,
    seed: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    challenger_hit = _topk_hits(challenger, labels, k=5)
    baseline_hit = _topk_hits(baseline, labels, k=5)
    paired = challenger_hit.float() - baseline_hit.float()
    n = len(paired)
    delta = 100.0 * paired.mean().item()
    standard_error = 100.0 * paired.double().std(unbiased=True).item() / math.sqrt(max(1, n))

    by_video: dict[str, list[int]] = {}
    for position, video_uid in enumerate(video_uids):
        by_video.setdefault(video_uid, []).append(position)
    videos = sorted(by_video)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    paired_np = paired.numpy()
    for iteration in range(bootstrap_samples):
        drawn = rng.integers(0, len(videos), size=len(videos))
        positions = [position for item in drawn for position in by_video[videos[int(item)]]]
        bootstrap[iteration] = 100.0 * float(paired_np[positions].mean())

    return {
        "n": n,
        "challenger_only_correct": int((challenger_hit & ~baseline_hit).sum()),
        "baseline_only_correct": int((baseline_hit & ~challenger_hit).sum()),
        "both_correct": int((challenger_hit & baseline_hit).sum()),
        "neither_correct": int((~challenger_hit & ~baseline_hit).sum()),
        "delta_top5_pp": delta,
        "normal_95ci_pp": [delta - 1.96 * standard_error, delta + 1.96 * standard_error],
        "video_bootstrap_95ci_pp": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "video_bootstrap_probability_positive": float((bootstrap > 0).mean()),
        "bootstrap_samples": bootstrap_samples,
    }


def _validate_next_indices(
    train_index: pd.DataFrame,
    val_index: pd.DataFrame,
    *,
    expected_train: int,
    expected_val: int,
) -> None:
    required = {
        "video_uid",
        "cache_sample_id",
        "verb_label",
        "noun_label",
        "action_label",
        "observed_action_label",
        "observed_action_end_sec",
        "target_start_sec",
        "annotation_level",
    }
    for split, frame, expected in (
        ("train", train_index, expected_train),
        ("val", val_index, expected_val),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{split} next-action index misses columns: {sorted(missing)}")
        if len(frame) != expected:
            raise RuntimeError(f"{split} next-action size is {len(frame)}, expected {expected}")
        if frame["cache_sample_id"].duplicated().any():
            raise RuntimeError(f"{split} next-action cache_sample_id is not unique")
        if (frame["target_start_sec"] < frame["observed_action_end_sec"] - 1e-6).any():
            raise RuntimeError(f"{split} includes a non-strict future target")
    overlap = set(train_index["video_uid"].astype(str)) & set(val_index["video_uid"].astype(str))
    if overlap:
        raise RuntimeError(f"Train/val video leakage: {len(overlap)} shared videos")


def _audit_cache(cache_dir: Path, split: str, sample_ids: Sequence[str]) -> None:
    directory = cache_dir / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing cache directory: {directory}")
    available = {path.stem for path in directory.glob("*.pt")}
    missing = [sample_id for sample_id in sample_ids if sample_id not in available]
    if missing:
        raise RuntimeError(
            f"Cache/index mismatch under {directory}: missing={len(missing)} first={missing[0]}"
        )


def _load_probe(
    spec: CheckpointSpec,
    *,
    embed_dim: int,
    num_classes: dict[str, int],
    device: torch.device,
) -> tuple[AnticipationHead, dict[str, Any]]:
    config = load_config(spec.config_path)
    classifier = get(config, "model.classifier", {})
    if bool(classifier.get("use_temporal_metadata", False)):
        raise RuntimeError(f"Phase-0 fixed-window checkpoint unexpectedly needs temporal metadata: {spec.path}")
    model = AnticipationHead(
        num_verb_classes=num_classes["verb"],
        num_noun_classes=num_classes["noun"],
        num_action_classes=num_classes["action"],
        embed_dim=embed_dim,
        num_heads=int(classifier.get("num_heads", 16)),
        depth=int(classifier.get("num_probe_blocks", 4)),
        repository_dir=get(config, "model.repository_dir"),
    )
    checkpoint = _load_torch(spec.path, mmap=True)
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != spec.expected_epoch:
        raise RuntimeError(f"{spec.path} says epoch={epoch}; expected {spec.expected_epoch}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    metadata = {
        **_checkpoint_signature(spec),
        "checkpoint_epoch": epoch,
        "checkpoint_metric": checkpoint.get("metric"),
        "checkpoint_metric_name": checkpoint.get("metric_name"),
    }
    del checkpoint
    return model.to(device).eval(), metadata


@torch.inference_mode()
def _infer_one_cache_pass(
    *,
    cache_dir: Path,
    source_sample_ids: Sequence[str],
    logical_sample_ids: Sequence[str],
    labels: dict[str, torch.Tensor],
    video_uids: Sequence[str],
    specs: Sequence[CheckpointSpec],
    num_classes: dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    contract: str,
) -> dict[str, Any]:
    if len(source_sample_ids) != len(logical_sample_ids):
        raise ValueError("source and logical sample ID lengths differ")
    dataset = FeatureCacheDataset(list(source_sample_ids), cache_dir)
    if dataset.sample_ids != list(source_sample_ids):
        raise RuntimeError("FeatureCacheDataset dropped or reordered requested samples")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=anticipation_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    embed_dim = int(dataset[0]["video"].shape[-1])
    models: dict[str, AnticipationHead] = {}
    candidate_metadata: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.name in models:
            raise ValueError(f"Duplicate checkpoint name: {spec.name}")
        model, metadata = _load_probe(
            spec, embed_dim=embed_dim, num_classes=num_classes, device=device
        )
        models[spec.name] = model
        candidate_metadata[spec.name] = metadata

    chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {head: [] for head in HEADS} for name in models
    }
    seen_source_ids: list[str] = []
    started = time.time()
    for batch_number, batch in enumerate(loader, start=1):
        features = batch["video"].to(device, non_blocking=True)
        for name, model in models.items():
            output = model(features)
            for head in HEADS:
                chunks[name][head].append(output[head].float().cpu())
        seen_source_ids.extend(batch["sample_id"])
        if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(loader):
            done = min(batch_number * batch_size, len(dataset))
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[{contract}] cache pass {done}/{len(dataset)} "
                f"({done / elapsed:.1f} samples/s, {len(models)} checkpoints)",
                flush=True,
            )
    if seen_source_ids != list(source_sample_ids):
        raise RuntimeError("DataLoader output order differs from requested cache order")

    candidates = {
        name: {
            "metadata": candidate_metadata[name],
            "logits": {head: torch.cat(chunks[name][head], dim=0) for head in HEADS},
        }
        for name in models
    }
    return {
        "format_version": FORMAT_VERSION,
        "contract": contract,
        "created_unix": time.time(),
        "logical_sample_ids": list(logical_sample_ids),
        "source_cache_sample_ids": list(source_sample_ids),
        "video_uids": list(video_uids),
        "labels": labels,
        "num_classes": num_classes,
        "candidates": candidates,
    }


def _validate_artifact(
    artifact: dict[str, Any],
    *,
    contract: str,
    logical_sample_ids: Sequence[str],
    source_sample_ids: Sequence[str],
    specs: Sequence[CheckpointSpec],
) -> None:
    if artifact.get("format_version") != FORMAT_VERSION or artifact.get("contract") != contract:
        raise RuntimeError(f"Incompatible logits artifact for {contract}")
    if artifact.get("logical_sample_ids") != list(logical_sample_ids):
        raise RuntimeError(f"Logical sample IDs changed for cached {contract} logits")
    if artifact.get("source_cache_sample_ids") != list(source_sample_ids):
        raise RuntimeError(f"Source cache IDs changed for cached {contract} logits")
    expected = {spec.name: _checkpoint_signature(spec) for spec in specs}
    candidates = artifact.get("candidates", {})
    if set(candidates) != set(expected):
        raise RuntimeError(
            f"Checkpoint set changed for {contract}; use --force-inference to replace the artifact"
        )
    for spec in specs:
        signature = expected[spec.name]
        metadata = candidates[spec.name]["metadata"]
        actual = {key: metadata.get(key) for key in signature}
        if actual == signature:
            continue
        # A gate launched during the final provenance review may contain the
        # epoch_03.pt pathname.  Accept it only when it is an exact tensor-wise
        # alias of the audited best.pt epoch-3 checkpoint.  This preserves
        # reusable logits without weakening checks for any other candidate.
        actual_path = Path(str(metadata.get("path", "")))
        alias_ok = (
            spec.name == "next_ep03"
            and int(metadata.get("checkpoint_epoch", -1)) == 3
            and actual_path.is_file()
            and _checkpoint_model_states_equal(actual_path, spec.path)
        )
        if not alias_ok:
            raise RuntimeError(
                f"Checkpoint signature changed for {contract}/{spec.name}; "
                "use --force-inference to replace the artifact"
            )


def _get_or_infer(
    *,
    artifact_path: Path,
    force: bool,
    cache_dir: Path,
    source_sample_ids: Sequence[str],
    logical_sample_ids: Sequence[str],
    labels: dict[str, torch.Tensor],
    video_uids: Sequence[str],
    specs: Sequence[CheckpointSpec],
    num_classes: dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    contract: str,
) -> dict[str, Any]:
    if artifact_path.is_file() and not force:
        print(f"[{contract}] validating and reusing {artifact_path}", flush=True)
        artifact = _load_torch(artifact_path, mmap=True)
        _validate_artifact(
            artifact,
            contract=contract,
            logical_sample_ids=logical_sample_ids,
            source_sample_ids=source_sample_ids,
            specs=specs,
        )
        return artifact
    artifact = _infer_one_cache_pass(
        cache_dir=cache_dir,
        source_sample_ids=source_sample_ids,
        logical_sample_ids=logical_sample_ids,
        labels=labels,
        video_uids=video_uids,
        specs=specs,
        num_classes=num_classes,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        contract=contract,
    )
    _atomic_torch_save(artifact, artifact_path)
    print(f"[{contract}] wrote {artifact_path}", flush=True)
    return artifact


def _transition_components(
    train_index: pd.DataFrame, num_actions: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observed = torch.tensor(train_index["observed_action_label"].astype(int).tolist())
    target = torch.tensor(train_index["action_label"].astype(int).tolist())
    if observed.min() < 0 or observed.max() >= num_actions or target.min() < 0 or target.max() >= num_actions:
        raise RuntimeError("Transition labels fall outside the action registry")
    counts = torch.zeros((num_actions, num_actions), dtype=torch.float64)
    flat = observed * num_actions + target
    counts.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
    row_counts = counts.sum(dim=1)
    global_counts = counts.sum(dim=0)
    global_prior = global_counts / global_counts.sum().clamp_min(1.0)
    return counts, row_counts, global_prior


def _transition_matrix(
    counts: torch.Tensor,
    row_counts: torch.Tensor,
    global_prior: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    numerator = counts + float(alpha) * global_prior[None, :]
    denominator = row_counts[:, None] + float(alpha)
    matrix = numerator / denominator.clamp_min(1e-12)
    zero_rows = row_counts == 0
    if zero_rows.any():
        matrix[zero_rows] = global_prior
    if not torch.allclose(matrix.sum(dim=1), torch.ones_like(row_counts), atol=1e-8):
        raise RuntimeError("Transition rows do not sum to one")
    return matrix.float()


def _transition_mixture(
    recognition_logits: torch.Tensor,
    transition: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    recognition_prob = torch.softmax(recognition_logits.to(device) / float(temperature), dim=-1)
    return (recognition_prob @ transition.to(device)).float().cpu()


def _choose_transition_hyperparameters(
    recognition_logits: torch.Tensor,
    labels: torch.Tensor,
    folds: torch.Tensor,
    counts: torch.Tensor,
    row_counts: torch.Tensor,
    global_prior: torch.Tensor,
    temperatures: Sequence[float],
    alphas: Sequence[float],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    best: dict[int, tuple[tuple[float, float, float, float], float, float]] = {}
    transitions = {
        alpha: _transition_matrix(counts, row_counts, global_prior, alpha) for alpha in alphas
    }
    for temperature in temperatures:
        for alpha in alphas:
            scores = _transition_mixture(
                recognition_logits, transitions[alpha], temperature, device
            )
            for tune_fold in (0, 1):
                mask = folds == tune_fold
                metrics = _metric_bundle(scores[mask], labels[mask], scores.shape[-1])
                # Primary objective Top-5; CMR@5 is a deterministic tie-breaker.
                key = (
                    metrics["top5"],
                    metrics["cmr5"],
                    -abs(float(temperature) - 1.0),
                    -float(alpha),
                )
                if tune_fold not in best or key > best[tune_fold][0]:
                    best[tune_fold] = (key, float(temperature), float(alpha))

    oof = torch.empty((len(labels), recognition_logits.shape[-1]), dtype=torch.float32)
    selections: list[dict[str, Any]] = []
    for test_fold in (0, 1):
        tune_fold = 1 - test_fold
        _, temperature, alpha = best[tune_fold]
        scores = _transition_mixture(
            recognition_logits, transitions[alpha], temperature, device
        )
        test_mask = folds == test_fold
        tune_mask = folds == tune_fold
        oof[test_mask] = scores[test_mask]
        selections.append(
            {
                "test_fold": test_fold,
                "tune_fold": tune_fold,
                "temperature": temperature,
                "alpha": alpha,
                "tune_size": int(tune_mask.sum()),
                "test_size": int(test_mask.sum()),
                "tune_metrics": _metric_bundle(
                    scores[tune_mask], labels[tune_mask], scores.shape[-1]
                ),
                "test_metrics": _metric_bundle(
                    scores[test_mask], labels[test_mask], scores.shape[-1]
                ),
            }
        )
    return oof, selections


def run_p0b(
    *,
    endpoint_artifact: dict[str, Any],
    train_index: pd.DataFrame,
    val_index: pd.DataFrame,
    folds: torch.Tensor,
    output_dir: Path,
    temperatures: Sequence[float],
    alphas: Sequence[float],
    tuning_device: torch.device,
    gate_threshold: float,
    seed: int,
) -> dict[str, Any]:
    print("[P0-b] evaluating soft transition mixture first", flush=True)
    recognition_logits = endpoint_artifact["candidates"]["recognition_ep15"]["logits"]["action"]
    direct_logits = endpoint_artifact["candidates"]["next_ep03"]["logits"]["action"]
    labels = endpoint_artifact["labels"]["action"]
    num_actions = int(endpoint_artifact["num_classes"]["action"])
    counts, row_counts, global_prior = _transition_components(train_index, num_actions)
    oof_scores, selections = _choose_transition_hyperparameters(
        recognition_logits,
        labels,
        folds,
        counts,
        row_counts,
        global_prior,
        temperatures,
        alphas,
        tuning_device,
    )

    unsmoothed = _transition_matrix(counts, row_counts, global_prior, alpha=0.0)
    observed_labels = torch.tensor(val_index["observed_action_label"].astype(int).tolist())
    oracle_scores = unsmoothed[observed_labels]
    prior_scores = global_prior.float().repeat(len(labels), 1)
    oof_metrics = _metric_bundle(oof_scores, labels, num_actions)
    direct_metrics = _metric_bundle(direct_logits, labels, num_actions)
    paired = _paired_delta(
        oof_scores,
        direct_logits,
        labels,
        endpoint_artifact["video_uids"],
        seed=seed,
    )
    gate_pass = bool(oof_metrics["top5"] >= gate_threshold)
    results = {
        "phase": "P0-b",
        "contract": "A2.end-1s visual evidence -> strict same-level A3",
        "sample_count": len(labels),
        "train_transition_rows": len(train_index),
        "transition_source": "train index only; validation labels never enter T",
        "folding": "two folds, deterministic and video-disjoint",
        "hyperparameter_selection": selections,
        "temperature_grid": list(temperatures),
        "alpha_grid": list(alphas),
        "soft_mixture_oof": oof_metrics,
        "direct_next_epoch3_baseline": direct_metrics,
        "gt_observed_action_oracle_unsmoothed_train_T": _metric_bundle(
            oracle_scores, labels, num_actions
        ),
        "global_train_target_prior": _metric_bundle(prior_scores, labels, num_actions),
        "paired_vs_direct_epoch3": paired,
        "gate": {
            "metric": "Action OOF instance Top-5 accuracy",
            "operator": ">=",
            "threshold_percent": gate_threshold,
            "observed_percent": oof_metrics["top5"],
            "passed": gate_pass,
            "note": "Paired delta is reported but is not an extra unregistered gate condition.",
        },
    }
    _atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "sample_ids": endpoint_artifact["logical_sample_ids"],
            "video_uids": endpoint_artifact["video_uids"],
            "folds": folds,
            "labels": labels,
            "oof_transition_scores": oof_scores,
            "direct_epoch3_logits": direct_logits,
            "oracle_scores": oracle_scores,
            "selections": selections,
        },
        output_dir / "p0b_oof_scores.pt",
    )
    _write_json(output_dir / "p0b_results.json", results)
    print(
        f"[P0-b] OOF Action Top-5={oof_metrics['top5']:.3f}; "
        f"direct={direct_metrics['top5']:.3f}; delta={paired['delta_top5_pp']:+.3f} pp; "
        f"gate={'PASS' if gate_pass else 'FAIL'}",
        flush=True,
    )
    return results


def _objective(scores: torch.Tensor, labels: torch.Tensor, num_classes: int, name: str) -> float:
    if name == "top5":
        return top_k_recall(scores, labels, k=5)
    if name == "cmr5":
        return class_mean_recall(scores, labels, num_classes, k=5)
    raise ValueError(f"Unknown ensemble objective: {name}")


def _caruana_select(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    candidate_names: Sequence[str],
    *,
    num_classes: int,
    rounds: int,
    objective: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    if probabilities.ndim != 3 or probabilities.shape[0] != len(candidate_names):
        raise ValueError("Expected candidate probabilities [models, samples, classes]")
    running_sum = torch.zeros_like(probabilities[0])
    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    best_prefix: list[int] = []
    best_prefix_value = float("-inf")
    for round_index in range(rounds):
        choices: list[tuple[float, int]] = []
        for candidate_index in range(len(candidate_names)):
            trial = (running_sum + probabilities[candidate_index]) / (len(selected) + 1)
            choices.append(
                (
                    _objective(trial, labels, num_classes, objective),
                    candidate_index,
                )
            )
        value, chosen = max(choices, key=lambda item: (item[0], -item[1]))
        selected.append(chosen)
        running_sum += probabilities[chosen]
        trace.append(
            {
                "round": round_index + 1,
                "candidate": candidate_names[chosen],
                "objective": value,
            }
        )
        if value > best_prefix_value + 1e-12:
            best_prefix_value = value
            best_prefix = selected.copy()
    if not best_prefix:
        raise RuntimeError("Caruana selection produced no candidates")
    return best_prefix, trace


def _fieldwise_oof_ensemble(
    candidate_logits: dict[str, dict[str, torch.Tensor]],
    labels: dict[str, torch.Tensor],
    num_classes: dict[str, int],
    folds: torch.Tensor,
    *,
    rounds: int,
    objective: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    candidate_names = list(candidate_logits)
    if not candidate_names:
        raise ValueError("No ensemble candidates")
    outputs: dict[str, torch.Tensor] = {}
    selections: dict[str, Any] = {}
    for head in HEADS:
        probabilities = torch.stack(
            [torch.softmax(candidate_logits[name][head], dim=-1) for name in candidate_names]
        )
        output = torch.empty_like(probabilities[0])
        selections[head] = []
        for test_fold in (0, 1):
            tune_fold = 1 - test_fold
            tune_mask = folds == tune_fold
            test_mask = folds == test_fold
            selected, trace = _caruana_select(
                probabilities[:, tune_mask],
                labels[head][tune_mask],
                candidate_names,
                num_classes=num_classes[head],
                rounds=rounds,
                objective=objective,
            )
            output[test_mask] = probabilities[selected][:, test_mask].mean(dim=0)
            selections[head].append(
                {
                    "test_fold": test_fold,
                    "tune_fold": tune_fold,
                    "selected_with_replacement": [candidate_names[index] for index in selected],
                    "selection_counts": dict(Counter(candidate_names[index] for index in selected)),
                    "best_prefix_length": len(selected),
                    "trace": trace,
                    "tune_metrics": _metric_bundle(
                        probabilities[selected][:, tune_mask].mean(dim=0),
                        labels[head][tune_mask],
                        num_classes[head],
                    ),
                    "test_metrics": _metric_bundle(
                        output[test_mask], labels[head][test_mask], num_classes[head]
                    ),
                }
            )
        outputs[head] = output
    return outputs, selections


def run_p0a(
    *,
    candidate_logits: dict[str, dict[str, torch.Tensor]],
    labels: dict[str, torch.Tensor],
    video_uids: Sequence[str],
    sample_ids: Sequence[str],
    num_classes: dict[str, int],
    folds: torch.Tensor,
    output_dir: Path,
    name: str,
    contract: str,
    deployable: bool,
    rounds: int,
    objective: str,
    seed: int,
) -> dict[str, Any]:
    outputs, selections = _fieldwise_oof_ensemble(
        candidate_logits,
        labels,
        num_classes,
        folds,
        rounds=rounds,
        objective=objective,
    )
    metrics = {
        head: _metric_bundle(outputs[head], labels[head], num_classes[head]) for head in HEADS
    }
    individual = {
        candidate: {
            head: _metric_bundle(logits[head], labels[head], num_classes[head])
            for head in HEADS
        }
        for candidate, logits in candidate_logits.items()
    }
    baseline_name = "next_ep03"
    paired = None
    if baseline_name in candidate_logits:
        paired = _paired_delta(
            outputs["action"],
            candidate_logits[baseline_name]["action"],
            labels["action"],
            video_uids,
            seed=seed,
        )
    results = {
        "phase": "P0-a",
        "name": name,
        "contract": contract,
        "deployable_at_A2_boundary": deployable,
        "sample_count": len(sample_ids),
        "candidate_count": len(candidate_logits),
        "candidate_names": list(candidate_logits),
        "selection": {
            "method": "Caruana greedy ensemble selection with replacement; best prefix",
            "objective": objective,
            "max_rounds": rounds,
            "folding": "two folds, deterministic and video-disjoint",
            "fieldwise": selections,
        },
        "oof_fieldwise_ensemble": metrics,
        "individual_candidates": individual,
        "paired_action_top5_vs_next_ep03": paired,
    }
    stem = f"p0a_{name}"
    _atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "sample_ids": list(sample_ids),
            "video_uids": list(video_uids),
            "folds": folds,
            "labels": labels,
            "oof_scores": outputs,
            "selections": selections,
            "contract": contract,
            "deployable_at_A2_boundary": deployable,
        },
        output_dir / f"{stem}_oof_scores.pt",
    )
    _write_json(output_dir / f"{stem}_results.json", results)
    print(
        f"[P0-a:{name}] Action OOF Top-5={metrics['action']['top5']:.3f}; "
        f"CMR@5={metrics['action']['cmr5']:.3f}",
        flush=True,
    )
    return results


def _checkpoint_specs(
    *, prefix: str, checkpoint_dir: Path, config_path: Path, epochs: Iterable[int]
) -> list[CheckpointSpec]:
    return [
        CheckpointSpec(
            name=f"{prefix}_ep{epoch:02d}",
            path=checkpoint_dir / f"epoch_{epoch:02d}.pt",
            expected_epoch=epoch,
            config_path=config_path,
        )
        for epoch in epochs
    ]


def _reconstruct_later_anchor_ids(
    endpoint_index: pd.DataFrame,
    next_index: pd.DataFrame,
    start_index: pd.DataFrame,
) -> list[str]:
    """Map each observed endpoint cache ID to its target A3 start-cache ID.

    This replays the exact next-action builder selection rather than relying on
    fuzzy timestamp joins.  It also proves that endpoint and start indices use
    the same canonical row identities before any later-anchor cache is read.
    """
    endpoint = endpoint_index.reset_index(drop=True).copy()
    endpoint["_cache_id"] = [
        f"{clip_uid}_{position}"
        for position, clip_uid in enumerate(endpoint["clip_uid"].astype(str))
    ]
    if len(endpoint) != len(start_index):
        raise RuntimeError("Endpoint and start indices differ in row count")
    for column in ("video_uid", "clip_uid", "verb_label", "noun_label", "action_label"):
        left = endpoint[column].astype(str).tolist()
        right = start_index[column].astype(str).tolist()
        if left != right:
            raise RuntimeError(f"Endpoint/start canonical row alignment failed for {column}")

    target_by_observed: dict[str, str] = {}
    row_by_observed = next_index.set_index("cache_sample_id", drop=False)
    for _, video in endpoint.groupby("video_uid", sort=False):
        ordered = video.sort_values(
            ["target_start_sec", "target_end_sec", "matched_level", "_cache_id"], kind="stable"
        )
        for _, observed in ordered.iterrows():
            observed_id = str(observed["_cache_id"])
            if observed_id not in row_by_observed.index:
                continue
            candidates = ordered[
                (ordered["matched_level"] == observed["matched_level"])
                & (ordered["target_start_sec"] >= float(observed["target_end_sec"]) - 1e-6)
            ]
            if candidates.empty:
                raise RuntimeError(f"Could not replay target selection for {observed_id}")
            target = candidates.iloc[0]
            audited = row_by_observed.loc[observed_id]
            if isinstance(audited, pd.DataFrame):
                raise RuntimeError(f"Duplicate next-action cache ID: {observed_id}")
            checks = (
                abs(float(target["target_start_sec"]) - float(audited["target_start_sec"])) < 1e-5,
                abs(float(target["target_end_sec"]) - float(audited["target_end_sec"])) < 1e-5,
                int(target["verb_label"]) == int(audited["verb_label"]),
                int(target["noun_label"]) == int(audited["noun_label"]),
                str(target["matched_level"]) == str(audited["annotation_level"]),
            )
            if not all(checks):
                raise RuntimeError(f"Reconstructed A3 differs from audited target for {observed_id}")
            target_by_observed[observed_id] = str(target["_cache_id"])
    logical_ids = next_index["cache_sample_id"].astype(str).tolist()
    if set(target_by_observed) != set(logical_ids):
        raise RuntimeError("Later-anchor reconstruction did not cover every next-action row")
    return [target_by_observed[sample_id] for sample_id in logical_ids]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=("audit", "gate", "primary", "mixed", "all"),
        default="gate",
        help="gate=P0-b first; primary=P0-b then 8-epoch same-contract P0-a; mixed adds the optional benchmark",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--next-index-dir", default=DEFAULT_NEXT_INDEX)
    parser.add_argument("--endpoint-index-dir", default=DEFAULT_ENDPOINT_INDEX)
    parser.add_argument("--endpoint-cache-dir", default=DEFAULT_ENDPOINT_CACHE)
    parser.add_argument("--recognition-config", default=DEFAULT_RECOGNITION_CONFIG)
    parser.add_argument("--recognition-checkpoint", default=DEFAULT_RECOGNITION_CHECKPOINT)
    parser.add_argument("--next-config", default=DEFAULT_ENDPOINT_CONFIG)
    parser.add_argument("--next-checkpoint-dir", default=DEFAULT_NEXT_CHECKPOINT_DIR)
    parser.add_argument(
        "--next-best-checkpoint",
        default=DEFAULT_NEXT_BEST_CHECKPOINT,
        help="Audited direct baseline (best.pt, required to encode epoch 3)",
    )
    parser.add_argument("--start8-index-dir", default=DEFAULT_START8_INDEX)
    parser.add_argument("--start8-cache-dir", default=DEFAULT_START8_CACHE)
    parser.add_argument("--start8-config", default=DEFAULT_START8_CONFIG)
    parser.add_argument("--start8-checkpoint-dir", default=DEFAULT_START8_CHECKPOINT_DIR)
    parser.add_argument("--start16-index-dir", default=DEFAULT_START16_INDEX)
    parser.add_argument("--start16-cache-dir", default=DEFAULT_START16_CACHE)
    parser.add_argument("--start16-config", default=DEFAULT_START16_CONFIG)
    parser.add_argument("--start16-checkpoint-dir", default=DEFAULT_START16_CHECKPOINT_DIR)
    parser.add_argument("--expected-train", type=int, default=29293)
    parser.add_argument("--expected-val", type=int, default=6960)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperatures", default="0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--alphas", default="0,0.1,1,3,10,30,100")
    parser.add_argument("--gate-threshold", type=float, default=27.7)
    parser.add_argument("--ensemble-rounds", type=int, default=16)
    parser.add_argument("--ensemble-objective", choices=("top5", "cmr5"), default="top5")
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Replace logits artifacts even when valid reusable artifacts exist",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.ensemble_rounds <= 0:
        raise ValueError("batch-size and ensemble-rounds must be >0; num-workers must be >=0")
    temperatures = _parse_grid(args.temperatures, name="temperature", positive=True)
    alphas = _parse_grid(args.alphas, name="alpha", positive=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    next_index_dir = _path(args.next_index_dir)
    endpoint_index_dir = _path(args.endpoint_index_dir)
    endpoint_cache_dir = _path(args.endpoint_cache_dir)
    train_index = _read_index(next_index_dir, "train")
    val_index = _read_index(next_index_dir, "val")
    _validate_next_indices(
        train_index,
        val_index,
        expected_train=args.expected_train,
        expected_val=args.expected_val,
    )
    mapping = tz1._load_registry(next_index_dir / "action_registry.json")
    num_classes = {
        "verb": mapping.num_verbs,
        "noun": mapping.num_nouns,
        "action": mapping.num_actions,
    }
    labels = _dense_labels(val_index, mapping)
    logical_ids = val_index["cache_sample_id"].astype(str).tolist()
    video_uids = val_index["video_uid"].astype(str).tolist()
    folds = _stable_video_folds(video_uids, args.seed)
    _audit_cache(endpoint_cache_dir, "val", logical_ids)

    recognition_spec = CheckpointSpec(
        "recognition_ep15",
        _path(args.recognition_checkpoint),
        15,
        _path(args.recognition_config),
    )
    next_specs = _checkpoint_specs(
        prefix="next",
        checkpoint_dir=_path(args.next_checkpoint_dir),
        config_path=_path(args.next_config),
        epochs=range(1, 9),
    )
    # The gate's direct baseline is the run's audited best.pt (epoch 3), not
    # merely an assumed filename in checkpoints/.  It also occupies the epoch
    # 3 slot of the eight-candidate primary ensemble, avoiding duplicate
    # inference while preserving the exact epoch 1..8 candidate grid.
    next_specs[2] = CheckpointSpec(
        name="next_ep03",
        path=_path(args.next_best_checkpoint),
        expected_epoch=3,
        config_path=_path(args.next_config),
    )
    endpoint_specs = [recognition_spec, *next_specs]
    for spec in endpoint_specs:
        if not spec.path.is_file() or not spec.config_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint/config for {spec.name}: {spec}")

    audit = {
        "format_version": FORMAT_VERSION,
        "stage": args.stage,
        "device": str(device),
        "next_action_index": str(next_index_dir),
        "endpoint_cache": str(endpoint_cache_dir),
        "train_rows": len(train_index),
        "val_rows": len(val_index),
        "train_videos": int(train_index["video_uid"].nunique()),
        "val_videos": int(val_index["video_uid"].nunique()),
        "fold_sample_counts": {str(fold): int((folds == fold).sum()) for fold in (0, 1)},
        "fold_video_counts": {
            str(fold): len(set(val_index.loc[folds.numpy() == fold, "video_uid"].astype(str)))
            for fold in (0, 1)
        },
        "taxonomy": num_classes,
        "endpoint_checkpoints": [_checkpoint_signature(spec) for spec in endpoint_specs],
        "contract": "A2.end-1s -> strict same-level A3; unique key is observed cache_sample_id",
    }
    _write_json(output_dir / "audit.json", audit)
    print(
        f"[audit] train={len(train_index)} val={len(val_index)}; "
        f"val videos={val_index['video_uid'].nunique()}; folds={audit['fold_sample_counts']}",
        flush=True,
    )
    if args.stage == "audit":
        return

    endpoint_artifact = _get_or_infer(
        artifact_path=output_dir / "endpoint_logits.pt",
        force=args.force_inference,
        cache_dir=endpoint_cache_dir / "val",
        source_sample_ids=logical_ids,
        logical_sample_ids=logical_ids,
        labels=labels,
        video_uids=video_uids,
        specs=endpoint_specs,
        num_classes=num_classes,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        contract="endpoint_A2_end_minus_1s",
    )

    # Pre-registered ordering: P0-b is always computed and written first.
    run_p0b(
        endpoint_artifact=endpoint_artifact,
        train_index=train_index,
        val_index=val_index,
        folds=folds,
        output_dir=output_dir,
        temperatures=temperatures,
        alphas=alphas,
        tuning_device=device,
        gate_threshold=args.gate_threshold,
        seed=args.seed,
    )
    if args.stage == "gate":
        return

    endpoint_candidate_logits = {
        name: record["logits"]
        for name, record in endpoint_artifact["candidates"].items()
        if name.startswith("next_ep")
    }
    run_p0a(
        candidate_logits=endpoint_candidate_logits,
        labels=labels,
        video_uids=video_uids,
        sample_ids=logical_ids,
        num_classes=num_classes,
        folds=folds,
        output_dir=output_dir,
        name="primary_same_decision",
        contract="All 8 candidates observe the identical A2.end-1s evidence and predict A3",
        deployable=True,
        rounds=args.ensemble_rounds,
        objective=args.ensemble_objective,
        seed=args.seed,
    )
    if args.stage == "primary":
        return

    endpoint_index = _read_index(endpoint_index_dir, "val")
    start_artifacts: dict[str, dict[str, Any]] = {}
    for label, index_arg, cache_arg, config_arg, checkpoint_arg in (
        (
            "later_start8",
            args.start8_index_dir,
            args.start8_cache_dir,
            args.start8_config,
            args.start8_checkpoint_dir,
        ),
        (
            "later_start16",
            args.start16_index_dir,
            args.start16_cache_dir,
            args.start16_config,
            args.start16_checkpoint_dir,
        ),
    ):
        start_index_dir = _path(index_arg)
        start_index = _read_index(start_index_dir, "val")
        source_ids = _reconstruct_later_anchor_ids(endpoint_index, val_index, start_index)
        cache_dir = _path(cache_arg)
        _audit_cache(cache_dir, "val", source_ids)
        specs = _checkpoint_specs(
            prefix=label,
            checkpoint_dir=_path(checkpoint_arg),
            config_path=_path(config_arg),
            epochs=range(1, 11),
        )
        for spec in specs:
            if not spec.path.is_file() or not spec.config_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint/config for {spec.name}: {spec}")
        start_artifacts[label] = _get_or_infer(
            artifact_path=output_dir / f"{label}_logits.pt",
            force=args.force_inference,
            cache_dir=cache_dir / "val",
            source_sample_ids=source_ids,
            logical_sample_ids=logical_ids,
            labels=labels,
            video_uids=video_uids,
            specs=specs,
            num_classes=num_classes,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            contract=f"{label}_oracle_A3_start_boundary",
        )

    mixed_logits = dict(endpoint_candidate_logits)
    for artifact in start_artifacts.values():
        for name, record in artifact["candidates"].items():
            if name in mixed_logits:
                raise RuntimeError(f"Duplicate mixed candidate name: {name}")
            mixed_logits[name] = record["logits"]
    if len(mixed_logits) != 28:
        raise RuntimeError(f"Mixed benchmark expected 28 candidates, got {len(mixed_logits)}")
    run_p0a(
        candidate_logits=mixed_logits,
        labels=labels,
        video_uids=video_uids,
        sample_ids=logical_ids,
        num_classes=num_classes,
        folds=folds,
        output_dir=output_dir,
        name="mixed_later_anchor_benchmark",
        contract=(
            "8 A2.end-1s candidates plus 20 probes evaluated at reconstructed A3.start-1s; "
            "the latter require oracle future boundary knowledge"
        ),
        deployable=False,
        rounds=args.ensemble_rounds,
        objective=args.ensemble_objective,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
