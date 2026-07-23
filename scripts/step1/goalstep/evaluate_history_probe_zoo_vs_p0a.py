#!/usr/bin/env python3
"""Leakage-safe outer-video-fold selection for the GoalStep Phase-2 zoo.

The twelve registered history arms (the completed Phase-1 default plus the
eleven Phase-2 LR/WD arms) are never selected on the rows on which they are
reported.  For each field and held-out video fold, this evaluator uses only
the opposite fold to:

1. relearn the P0-a recipe from the eight raw endpoint checkpoints;
2. retain the best two epochs per Phase-2 arm (at most 24 candidates);
3. run a fixed 16-round Caruana selection with replacement, independently
   for ``fused`` and ``current_only``; and
4. choose the P0-a/Phase-2 probability-blend weight.

Every selected object is then applied unchanged to the held-out fold.  This
avoids both full-validation tuning and the reverse-OOF leak that would arise
from using stored P0-a OOF scores while tuning the final blend.

P0-a remains the foundation diagnostic, but it is not the promotion bar:
the incumbent entering Phase 2 is the Phase-1 ``final_blend`` OOF score.
Phase 2 is promoted only when its final blend has a positive paired Action
Top-5 delta over that incumbent and the video-cluster bootstrap 95% interval
also has a strictly positive lower bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ego.step1_action_anticipation.metrics import (  # noqa: E402
    class_mean_recall,
    top_k_recall,
)


HEADS = ("verb", "noun", "action")
MODES = ("visual", "history", "current_only", "fused")
SELECTED_MODES = ("fused", "current_only")
FORMAT_VERSION = 1
PREDICTION_KIND = "goalstep_history_context_val_predictions"
PHASE1_OOF_KIND = "goalstep_history_context_crossfit_oof_scores"
PHASE1_OOF_FILENAME = "history_context_vs_p0a_oof_scores.pt"
CONTRACT = "A2.end-1s -> strict same-level A3"
ENDPOINT_CONTRACT = "endpoint_A2_end_minus_1s"
P0A_CANDIDATES = tuple(f"next_ep{epoch:02d}" for epoch in range(1, 9))
REGISTERED_GRID = tuple(
    (learning_rate, weight_decay)
    for learning_rate in (1e-4, 3e-4, 1e-3)
    for weight_decay in (1e-5, 1e-4, 1e-3, 1e-2)
)
REGISTERED_DEFAULT = (3e-4, 1e-4)
CARUANA_ROUNDS = 16
PROVENANCE_KEYS = (
    "format_version",
    "kind",
    "contract",
    "feature_reextraction",
    "config",
    "phase1_config",
    "store_manifest",
    "indices",
    "default_phase1",
    "train_rows",
    "val_rows",
    "num_classes",
    "summary_shape",
    "max_history",
    "seed",
    "epochs",
    "registered_grid",
    "skipped_default_arm",
)


def _torch_load(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:  # pragma: no cover - older torch
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_tensor(
    value: Any, *, name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _validate_video_folds(folds: torch.Tensor, video_uids: Sequence[str]) -> None:
    if tuple(folds.shape) != (len(video_uids),) or folds.dtype != torch.int64:
        raise ValueError("P0-a folds must be int64[N]")
    if set(folds.tolist()) != {0, 1}:
        raise ValueError("P0-a must contain exactly two non-empty folds")
    by_video: dict[str, int] = {}
    for video_uid, fold in zip(video_uids, folds.tolist()):
        previous = by_video.setdefault(str(video_uid), int(fold))
        if previous != int(fold):
            raise ValueError(f"video {video_uid!r} crosses outer folds")


def _validate_probabilities(value: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} is not finite and non-negative")
    if not torch.allclose(
        value.sum(dim=-1),
        torch.ones(value.shape[0], dtype=value.dtype),
        atol=2e-5,
        rtol=2e-5,
    ):
        raise ValueError(f"{name} rows do not sum to one")


def _metric_block(
    scores: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, float]:
    return {
        "cmr@5": class_mean_recall(scores, labels, num_classes, k=5),
        "top1": top_k_recall(scores, labels, k=1),
        "top5": top_k_recall(scores, labels, k=5),
        "top10": top_k_recall(scores, labels, k=10),
        "top15": top_k_recall(scores, labels, k=15),
    }


def _all_metrics(
    scores: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    num_classes: dict[str, int],
) -> dict[str, dict[str, float]]:
    return {
        head: _metric_block(scores[head], labels[head], num_classes[head])
        for head in HEADS
    }


def _top5_hits(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    indices = scores.topk(min(5, scores.shape[-1]), dim=-1).indices
    return (indices == labels[:, None]).any(dim=1)


def _paired_video_bootstrap(
    challenger: torch.Tensor,
    baseline: torch.Tensor,
    labels: torch.Tensor,
    video_uids: Sequence[str],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    challenger_hit = _top5_hits(challenger, labels)
    baseline_hit = _top5_hits(baseline, labels)
    paired = challenger_hit.to(torch.int8) - baseline_hit.to(torch.int8)
    delta = 100.0 * float(paired.float().mean())
    standard_error = (
        100.0 * float(paired.double().std(unbiased=True)) / math.sqrt(len(paired))
        if len(paired) > 1
        else 0.0
    )
    positions_by_video: dict[str, list[int]] = {}
    for position, video_uid in enumerate(video_uids):
        positions_by_video.setdefault(str(video_uid), []).append(position)
    videos = sorted(positions_by_video)
    if not videos:
        raise ValueError("Cannot bootstrap an empty validation cohort")
    paired_np = paired.numpy().astype(np.float64, copy=False)
    sums = np.asarray(
        [paired_np[positions_by_video[video]].sum() for video in videos], dtype=np.float64
    )
    counts = np.asarray([len(positions_by_video[video]) for video in videos], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(videos), size=(bootstrap_samples, len(videos)))
    bootstrap = 100.0 * sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "n": len(paired),
        "videos": len(videos),
        "challenger_only_correct": int((challenger_hit & ~baseline_hit).sum()),
        "baseline_only_correct": int((baseline_hit & ~challenger_hit).sum()),
        "both_correct": int((challenger_hit & baseline_hit).sum()),
        "neither_correct": int((~challenger_hit & ~baseline_hit).sum()),
        "delta_top5_pp": delta,
        "normal_95ci_pp": [
            delta - 1.96 * standard_error,
            delta + 1.96 * standard_error,
        ],
        "video_bootstrap_95ci_pp": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "video_bootstrap_probability_positive": float((bootstrap > 0).mean()),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def _caruana_select(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    candidate_names: Sequence[str],
    *,
    num_classes: int,
    rounds: int = CARUANA_ROUNDS,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Exact Top-5 Caruana contract used by Phase-0 P0-a."""
    if probabilities.ndim != 3 or probabilities.shape[0] != len(candidate_names):
        raise ValueError("Expected candidate probabilities [models,samples,classes]")
    if rounds != CARUANA_ROUNDS:
        raise ValueError(f"Phase-2 Caruana rounds are fixed at {CARUANA_ROUNDS}")
    running_sum = torch.zeros_like(probabilities[0])
    selected: list[int] = []
    best_prefix: list[int] = []
    best_value = -math.inf
    trace: list[dict[str, Any]] = []
    for round_index in range(rounds):
        choices: list[tuple[float, int]] = []
        for candidate_index in range(len(candidate_names)):
            trial = (running_sum + probabilities[candidate_index]) / (len(selected) + 1)
            choices.append(
                (top_k_recall(trial, labels, k=5), candidate_index)
            )
        value, chosen = max(choices, key=lambda item: (item[0], -item[1]))
        selected.append(chosen)
        running_sum += probabilities[chosen]
        trace.append(
            {
                "round": round_index + 1,
                "candidate": str(candidate_names[chosen]),
                "tune_top5": value,
            }
        )
        if value > best_value + 1e-12:
            best_value = value
            best_prefix = selected.copy()
    if not best_prefix:
        raise RuntimeError("Caruana selection produced no candidates")
    return best_prefix, trace


def _load_base(
    endpoint_path: Path, p0a_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
]:
    endpoint = _torch_load(endpoint_path, mmap=True)
    p0a = _torch_load(p0a_path, mmap=True)
    if int(endpoint.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported endpoint artifact format")
    if endpoint.get("contract") != ENDPOINT_CONTRACT:
        raise ValueError("Endpoint artifact has the wrong decision-time contract")
    if int(p0a.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported P0-a artifact format")
    if p0a.get("deployable_at_A2_boundary") is not True:
        raise ValueError("P0-a artifact is not same-decision-time deployable")
    sample_ids = [str(value) for value in p0a.get("sample_ids", [])]
    video_uids = [str(value) for value in p0a.get("video_uids", [])]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("P0-a sample IDs must be non-empty and unique")
    if len(video_uids) != len(sample_ids):
        raise ValueError("P0-a sample/video lengths differ")
    n = len(sample_ids)
    folds = _require_tensor(
        p0a.get("folds"), name="p0a.folds", shape=(n,), dtype=torch.int64
    )
    _validate_video_folds(folds, video_uids)
    if endpoint.get("logical_sample_ids") != sample_ids:
        raise ValueError("Endpoint/P0-a sample order differs")
    if endpoint.get("source_cache_sample_ids") != sample_ids:
        raise ValueError("Endpoint source-cache IDs differ from logical IDs")
    if endpoint.get("video_uids") != video_uids:
        raise ValueError("Endpoint/P0-a video order differs")
    raw_classes = endpoint.get("num_classes", {})
    if set(raw_classes) != set(HEADS):
        raise ValueError("Endpoint taxonomy is incomplete")
    num_classes = {head: int(raw_classes[head]) for head in HEADS}
    labels: dict[str, torch.Tensor] = {}
    for head in HEADS:
        labels[head] = _require_tensor(
            p0a.get("labels", {}).get(head),
            name=f"p0a.labels.{head}",
            shape=(n,),
            dtype=torch.int64,
        )
        endpoint_labels = _require_tensor(
            endpoint.get("labels", {}).get(head),
            name=f"endpoint.labels.{head}",
            shape=(n,),
            dtype=torch.int64,
        )
        if not torch.equal(labels[head], endpoint_labels):
            raise ValueError(f"Endpoint/P0-a {head} labels differ")
        stored = _require_tensor(
            p0a.get("oof_scores", {}).get(head),
            name=f"p0a.oof_scores.{head}",
            shape=(n, num_classes[head]),
            dtype=torch.float32,
        )
        _validate_probabilities(stored, name=f"p0a.oof_scores.{head}")
    candidates = endpoint.get("candidates", {})
    observed = tuple(name for name in candidates if str(name).startswith("next_ep"))
    if observed != P0A_CANDIDATES:
        raise ValueError(
            f"Endpoint next-action candidates must be exactly {P0A_CANDIDATES}, got {observed}"
        )
    for name in P0A_CANDIDATES:
        logits = candidates[name].get("logits", {})
        if set(logits) != set(HEADS):
            raise ValueError(f"Endpoint {name} head set is invalid")
        for head in HEADS:
            _require_tensor(
                logits[head],
                name=f"endpoint.{name}.{head}",
                shape=(n, num_classes[head]),
                dtype=torch.float32,
            )
    return endpoint, p0a, sample_ids, video_uids, folds, labels, num_classes


def _load_phase1_incumbent(
    path: Path,
    *,
    sample_ids: list[str],
    video_uids: list[str],
    folds: torch.Tensor,
    labels: dict[str, torch.Tensor],
    num_classes: dict[str, int],
    p0a: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load the Phase-1 final-blend OOF champion with exact cohort checks."""
    artifact = _torch_load(path, mmap=True)
    if int(artifact.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported Phase-1 OOF artifact format")
    if artifact.get("kind") != PHASE1_OOF_KIND:
        raise ValueError("Unexpected Phase-1 OOF artifact kind")
    if artifact.get("contract") != CONTRACT:
        raise ValueError("Phase-1 incumbent has the wrong anticipation contract")
    if artifact.get("sample_ids") != sample_ids:
        raise ValueError("Phase-1 incumbent sample order mismatch")
    if artifact.get("video_uids") != video_uids:
        raise ValueError("Phase-1 incumbent video order mismatch")
    artifact_folds = _require_tensor(
        artifact.get("folds"),
        name="phase1_incumbent.folds",
        shape=(len(sample_ids),),
        dtype=torch.int64,
    )
    if not torch.equal(artifact_folds, folds):
        raise ValueError("Phase-1 incumbent outer folds differ from P0-a")
    artifact_classes = {
        head: int(artifact.get("num_classes", {}).get(head, -1)) for head in HEADS
    }
    if artifact_classes != num_classes:
        raise ValueError("Phase-1 incumbent taxonomy mismatch")
    if set(artifact.get("labels", {})) != set(HEADS):
        raise ValueError("Phase-1 incumbent label head set mismatch")
    for head in HEADS:
        value = _require_tensor(
            artifact["labels"][head],
            name=f"phase1_incumbent.labels.{head}",
            shape=(len(sample_ids),),
            dtype=torch.int64,
        )
        if not torch.equal(value, labels[head]):
            raise ValueError(f"Phase-1 incumbent {head} labels differ")
    expected_score_names = {
        "p0a",
        "phase1",
        "final_blend",
        "visual_same_epoch",
        "history_same_epoch",
        "current_only_same_epoch",
    }
    raw_scores = artifact.get("scores", {})
    if set(raw_scores) != expected_score_names:
        raise ValueError(
            "Phase-1 incumbent score set is incomplete or unexpected: "
            f"{sorted(raw_scores)}"
        )
    validated: dict[str, dict[str, torch.Tensor]] = {}
    for score_name in expected_score_names:
        if set(raw_scores[score_name]) != set(HEADS):
            raise ValueError(f"Phase-1 incumbent {score_name} head set mismatch")
        validated[score_name] = {}
        for head in HEADS:
            value = _require_tensor(
                raw_scores[score_name][head],
                name=f"phase1_incumbent.scores.{score_name}.{head}",
                shape=(len(sample_ids), num_classes[head]),
                dtype=torch.float32,
            )
            _validate_probabilities(
                value, name=f"phase1_incumbent.scores.{score_name}.{head}"
            )
            validated[score_name][head] = value
    for head in HEADS:
        if not torch.allclose(
            validated["p0a"][head],
            p0a["oof_scores"][head],
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError(
                f"Phase-1 incumbent's P0-a foundation differs for {head}"
            )
    return validated["final_blend"], {
        "path": str(path),
        "sha256": _sha256(path),
        "kind": PHASE1_OOF_KIND,
        "score_key": "scores.final_blend",
        "sample_id_order_exact": True,
        "video_uid_order_exact": True,
        "folds_exact": True,
        "labels_exact": True,
        "taxonomy_exact": True,
        "p0a_foundation_exact": True,
    }


def _p0a_entry(p0a: dict[str, Any], head: str, test_fold: int) -> dict[str, Any]:
    entries = p0a.get("selections", {}).get(head, [])
    matching = [entry for entry in entries if int(entry.get("test_fold", -1)) == test_fold]
    if len(matching) != 1:
        raise ValueError(f"P0-a {head}/fold {test_fold} selection is missing or duplicated")
    return matching[0]


def _relearn_p0a(
    endpoint: dict[str, Any],
    p0a: dict[str, Any],
    head: str,
    labels: torch.Tensor,
    tune_mask: torch.Tensor,
    test_mask: torch.Tensor,
    test_fold: int,
    num_classes: int,
) -> tuple[torch.Tensor, dict[str, Any], float]:
    probabilities = torch.stack(
        [
            torch.softmax(endpoint["candidates"][name]["logits"][head], dim=-1)
            for name in P0A_CANDIDATES
        ]
    )
    selected, trace = _caruana_select(
        probabilities[:, tune_mask],
        labels[tune_mask],
        P0A_CANDIDATES,
        num_classes=num_classes,
    )
    names = [P0A_CANDIDATES[index] for index in selected]
    stored_entry = _p0a_entry(p0a, head, test_fold)
    if int(stored_entry.get("tune_fold", -1)) != 1 - test_fold:
        raise ValueError(f"Stored P0-a {head}/fold {test_fold} used the wrong tune fold")
    stored_names = [str(value) for value in stored_entry.get("selected_with_replacement", [])]
    if names != stored_names:
        raise ValueError(
            f"Relearned P0-a recipe differs for {head}/fold {test_fold}: "
            f"relearned={names}, stored={stored_names}"
        )
    full = probabilities[selected].mean(dim=0)
    stored_scores = p0a["oof_scores"][head]
    error = float((full[test_mask] - stored_scores[test_mask]).abs().max())
    if not torch.allclose(full[test_mask], stored_scores[test_mask], atol=2e-6, rtol=2e-6):
        raise ValueError(
            f"Relearned P0-a scores differ for {head}/fold {test_fold}; "
            f"max_abs_error={error}"
        )
    return full, {
        "selected_with_replacement": names,
        "selection_counts": dict(Counter(names)),
        "best_prefix_length": len(names),
        "trace": trace,
        "tune_metrics": _metric_block(full[tune_mask], labels[tune_mask], num_classes),
        "test_metrics": _metric_block(full[test_mask], labels[test_mask], num_classes),
    }, error


def _arm_id(learning_rate: float, weight_decay: float) -> str:
    return f"lr_{learning_rate:.0e}__wd_{weight_decay:.0e}".replace("+", "")


def _validate_arm_specs(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported Phase-2 manifest format")
    if manifest.get("kind") != "goalstep_history_probe_zoo_provenance":
        raise ValueError("Unexpected Phase-2 manifest kind")
    if int(manifest.get("trained_arm_count", -1)) != 11:
        raise ValueError("Phase-2 manifest must contain eleven trained arms")
    if int(manifest.get("total_grid_arm_count", -1)) != 12:
        raise ValueError("Phase-2 manifest must contain twelve total arms")
    raw_specs = manifest.get("registered_grid", [])
    if not isinstance(raw_specs, list) or len(raw_specs) != 12:
        raise ValueError("Phase-2 registered grid must contain twelve entries")
    by_pair: dict[tuple[float, float], dict[str, Any]] = {}
    for raw in raw_specs:
        spec = dict(raw)
        pair = (float(spec.get("learning_rate", math.nan)), float(spec.get("weight_decay", math.nan)))
        expected_id = _arm_id(*pair)
        if spec.get("arm_id") != expected_id:
            raise ValueError(f"Phase-2 arm ID disagrees with LR/WD: {spec}")
        if pair in by_pair:
            raise ValueError(f"Duplicate Phase-2 arm pair: {pair}")
        by_pair[pair] = spec
    if set(by_pair) != set(REGISTERED_GRID):
        raise ValueError("Phase-2 LR/WD grid differs from the registered 3x4 grid")
    ordered = [by_pair[pair] for pair in REGISTERED_GRID]
    for grid_index, spec in enumerate(ordered):
        if int(spec.get("grid_index", -1)) != grid_index:
            raise ValueError(f"Phase-2 grid index mismatch for {spec['arm_id']}")
    default = by_pair[REGISTERED_DEFAULT]
    skipped = manifest.get("skipped_default_arm", {})
    if skipped.get("arm_id") != default["arm_id"]:
        raise ValueError("Phase-2 skipped default arm is inconsistent")
    return ordered, default


def _prediction_paths(directory: Path, expected_epochs: int, *, include_zero: bool) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(directory.glob("epoch_*.pt")):
        try:
            epoch = int(path.stem.rsplit("_", 1)[1])
        except ValueError as error:
            raise ValueError(f"Invalid prediction filename: {path}") from error
        if epoch in paths:
            raise ValueError(f"Duplicate prediction epoch {epoch} under {directory}")
        paths[epoch] = path
    expected = set(range(0 if include_zero else 1, expected_epochs + 1))
    if set(paths) != expected:
        raise ValueError(
            f"Prediction epochs under {directory} are {sorted(paths)}, expected {sorted(expected)}"
        )
    return paths


def _validate_frozen_phase1_oof_inventory(
    manifest: dict[str, Any], phase1_oof_path: Path
) -> dict[str, Any]:
    """Require the Phase-1 OOF input to be the exact artifact frozen by Phase 2."""
    default_phase1 = manifest.get("default_phase1")
    if not isinstance(default_phase1, dict):
        raise ValueError("Phase-2 manifest default Phase-1 inventory is invalid")
    raw_run_dir = default_phase1.get("run_dir")
    entries = default_phase1.get("files")
    if not isinstance(raw_run_dir, str) or not raw_run_dir:
        raise ValueError("Phase-2 manifest default Phase-1 run directory is invalid")
    if not isinstance(entries, list):
        raise ValueError("Phase-2 manifest default Phase-1 inventory is invalid")

    candidates = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and Path(str(entry.get("path", ""))).name == PHASE1_OOF_FILENAME
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Phase-2 frozen default Phase-1 inventory must contain exactly one "
            f"{PHASE1_OOF_FILENAME}; found {len(candidates)}"
        )
    entry = candidates[0]
    raw_frozen_path = entry.get("path")
    if not isinstance(raw_frozen_path, str) or not raw_frozen_path:
        raise ValueError("Phase-2 frozen Phase-1 OOF inventory path is invalid")

    try:
        input_path = phase1_oof_path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Phase-1 OOF input does not exist: {phase1_oof_path}"
        ) from error
    try:
        frozen_path = Path(raw_frozen_path).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Frozen Phase-1 OOF inventory path does not exist: {raw_frozen_path}"
        ) from error
    expected_path = (
        Path(raw_run_dir).expanduser() / PHASE1_OOF_FILENAME
    ).resolve(strict=True)
    if frozen_path != expected_path:
        raise ValueError(
            "Phase-2 frozen Phase-1 OOF inventory path is outside/inconsistent with "
            "its frozen run directory"
        )
    if input_path != frozen_path:
        raise ValueError(
            "Phase-1 OOF input path differs from the frozen Phase-2 inventory: "
            f"input={input_path}, frozen={frozen_path}"
        )

    frozen_sha256 = str(entry.get("sha256", "")).lower()
    if len(frozen_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in frozen_sha256
    ):
        raise ValueError("Phase-2 frozen Phase-1 OOF SHA-256 is invalid")
    actual_sha256 = _sha256(input_path)
    if actual_sha256 != frozen_sha256:
        raise ValueError(
            "Phase-1 OOF SHA-256 differs from the frozen Phase-2 inventory: "
            f"actual={actual_sha256}, frozen={frozen_sha256}"
        )

    frozen_bytes = entry.get("bytes")
    if not isinstance(frozen_bytes, int) or frozen_bytes < 0:
        raise ValueError("Phase-2 frozen Phase-1 OOF byte count is invalid")
    actual_bytes = input_path.stat().st_size
    if actual_bytes != frozen_bytes:
        raise ValueError(
            "Phase-1 OOF byte count differs from the frozen Phase-2 inventory: "
            f"actual={actual_bytes}, frozen={frozen_bytes}"
        )
    return {
        "inventory_path": raw_frozen_path,
        "canonical_path": str(input_path),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
        "canonical_path_exact": True,
        "sha256_exact": True,
        "bytes_exact": True,
        "inventory_entry_unique": True,
    }


def _load_phase2(
    *,
    phase1_oof_path: Path,
    default_predictions_dir: Path,
    zoo_run_dir: Path,
    sample_ids: list[str],
    video_uids: list[str],
    labels: dict[str, torch.Tensor],
    num_classes: dict[str, int],
    endpoint: dict[str, Any],
    expected_epochs: int,
) -> tuple[dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]], dict[str, Any]]:
    if expected_epochs < 2:
        raise ValueError("Phase-2 top-2 epoch prefilter requires at least two epochs")
    manifest_path = zoo_run_dir / "run_manifest.json"
    final_path = zoo_run_dir / "final_metrics.json"
    if not manifest_path.is_file() or not final_path.is_file():
        raise FileNotFoundError("Phase-2 run_manifest.json and final_metrics.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    specs, default_spec = _validate_arm_specs(manifest)
    fingerprint = str(manifest.get("provenance_fingerprint", ""))
    if not fingerprint:
        raise ValueError("Phase-2 manifest has no provenance fingerprint")
    missing_provenance = [key for key in PROVENANCE_KEYS if key not in manifest]
    if missing_provenance:
        raise ValueError(f"Phase-2 manifest lacks provenance keys: {missing_provenance}")
    recomputed_fingerprint = _fingerprint(
        {key: manifest[key] for key in PROVENANCE_KEYS}
    )
    if recomputed_fingerprint != fingerprint:
        raise ValueError("Phase-2 manifest provenance fingerprint is internally inconsistent")
    if (
        int(manifest.get("epochs", -1)) != expected_epochs
        or int(manifest.get("val_rows", -1)) != len(sample_ids)
        or {head: int(manifest.get("num_classes", {}).get(head, -1)) for head in HEADS}
        != num_classes
    ):
        raise ValueError("Phase-2 manifest epoch/cohort/taxonomy contract mismatch")
    if (
        final.get("kind") != "goalstep_history_probe_zoo_result"
        or final.get("status") != "complete"
        or int(final.get("completed_epoch", -1)) != expected_epochs
        or int(final.get("epochs", -1)) != expected_epochs
        or final.get("provenance_fingerprint") != fingerprint
    ):
        raise ValueError("Phase-2 final_metrics.json is incomplete or has different provenance")

    frozen_phase1_oof = _validate_frozen_phase1_oof_inventory(
        manifest, phase1_oof_path
    )

    expected_trained_ids = {spec["arm_id"] for spec in specs if spec != default_spec}
    arms_root = zoo_run_dir / "arms"
    observed_trained_ids = {path.name for path in arms_root.iterdir() if path.is_dir()}
    if observed_trained_ids != expected_trained_ids:
        raise ValueError(
            "Phase-2 trained arm directories differ from the manifest: "
            f"missing={sorted(expected_trained_ids - observed_trained_ids)}, "
            f"extra={sorted(observed_trained_ids - expected_trained_ids)}"
        )

    n = len(sample_ids)
    phase2: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    file_inventory: dict[str, Any] = {}
    reference_history_lengths: torch.Tensor | None = None
    endpoint_visual = {
        head: endpoint["candidates"]["next_ep03"]["logits"][head] for head in HEADS
    }
    default_inventory_entries = manifest.get("default_phase1", {}).get("files", [])
    if not isinstance(default_inventory_entries, list):
        raise ValueError("Phase-2 manifest default Phase-1 inventory is invalid")
    default_hash_by_epoch: dict[int, str] = {}
    for entry in default_inventory_entries:
        raw_path = Path(str(entry.get("path", "")))
        if raw_path.parent.name != "val_predictions" or not raw_path.name.startswith("epoch_"):
            continue
        try:
            inventory_epoch = int(raw_path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if inventory_epoch in default_hash_by_epoch:
            raise ValueError("Default Phase-1 inventory duplicates a prediction epoch")
        default_hash_by_epoch[inventory_epoch] = str(entry.get("sha256", ""))
    if set(default_hash_by_epoch) != set(range(expected_epochs + 1)):
        raise ValueError("Default Phase-1 provenance inventory lacks epochs 0..expected")
    for spec in specs:
        arm_id = str(spec["arm_id"])
        is_default = arm_id == default_spec["arm_id"]
        directory = (
            default_predictions_dir
            if is_default
            else arms_root / arm_id / "val_predictions"
        )
        paths = _prediction_paths(
            directory, expected_epochs, include_zero=is_default
        )
        phase2[arm_id] = {}
        file_inventory[arm_id] = {
            "learning_rate": float(spec["learning_rate"]),
            "weight_decay": float(spec["weight_decay"]),
            "grid_index": int(spec["grid_index"]),
            "default_phase1_reuse": is_default,
            "epochs": {},
        }
        for epoch in range(1, expected_epochs + 1):
            path = paths[epoch]
            path_hash = _sha256(path)
            if is_default and path_hash != default_hash_by_epoch[epoch]:
                raise ValueError(f"{path}: hash differs from the frozen default-arm inventory")
            artifact = _torch_load(path, mmap=True)
            if artifact.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported prediction format")
            if artifact.get("kind") != PREDICTION_KIND:
                raise ValueError(f"{path}: unexpected prediction kind")
            if artifact.get("contract") != CONTRACT or int(artifact.get("epoch", -1)) != epoch:
                raise ValueError(f"{path}: contract or epoch mismatch")
            if artifact.get("sample_ids") != sample_ids:
                raise ValueError(f"{path}: sample order mismatch")
            if artifact.get("video_uids") != video_uids:
                raise ValueError(f"{path}: video order mismatch")
            artifact_classes = {
                head: int(artifact.get("num_classes", {}).get(head, -1)) for head in HEADS
            }
            if artifact_classes != num_classes:
                raise ValueError(f"{path}: taxonomy mismatch")
            if set(artifact.get("labels", {})) != set(HEADS):
                raise ValueError(f"{path}: label head set mismatch")
            for head in HEADS:
                value = _require_tensor(
                    artifact["labels"][head],
                    name=f"{path}.labels.{head}",
                    shape=(n,),
                    dtype=torch.int64,
                )
                if not torch.equal(value, labels[head]):
                    raise ValueError(f"{path}: {head} labels differ")
            history_lengths = _require_tensor(
                artifact.get("history_lengths"),
                name=f"{path}.history_lengths",
                shape=(n,),
                dtype=torch.int64,
            )
            if reference_history_lengths is None:
                reference_history_lengths = history_lengths
            elif not torch.equal(reference_history_lengths, history_lengths):
                raise ValueError(f"{path}: history lengths differ across arms/epochs")
            logits = artifact.get("logits", {})
            if set(logits) != set(MODES):
                raise ValueError(f"{path}: logits must contain exactly {MODES}")
            for mode in MODES:
                if set(logits[mode]) != set(HEADS):
                    raise ValueError(f"{path}: logits[{mode}] head set mismatch")
                for head in HEADS:
                    _require_tensor(
                        logits[mode][head],
                        name=f"{path}.logits.{mode}.{head}",
                        shape=(n, num_classes[head]),
                        dtype=torch.float32,
                    )
            for head in HEADS:
                if not torch.allclose(
                    logits["visual"][head], endpoint_visual[head], atol=1e-5, rtol=1e-5
                ):
                    raise ValueError(f"{path}: visual source differs from endpoint next_ep03/{head}")
            if not is_default:
                if (
                    artifact.get("phase") != "P2"
                    or artifact.get("arm_id") != arm_id
                    or not math.isclose(
                        float(artifact.get("learning_rate", math.nan)),
                        float(spec["learning_rate"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        float(artifact.get("weight_decay", math.nan)),
                        float(spec["weight_decay"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    or artifact.get("provenance_fingerprint") != fingerprint
                ):
                    raise ValueError(f"{path}: Phase-2 arm provenance mismatch")
            phase2[arm_id][epoch] = {
                mode: {head: logits[mode][head] for head in HEADS}
                for mode in SELECTED_MODES
            }
            file_inventory[arm_id]["epochs"][str(epoch)] = {
                "path": str(path),
                "sha256": path_hash,
            }
    return phase2, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "final_metrics": str(final_path),
        "final_metrics_sha256": _sha256(final_path),
        "provenance_fingerprint": fingerprint,
        "frozen_phase1_oof": frozen_phase1_oof,
        "arm_files": file_inventory,
        "history_lengths_exact": True,
    }


def _prefilter_and_select(
    phase2: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    *,
    mode: str,
    head: str,
    labels: torch.Tensor,
    tune_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode not in SELECTED_MODES:
        raise ValueError(f"Unsupported selected mode: {mode}")
    candidate_names: list[str] = []
    candidate_probabilities: list[torch.Tensor] = []
    prefilter: dict[str, Any] = {}
    for arm_id, epoch_records in phase2.items():
        scored: list[tuple[float, int]] = []
        epoch_trace: list[dict[str, Any]] = []
        for epoch in sorted(epoch_records):
            logits = epoch_records[epoch][mode][head]
            value = top_k_recall(logits[tune_mask], labels[tune_mask], k=5)
            scored.append((value, epoch))
            epoch_trace.append({"epoch": epoch, "tune_top5": value})
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        chosen_epochs = [epoch for _, epoch in ranked[:2]]
        prefilter[arm_id] = {
            "epoch_tuning_trace": epoch_trace,
            "selected_top2_epochs": chosen_epochs,
        }
        for epoch in chosen_epochs:
            candidate_names.append(f"{arm_id}@epoch_{epoch:02d}")
            candidate_probabilities.append(
                torch.softmax(epoch_records[epoch][mode][head], dim=-1)
            )
    if len(candidate_names) != 24:
        raise RuntimeError(f"Phase-2 prefilter must produce 24 candidates, got {len(candidate_names)}")
    probabilities = torch.stack(candidate_probabilities)
    selected, trace = _caruana_select(
        probabilities[:, tune_mask],
        labels[tune_mask],
        candidate_names,
        num_classes=num_classes,
    )
    full = probabilities[selected].mean(dim=0)
    selected_names = [candidate_names[index] for index in selected]
    return full, {
        "mode": mode,
        "per_arm_top2_prefilter": prefilter,
        "prefilter_candidate_count": len(candidate_names),
        "prefilter_candidate_names": candidate_names,
        "caruana_max_rounds": CARUANA_ROUNDS,
        "selected_with_replacement": selected_names,
        "selection_counts": dict(Counter(selected_names)),
        "best_prefix_length": len(selected_names),
        "caruana_trace": trace,
        "tune_metrics": _metric_block(full[tune_mask], labels[tune_mask], num_classes),
        "test_metrics": _metric_block(full[test_mask], labels[test_mask], num_classes),
    }


def _select_alpha(
    p0a: torch.Tensor,
    phase2: torch.Tensor,
    labels: torch.Tensor,
    tune_mask: torch.Tensor,
    alpha_grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    best_alpha = float(alpha_grid[0])
    best_value = -math.inf
    trace: list[dict[str, float]] = []
    for raw_alpha in alpha_grid:
        alpha = float(raw_alpha)
        blend = (1.0 - alpha) * p0a[tune_mask] + alpha * phase2[tune_mask]
        value = top_k_recall(blend, labels[tune_mask], k=5)
        trace.append({"phase2_weight": alpha, "tune_top5": value})
        # Ascending grid + strict improvement conservatively keeps more P0-a on a tie.
        if value > best_value + 1e-12:
            best_alpha = alpha
            best_value = value
    return best_alpha, trace


def _crossfit(
    *,
    endpoint: dict[str, Any],
    p0a: dict[str, Any],
    phase2: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    folds: torch.Tensor,
    labels: dict[str, torch.Tensor],
    num_classes: dict[str, int],
    alpha_grid: Sequence[float],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], dict[str, float]]:
    n = len(folds)
    output_names = ("p0a", "phase2_selected", "current_only_control", "final_blend")
    outputs = {
        name: {
            head: torch.empty((n, num_classes[head]), dtype=torch.float32)
            for head in HEADS
        }
        for name in output_names
    }
    selections: dict[str, list[dict[str, Any]]] = {head: [] for head in HEADS}
    reconstruction_error = {head: 0.0 for head in HEADS}
    for head in HEADS:
        for test_fold in (0, 1):
            tune_fold = 1 - test_fold
            tune_mask = folds == tune_fold
            test_mask = folds == test_fold
            p0a_full, p0a_selection, error = _relearn_p0a(
                endpoint,
                p0a,
                head,
                labels[head],
                tune_mask,
                test_mask,
                test_fold,
                num_classes[head],
            )
            reconstruction_error[head] = max(reconstruction_error[head], error)
            fused_full, fused_selection = _prefilter_and_select(
                phase2,
                mode="fused",
                head=head,
                labels=labels[head],
                tune_mask=tune_mask,
                test_mask=test_mask,
                num_classes=num_classes[head],
            )
            current_full, current_selection = _prefilter_and_select(
                phase2,
                mode="current_only",
                head=head,
                labels=labels[head],
                tune_mask=tune_mask,
                test_mask=test_mask,
                num_classes=num_classes[head],
            )
            alpha, alpha_trace = _select_alpha(
                p0a_full, fused_full, labels[head], tune_mask, alpha_grid
            )
            outputs["p0a"][head][test_mask] = p0a_full[test_mask]
            outputs["phase2_selected"][head][test_mask] = fused_full[test_mask]
            outputs["current_only_control"][head][test_mask] = current_full[test_mask]
            outputs["final_blend"][head][test_mask] = (
                (1.0 - alpha) * p0a_full[test_mask] + alpha * fused_full[test_mask]
            )
            selections[head].append(
                {
                    "test_fold": test_fold,
                    "tune_fold": tune_fold,
                    "test_samples": int(test_mask.sum()),
                    "tune_samples": int(tune_mask.sum()),
                    "p0a": p0a_selection,
                    "phase2_fused": fused_selection,
                    "phase2_current_only": current_selection,
                    "phase2_blend_weight": alpha,
                    "blend_tuning_trace": alpha_trace,
                    "test_top5": {
                        "p0a": top_k_recall(
                            p0a_full[test_mask], labels[head][test_mask], k=5
                        ),
                        "phase2_selected": top_k_recall(
                            fused_full[test_mask], labels[head][test_mask], k=5
                        ),
                        "current_only_control": top_k_recall(
                            current_full[test_mask], labels[head][test_mask], k=5
                        ),
                        "final_blend": top_k_recall(
                            outputs["final_blend"][head][test_mask],
                            labels[head][test_mask],
                            k=5,
                        ),
                    },
                }
            )
    for name, by_head in outputs.items():
        for head, value in by_head.items():
            _validate_probabilities(value, name=f"OOF {name}/{head}")
    return outputs, selections, reconstruction_error


def _decision(comparison: dict[str, Any]) -> dict[str, Any]:
    delta = float(comparison["delta_top5_pp"])
    lower = float(comparison["video_bootstrap_95ci_pp"][0])
    outer_rule_passed = delta > 0.0 and lower > 0.0
    return {
        "criterion": "delta_top5_pp > 0 and video-bootstrap 95% CI lower bound > 0",
        "delta_positive": delta > 0.0,
        "bootstrap_ci_lower_positive": lower > 0.0,
        "outer_fold_rule_passed": outer_rule_passed,
        "provisional_engineering_adoption": outer_rule_passed,
        "confirmatory_claim_allowed": False,
        "gain_at_least_1pp_descriptive_only": delta >= 1.0,
    }


def evaluate_phase2(
    *,
    endpoint_path: Path,
    p0a_path: Path,
    phase1_oof_path: Path,
    default_predictions_dir: Path,
    zoo_run_dir: Path,
    output_json: Path,
    output_scores: Path,
    expected_epochs: int = 10,
    alpha_step: float = 0.05,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0.0 < alpha_step <= 1.0:
        raise ValueError("alpha_step must be in (0,1]")
    steps = int(round(1.0 / alpha_step))
    if not math.isclose(steps * alpha_step, 1.0, abs_tol=1e-9):
        raise ValueError("alpha_step must divide [0,1] exactly")
    alpha_grid = [round(index * alpha_step, 10) for index in range(steps + 1)]
    endpoint, p0a, sample_ids, video_uids, folds, labels, num_classes = _load_base(
        endpoint_path, p0a_path
    )
    phase2, phase2_provenance = _load_phase2(
        phase1_oof_path=phase1_oof_path,
        default_predictions_dir=default_predictions_dir,
        zoo_run_dir=zoo_run_dir,
        sample_ids=sample_ids,
        video_uids=video_uids,
        labels=labels,
        num_classes=num_classes,
        endpoint=endpoint,
        expected_epochs=expected_epochs,
    )
    phase1_incumbent, phase1_incumbent_provenance = _load_phase1_incumbent(
        phase1_oof_path,
        sample_ids=sample_ids,
        video_uids=video_uids,
        folds=folds,
        labels=labels,
        num_classes=num_classes,
        p0a=p0a,
    )
    outputs, selections, reconstruction_error = _crossfit(
        endpoint=endpoint,
        p0a=p0a,
        phase2=phase2,
        folds=folds,
        labels=labels,
        num_classes=num_classes,
        alpha_grid=alpha_grid,
    )
    outputs["phase1_incumbent"] = phase1_incumbent
    metrics = {
        name: _all_metrics(scores, labels, num_classes) for name, scores in outputs.items()
    }
    paired = {
        "phase2_selected_vs_p0a": _paired_video_bootstrap(
            outputs["phase2_selected"]["action"],
            outputs["p0a"]["action"],
            labels["action"],
            video_uids,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "final_blend_vs_p0a": _paired_video_bootstrap(
            outputs["final_blend"]["action"],
            outputs["p0a"]["action"],
            labels["action"],
            video_uids,
            seed=seed + 1,
            bootstrap_samples=bootstrap_samples,
        ),
        "phase2_final_blend_vs_phase1_incumbent": _paired_video_bootstrap(
            outputs["final_blend"]["action"],
            outputs["phase1_incumbent"]["action"],
            labels["action"],
            video_uids,
            seed=seed + 2,
            bootstrap_samples=bootstrap_samples,
        ),
        "fused_vs_current_only_attribution": _paired_video_bootstrap(
            outputs["phase2_selected"]["action"],
            outputs["current_only_control"]["action"],
            labels["action"],
            video_uids,
            seed=seed + 3,
            bootstrap_samples=bootstrap_samples,
        ),
    }
    promotion_decision = _decision(
        paired["phase2_final_blend_vs_phase1_incumbent"]
    )
    champion_after_phase2 = (
        "phase2_final_blend_promoted_provisionally"
        if promotion_decision["provisional_engineering_adoption"]
        else "phase1_final_blend_incumbent_retained"
    )
    results: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "Phase-2 probe-zoo outer-video-fold selection",
        "contract": CONTRACT,
        "sample_count": len(sample_ids),
        "video_count": len(set(video_uids)),
        "num_classes": num_classes,
        "fold_sample_counts": {
            str(fold): int((folds == fold).sum()) for fold in (0, 1)
        },
        "inputs": {
            "endpoint_logits": str(endpoint_path),
            "endpoint_logits_sha256": _sha256(endpoint_path),
            "p0a_oof": str(p0a_path),
            "p0a_oof_sha256": _sha256(p0a_path),
            "phase1_incumbent_oof": phase1_incumbent_provenance,
            "default_phase1_predictions_dir": str(default_predictions_dir),
            "phase2_zoo_run_dir": str(zoo_run_dir),
            "phase2": phase2_provenance,
        },
        "selection_protocol": {
            "outer_folds": "exact P0-a video-disjoint folds",
            "p0a": (
                "relearn 8-checkpoint, 16-round Top-5 Caruana recipe on the opposite "
                "tune fold; apply unchanged to heldout"
            ),
            "phase2": (
                "12 arms x epochs 1..10; independently per field/mode retain each arm's "
                "tune-only top-2 epochs (24 candidates), then fixed 16-round Caruana"
            ),
            "selected_modes": list(SELECTED_MODES),
            "blend": "(1-alpha)*P0-a + alpha*selected Phase-2 fused probabilities",
            "alpha_grid": alpha_grid,
            "tie_breaks": (
                "earliest epoch within arm; registered arm/rank candidate order in Caruana; "
                "smaller Phase-2 alpha"
            ),
            "fieldwise_outer_fold_selections": selections,
            "no_full_validation_tuning": True,
        },
        "audit": {
            "sample_id_order_exact": True,
            "video_uid_order_exact": True,
            "labels_exact": True,
            "history_lengths_exact_across_arms_epochs": True,
            "raw_endpoint_candidate_set_exact": list(P0A_CANDIDATES),
            "registered_phase2_arms": 12,
            "epochs_per_arm": expected_epochs,
            "p0a_raw_reconstruction_max_abs_error": reconstruction_error,
            "inherited_validation_adaptivity": {
                "present": True,
                "source": (
                    "Every Phase-1/2 history arm inherits frozen visual logits from "
                    "next_ep03 (best.pt), whose epoch was previously chosen by Action Top-5 "
                    "on a seed-42 2,000-row subset drawn from this validation split."
                ),
                "impact": (
                    "The outer-fold selector removes new Phase-2 arm/epoch/ensemble/alpha "
                    "selection leakage, but cannot undo the earlier visual-base choice."
                ),
                "confirmatory_claim_allowed": False,
                "confirmation_required": (
                    "Use a fresh heldout/test split, or nest/repeat the visual-base checkpoint "
                    "selection inside each outer training/tuning fold before fitting residuals."
                ),
            },
        },
        "metrics_percent": metrics,
        "paired_action_top5": paired,
        "incumbent_before_phase2": {
            "name": "phase1_final_blend_oof",
            "score_key": "scores.final_blend",
            "source": str(phase1_oof_path),
            "action_metrics_percent": metrics["phase1_incumbent"]["action"],
            "role": "promotion baseline; P0-a remains a foundation diagnostic only",
            "confirmatory_claim_allowed": False,
        },
        "phase2_promotion": {
            "challenger": "phase2_final_blend_oof",
            "incumbent": "phase1_final_blend_oof",
            "paired_action_top5": paired[
                "phase2_final_blend_vs_phase1_incumbent"
            ],
            "decision": promotion_decision,
            "champion_after_phase2": champion_after_phase2,
            "promoted": promotion_decision["provisional_engineering_adoption"],
            "retained": not promotion_decision["provisional_engineering_adoption"],
            "status": (
                "provisional_promotion"
                if promotion_decision["provisional_engineering_adoption"]
                else "incumbent_retained"
            ),
            "confirmatory_claim_allowed": False,
        },
        "decisions": {
            "phase2_selected_over_p0a": _decision(paired["phase2_selected_vs_p0a"]),
            "final_blend_over_p0a": _decision(paired["final_blend_vs_p0a"]),
            "phase2_final_blend_over_phase1_incumbent": promotion_decision,
            "history_attribution_over_current_only": _decision(
                paired["fused_vs_current_only_attribution"]
            ),
            "note": (
                "+1pp is descriptive only. Phase-2 promotion is judged against the Phase-1 "
                "final-blend incumbent, not P0-a: it requires positive paired Action Top-5 "
                "delta and a positive video-cluster bootstrap 95% CI lower bound. P0-a remains "
                "a foundation diagnostic. Any promotion is provisional engineering adoption, "
                "not a confirmatory claim, because all residual arms inherit a "
                "validation-selected next_ep03 visual base."
            ),
            "confirmatory_claim_allowed": False,
        },
        "limitations": [
            (
                "The current_only control is independently prefiltered and ensembled as "
                "required. Its paired delta against fused compares two selected pipelines; "
                "it is not the stricter same-arm/same-epoch token intervention estimate."
            ),
            (
                "Two outer folds provide cross-fitted row predictions but no untouched final "
                "test set. Inherited next_ep03 validation selection makes all champion decisions "
                "provisional until fresh-heldout or fully nested confirmation."
            ),
        ],
    }
    score_artifact = {
        "format_version": FORMAT_VERSION,
        "kind": "goalstep_history_probe_zoo_crossfit_oof_scores",
        "contract": CONTRACT,
        "sample_ids": sample_ids,
        "video_uids": video_uids,
        "folds": folds,
        "labels": labels,
        "num_classes": num_classes,
        "scores": outputs,
        "selections": selections,
        "metrics_percent": metrics,
        "phase2_provenance_fingerprint": phase2_provenance["provenance_fingerprint"],
    }
    _atomic_torch(score_artifact, output_scores)
    _atomic_json(results, output_json)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-logits",
        type=Path,
        default=Path("outputs/goalstep/runs/history_context_phase0/endpoint_logits.pt"),
    )
    parser.add_argument(
        "--p0a-oof",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/history_context_phase0/"
            "p0a_primary_same_decision_oof_scores.pt"
        ),
    )
    parser.add_argument(
        "--phase1-oof",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_k8_vna_ep10/"
            "history_context_vs_p0a_oof_scores.pt"
        ),
    )
    parser.add_argument(
        "--default-predictions-dir",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_k8_vna_ep10/val_predictions"
        ),
    )
    parser.add_argument(
        "--zoo-run-dir",
        type=Path,
        default=Path("outputs/goalstep/runs/z1_history_context_probe_zoo_ep10"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_probe_zoo_ep10/"
            "phase2_vs_p0a_results.json"
        ),
    )
    parser.add_argument(
        "--output-scores",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_probe_zoo_ep10/"
            "phase2_vs_p0a_oof_scores.pt"
        ),
    )
    parser.add_argument("--expected-epochs", type=int, default=10)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = evaluate_phase2(
        endpoint_path=args.endpoint_logits,
        p0a_path=args.p0a_oof,
        phase1_oof_path=args.phase1_oof,
        default_predictions_dir=args.default_predictions_dir,
        zoo_run_dir=args.zoo_run_dir,
        output_json=args.output_json,
        output_scores=args.output_scores,
        expected_epochs=args.expected_epochs,
        alpha_step=args.alpha_step,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    action = results["metrics_percent"]
    print(
        "Action Top-5: "
        f"P0-a={action['p0a']['action']['top5']:.3f} "
        f"Phase-1-incumbent={action['phase1_incumbent']['action']['top5']:.3f} "
        f"Phase-2={action['phase2_selected']['action']['top5']:.3f} "
        f"final-blend={action['final_blend']['action']['top5']:.3f}",
        flush=True,
    )
    print(f"wrote {args.output_json} and {args.output_scores}", flush=True)


if __name__ == "__main__":
    main()
