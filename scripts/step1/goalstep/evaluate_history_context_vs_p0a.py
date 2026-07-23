#!/usr/bin/env python3
"""Leakage-conscious comparison of GoalStep history context against P0-a.

The evaluator deliberately does not select a single Phase-1 epoch on the
entire validation set.  For each video-disjoint test fold it uses only the
other fold to:

1. select a Phase-1 epoch independently for verb, noun, and action;
2. reconstruct the corresponding P0-a endpoint ensemble from *raw* next-probe
   checkpoint logits; and
3. select a field-wise probability blend weight.

The selected epoch and blend are then applied to the held-out fold.  The raw
endpoint reconstruction is important: using P0-a's OOF predictions on the
tuning fold would leak the held-out fold labels through P0-a's own two-fold
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ego.step1_action_anticipation.metrics import (  # noqa: E402
    class_mean_recall,
    top_k_recall,
)


HEADS = ("verb", "noun", "action")
MODES = ("visual", "history", "current_only", "fused")
PREDICTION_KIND = "goalstep_history_context_val_predictions"
CONTRACT = "A2.end-1s -> strict same-level A3"
FORMAT_VERSION = 1


def _torch_load(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:  # pragma: no cover - older PyTorch
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _require_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
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


def _topk_hits(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    top = scores.topk(min(k, scores.shape[-1]), dim=-1).indices
    return (top == labels[:, None]).any(dim=1)


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
    challenger_hit = _topk_hits(challenger, labels, k=5)
    baseline_hit = _topk_hits(baseline, labels, k=5)
    paired = challenger_hit.to(torch.int8) - baseline_hit.to(torch.int8)
    delta = 100.0 * float(paired.float().mean())
    if len(paired) > 1:
        standard_error = 100.0 * float(paired.double().std(unbiased=True)) / math.sqrt(len(paired))
    else:
        standard_error = 0.0

    positions_by_video: dict[str, list[int]] = {}
    for position, video_uid in enumerate(video_uids):
        positions_by_video.setdefault(str(video_uid), []).append(position)
    videos = sorted(positions_by_video)
    if not videos:
        raise ValueError("Cannot bootstrap an empty validation set")
    paired_np = paired.numpy().astype(np.float64, copy=False)
    cluster_sums = np.asarray(
        [paired_np[positions_by_video[video]].sum() for video in videos], dtype=np.float64
    )
    cluster_counts = np.asarray(
        [len(positions_by_video[video]) for video in videos], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(videos), size=(bootstrap_samples, len(videos)))
    bootstrap = 100.0 * cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "n": len(paired),
        "videos": len(videos),
        "challenger_only_correct": int((challenger_hit & ~baseline_hit).sum()),
        "baseline_only_correct": int((baseline_hit & ~challenger_hit).sum()),
        "both_correct": int((challenger_hit & baseline_hit).sum()),
        "neither_correct": int((~challenger_hit & ~baseline_hit).sum()),
        "delta_top5_pp": delta,
        "normal_95ci_pp": [delta - 1.96 * standard_error, delta + 1.96 * standard_error],
        "video_bootstrap_95ci_pp": interval,
        "video_bootstrap_probability_positive": float((bootstrap > 0).mean()),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def _validate_video_folds(folds: torch.Tensor, video_uids: Sequence[str]) -> None:
    if tuple(folds.shape) != (len(video_uids),) or folds.dtype != torch.int64:
        raise ValueError("P0-a folds must be int64[N]")
    if set(folds.tolist()) != {0, 1}:
        raise ValueError("P0-a artifact must contain exactly two non-empty folds")
    fold_by_video: dict[str, int] = {}
    for video_uid, fold in zip(video_uids, folds.tolist()):
        previous = fold_by_video.setdefault(str(video_uid), int(fold))
        if previous != int(fold):
            raise ValueError(f"video {video_uid!r} crosses validation folds")


def _validate_probability_scores(value: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} is not a finite non-negative probability tensor")
    row_sums = value.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-5, rtol=2e-5):
        raise ValueError(f"{name} rows do not sum to one")


def _load_and_validate_inputs(
    endpoint_path: Path,
    p0a_path: Path,
    predictions_dir: Path,
    *,
    expected_last_epoch: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], dict[str, int]]:
    endpoint = _torch_load(endpoint_path, mmap=True)
    p0a = _torch_load(p0a_path, mmap=True)
    if int(p0a.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported P0-a artifact format")
    if p0a.get("deployable_at_A2_boundary") is not True:
        raise ValueError("P0-a artifact is not the deployable same-decision ensemble")
    sample_ids = [str(value) for value in p0a.get("sample_ids", [])]
    video_uids = [str(value) for value in p0a.get("video_uids", [])]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("P0-a sample IDs must be non-empty and unique")
    if len(video_uids) != len(sample_ids):
        raise ValueError("P0-a video/sample lengths differ")
    n = len(sample_ids)
    folds = _require_tensor(
        p0a.get("folds"), name="p0a.folds", shape=(n,), dtype=torch.int64
    )
    _validate_video_folds(folds, video_uids)

    if endpoint.get("logical_sample_ids") != sample_ids:
        raise ValueError("Endpoint logits and P0-a logical sample order differ")
    if endpoint.get("source_cache_sample_ids") != sample_ids:
        raise ValueError("Endpoint source-cache IDs do not equal the strict-future cohort IDs")
    if endpoint.get("video_uids") != video_uids:
        raise ValueError("Endpoint logits and P0-a video order differ")
    raw_classes = endpoint.get("num_classes", {})
    if set(raw_classes) != set(HEADS):
        raise ValueError("Endpoint num_classes is incomplete")
    num_classes = {head: int(raw_classes[head]) for head in HEADS}
    labels: dict[str, torch.Tensor] = {}
    for head in HEADS:
        labels[head] = _require_tensor(
            p0a.get("labels", {}).get(head),
            name=f"p0a.labels.{head}",
            shape=(n,),
            dtype=torch.int64,
        )
        endpoint_label = _require_tensor(
            endpoint.get("labels", {}).get(head),
            name=f"endpoint.labels.{head}",
            shape=(n,),
            dtype=torch.int64,
        )
        if not torch.equal(labels[head], endpoint_label):
            raise ValueError(f"Endpoint and P0-a {head} labels differ")
        p0a_score = _require_tensor(
            p0a.get("oof_scores", {}).get(head),
            name=f"p0a.oof_scores.{head}",
            shape=(n, num_classes[head]),
            dtype=torch.float32,
        )
        _validate_probability_scores(p0a_score, name=f"p0a.oof_scores.{head}")

    candidates = endpoint.get("candidates", {})
    next_candidates = sorted(name for name in candidates if str(name).startswith("next_ep"))
    if len(next_candidates) < 2:
        raise ValueError("Endpoint artifact needs raw logits from at least two next-probe checkpoints")
    for candidate in next_candidates:
        logits = candidates[candidate].get("logits", {})
        for head in HEADS:
            _require_tensor(
                logits.get(head),
                name=f"endpoint.{candidate}.{head}",
                shape=(n, num_classes[head]),
                dtype=torch.float32,
            )

    expected_epochs = set(range(expected_last_epoch + 1))
    paths: dict[int, Path] = {}
    for path in sorted(predictions_dir.glob("epoch_*.pt")):
        try:
            epoch = int(path.stem.split("_")[-1])
        except ValueError as error:
            raise ValueError(f"Invalid Phase-1 prediction filename: {path.name}") from error
        if epoch in paths:
            raise ValueError(f"Duplicate Phase-1 epoch artifact: {epoch}")
        paths[epoch] = path
    if set(paths) != expected_epochs:
        raise ValueError(
            f"Phase-1 prediction epochs are {sorted(paths)}, expected {sorted(expected_epochs)}"
        )

    phase1: dict[int, dict[str, Any]] = {}
    reference_history_lengths: torch.Tensor | None = None
    reference_visual: dict[str, torch.Tensor] | None = None
    for epoch in sorted(paths):
        artifact = _torch_load(paths[epoch], mmap=True)
        if int(artifact.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"Unsupported Phase-1 prediction format in {paths[epoch]}")
        if artifact.get("kind") != PREDICTION_KIND:
            raise ValueError(f"Unexpected prediction kind in {paths[epoch]}")
        if int(artifact.get("epoch", -1)) != epoch:
            raise ValueError(f"Filename/payload epoch mismatch in {paths[epoch]}")
        if artifact.get("contract") != CONTRACT:
            raise ValueError(f"Unexpected anticipation contract in {paths[epoch]}")
        if artifact.get("sample_ids") != sample_ids:
            raise ValueError(f"Phase-1/P0-a sample order mismatch at epoch {epoch}")
        if artifact.get("video_uids") != video_uids:
            raise ValueError(f"Phase-1/P0-a video order mismatch at epoch {epoch}")
        artifact_classes = {head: int(artifact.get("num_classes", {}).get(head, -1)) for head in HEADS}
        if artifact_classes != num_classes:
            raise ValueError(f"Phase-1 taxonomy mismatch at epoch {epoch}")
        for head in HEADS:
            artifact_label = _require_tensor(
                artifact.get("labels", {}).get(head),
                name=f"phase1[{epoch}].labels.{head}",
                shape=(n,),
                dtype=torch.int64,
            )
            if not torch.equal(artifact_label, labels[head]):
                raise ValueError(f"Phase-1/P0-a {head} labels differ at epoch {epoch}")
        history_lengths = _require_tensor(
            artifact.get("history_lengths"),
            name=f"phase1[{epoch}].history_lengths",
            shape=(n,),
            dtype=torch.int64,
        )
        if reference_history_lengths is None:
            reference_history_lengths = history_lengths
        elif not torch.equal(reference_history_lengths, history_lengths):
            raise ValueError(f"History lengths changed at epoch {epoch}")
        logits = artifact.get("logits", {})
        if set(logits) != set(MODES):
            raise ValueError(f"Phase-1 epoch {epoch} must contain exactly modes {MODES}")
        for mode in MODES:
            if set(logits[mode]) != set(HEADS):
                raise ValueError(f"Phase-1 epoch {epoch}/{mode} head set is invalid")
            for head in HEADS:
                _require_tensor(
                    logits[mode][head],
                    name=f"phase1[{epoch}].{mode}.{head}",
                    shape=(n, num_classes[head]),
                    dtype=torch.float32,
                )
        if reference_visual is None:
            reference_visual = logits["visual"]
        else:
            for head in HEADS:
                if not torch.equal(reference_visual[head], logits["visual"][head]):
                    raise ValueError(f"Frozen visual logits changed at epoch {epoch}/{head}")
        phase1[epoch] = artifact

    for head in HEADS:
        if not torch.equal(phase1[0]["logits"]["fused"][head], phase1[0]["logits"]["visual"][head]):
            raise ValueError(f"Epoch-0 fused/visual fallback is not bit-exact for {head}")
        endpoint_ep3 = candidates.get("next_ep03", {}).get("logits", {}).get(head)
        if endpoint_ep3 is None:
            raise ValueError("Endpoint artifact is missing next_ep03")
        if not torch.allclose(
            phase1[0]["logits"]["visual"][head], endpoint_ep3, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(f"Phase-1 visual source does not match endpoint next_ep03/{head}")
    return endpoint, p0a, phase1, num_classes


def _selection_for_test_fold(
    p0a: dict[str, Any], head: str, test_fold: int
) -> dict[str, Any]:
    entries = p0a.get("selections", {}).get(head, [])
    matching = [entry for entry in entries if int(entry.get("test_fold", -1)) == test_fold]
    if len(matching) != 1:
        raise ValueError(f"P0-a {head} needs one selection for test fold {test_fold}")
    entry = matching[0]
    if int(entry.get("tune_fold", -1)) != 1 - test_fold:
        raise ValueError(f"P0-a {head}/fold {test_fold} was not tuned on the opposite fold")
    names = [str(value) for value in entry.get("selected_with_replacement", [])]
    if not names:
        raise ValueError(f"P0-a {head}/fold {test_fold} has an empty ensemble")
    return entry


def _endpoint_probability_ensemble(
    endpoint: dict[str, Any], head: str, candidate_names: Sequence[str]
) -> torch.Tensor:
    candidates = endpoint["candidates"]
    missing = sorted(set(candidate_names) - set(candidates))
    if missing:
        raise ValueError(f"P0-a selection references absent raw candidates: {missing}")
    return torch.stack(
        [torch.softmax(candidates[name]["logits"][head], dim=-1) for name in candidate_names]
    ).mean(dim=0)


def _best_epoch_on_mask(
    phase1: dict[int, dict[str, Any]],
    head: str,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[int, list[dict[str, float | int]]]:
    trace: list[dict[str, float | int]] = []
    best_epoch = -1
    best_value = -math.inf
    for epoch in sorted(phase1):
        value = top_k_recall(phase1[epoch]["logits"]["fused"][head][mask], labels[mask], k=5)
        trace.append({"epoch": epoch, "tune_top5": value})
        # Sorted iteration plus strict improvement gives deterministic earliest-epoch ties.
        if value > best_value + 1e-12:
            best_epoch = epoch
            best_value = value
    return best_epoch, trace


def _best_alpha_on_mask(
    endpoint_probability: torch.Tensor,
    phase1_probability: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha_grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    trace: list[dict[str, float]] = []
    best_alpha = float(alpha_grid[0])
    best_value = -math.inf
    for raw_alpha in alpha_grid:
        alpha = float(raw_alpha)
        blend = (1.0 - alpha) * endpoint_probability[mask] + alpha * phase1_probability[mask]
        value = top_k_recall(blend, labels[mask], k=5)
        trace.append({"phase1_weight": alpha, "tune_top5": value})
        # The grid is ascending, so a tie keeps the smaller Phase-1 weight and
        # therefore the already-established P0-a champion.
        if value > best_value + 1e-12:
            best_alpha = alpha
            best_value = value
    return best_alpha, trace


def _crossfit(
    endpoint: dict[str, Any],
    p0a: dict[str, Any],
    phase1: dict[int, dict[str, Any]],
    num_classes: dict[str, int],
    *,
    alpha_grid: Sequence[float],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], dict[str, float]]:
    labels = p0a["labels"]
    folds = p0a["folds"]
    n = len(p0a["sample_ids"])
    outputs = {
        name: {
            head: torch.empty((n, num_classes[head]), dtype=torch.float32)
            for head in HEADS
        }
        for name in (
            "p0a",
            "phase1",
            "final_blend",
            "visual_same_epoch",
            "history_same_epoch",
            "current_only_same_epoch",
        )
    }
    selections: dict[str, list[dict[str, Any]]] = {head: [] for head in HEADS}
    max_reconstruction_error: dict[str, float] = {head: 0.0 for head in HEADS}

    for head in HEADS:
        for test_fold in (0, 1):
            tune_fold = 1 - test_fold
            tune_mask = folds == tune_fold
            test_mask = folds == test_fold
            p0a_selection = _selection_for_test_fold(p0a, head, test_fold)
            candidate_names = [
                str(value) for value in p0a_selection["selected_with_replacement"]
            ]
            endpoint_probability = _endpoint_probability_ensemble(
                endpoint, head, candidate_names
            )
            stored_p0a = p0a["oof_scores"][head]
            error = float(
                (endpoint_probability[test_mask] - stored_p0a[test_mask]).abs().max()
            )
            max_reconstruction_error[head] = max(max_reconstruction_error[head], error)
            if not torch.allclose(
                endpoint_probability[test_mask],
                stored_p0a[test_mask],
                atol=2e-6,
                rtol=2e-6,
            ):
                raise ValueError(
                    f"Raw endpoint reconstruction differs from P0-a OOF for {head}/fold "
                    f"{test_fold}; max_abs_error={error}"
                )

            epoch, epoch_trace = _best_epoch_on_mask(
                phase1, head, labels[head], tune_mask
            )
            selected = phase1[epoch]["logits"]
            phase1_probability = torch.softmax(selected["fused"][head], dim=-1)
            alpha, alpha_trace = _best_alpha_on_mask(
                endpoint_probability,
                phase1_probability,
                labels[head],
                tune_mask,
                alpha_grid,
            )
            outputs["p0a"][head][test_mask] = stored_p0a[test_mask]
            outputs["phase1"][head][test_mask] = phase1_probability[test_mask]
            outputs["final_blend"][head][test_mask] = (
                (1.0 - alpha) * endpoint_probability[test_mask]
                + alpha * phase1_probability[test_mask]
            )
            for mode, output_name in (
                ("visual", "visual_same_epoch"),
                ("history", "history_same_epoch"),
                ("current_only", "current_only_same_epoch"),
            ):
                outputs[output_name][head][test_mask] = torch.softmax(
                    selected[mode][head][test_mask], dim=-1
                )
            selections[head].append(
                {
                    "test_fold": test_fold,
                    "tune_fold": tune_fold,
                    "test_samples": int(test_mask.sum()),
                    "tune_samples": int(tune_mask.sum()),
                    "phase1_epoch": epoch,
                    "phase1_epoch_tuning_trace": epoch_trace,
                    "phase1_weight": alpha,
                    "blend_tuning_trace": alpha_trace,
                    "endpoint_selected_with_replacement": candidate_names,
                    "endpoint_selection_source": (
                        "P0-a selection fitted on this tune fold; raw next_ep logits are used "
                        "on the tune fold instead of reverse-OOF P0-a scores"
                    ),
                    "test_phase1_top5": top_k_recall(
                        phase1_probability[test_mask], labels[head][test_mask], k=5
                    ),
                    "test_p0a_top5": top_k_recall(
                        stored_p0a[test_mask], labels[head][test_mask], k=5
                    ),
                    "test_blend_top5": top_k_recall(
                        outputs["final_blend"][head][test_mask], labels[head][test_mask], k=5
                    ),
                }
            )
    return outputs, selections, max_reconstruction_error


def evaluate_artifacts(
    *,
    endpoint_path: Path,
    p0a_path: Path,
    predictions_dir: Path,
    output_json: Path,
    output_scores: Path,
    expected_last_epoch: int = 10,
    alpha_step: float = 0.05,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if expected_last_epoch < 1:
        raise ValueError("expected_last_epoch must be >= 1")
    if not 0.0 < alpha_step <= 1.0:
        raise ValueError("alpha_step must be in (0, 1]")
    steps = int(round(1.0 / alpha_step))
    if not math.isclose(steps * alpha_step, 1.0, abs_tol=1e-9):
        raise ValueError("alpha_step must divide [0,1] exactly")
    alpha_grid = [round(index * alpha_step, 10) for index in range(steps + 1)]

    endpoint, p0a, phase1, num_classes = _load_and_validate_inputs(
        endpoint_path,
        p0a_path,
        predictions_dir,
        expected_last_epoch=expected_last_epoch,
    )
    outputs, selections, reconstruction_error = _crossfit(
        endpoint, p0a, phase1, num_classes, alpha_grid=alpha_grid
    )
    labels = p0a["labels"]
    video_uids = p0a["video_uids"]
    metrics = {
        name: _all_metrics(scores, labels, num_classes) for name, scores in outputs.items()
    }
    paired = {
        "phase1_vs_p0a": _paired_video_bootstrap(
            outputs["phase1"]["action"],
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
        "history_fused_vs_current_only": _paired_video_bootstrap(
            outputs["phase1"]["action"],
            outputs["current_only_same_epoch"]["action"],
            labels["action"],
            video_uids,
            seed=seed + 2,
            bootstrap_samples=bootstrap_samples,
        ),
    }

    def decision(comparison: dict[str, Any]) -> dict[str, Any]:
        delta = float(comparison["delta_top5_pp"])
        lower = float(comparison["video_bootstrap_95ci_pp"][0])
        passes_engineering_rule = delta > 0.0 and lower > 0.0
        return {
            "criterion": "delta_top5_pp > 0 and video-bootstrap 95% CI lower bound > 0",
            "delta_positive": delta > 0.0,
            "bootstrap_ci_lower_positive": lower > 0.0,
            # The frozen direct epoch-3 source was selected on a 2,000-row
            # subset of this validation split before this evaluator existed.
            # Preserve the engineering decision while refusing to relabel it
            # as an independent confirmatory claim.
            "provisional_engineering_adopted": passes_engineering_rule,
            "confirmatory_adopted": False,
            "gain_at_least_1pp_descriptive_only": delta >= 1.0,
        }

    results: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "Phase-1 crossfit selection and P0-a-aware final ensemble",
        "contract": CONTRACT,
        "sample_count": len(p0a["sample_ids"]),
        "video_count": len(set(video_uids)),
        "num_classes": num_classes,
        "fold_sample_counts": {
            str(fold): int((p0a["folds"] == fold).sum()) for fold in (0, 1)
        },
        "inputs": {
            "endpoint_logits": str(endpoint_path),
            "endpoint_logits_sha256": _sha256(endpoint_path),
            "p0a_oof": str(p0a_path),
            "p0a_oof_sha256": _sha256(p0a_path),
            "phase1_predictions_dir": str(predictions_dir),
            "phase1_epoch_files": {
                str(epoch): {
                    "path": str(predictions_dir / f"epoch_{epoch:02d}.pt"),
                    "sha256": _sha256(predictions_dir / f"epoch_{epoch:02d}.pt"),
                }
                for epoch in sorted(phase1)
            },
        },
        "selection_protocol": {
            "folding": "the exact deterministic video-disjoint two folds from P0-a",
            "phase1_epoch_objective": "field-wise instance Top-5 on the opposite tune fold",
            "blend": "(1-alpha)*P0-a endpoint ensemble + alpha*Phase-1",
            "alpha_grid": alpha_grid,
            "tie_breaks": "earliest epoch; then smaller Phase-1 alpha",
            "leakage_defense": (
                "For each held-out fold, P0-a checkpoint membership, Phase-1 epoch, and alpha "
                "use only the opposite fold. Alpha tuning reconstructs P0-a from raw next_ep "
                "checkpoint logits; it never uses reverse-OOF P0-a tune-fold predictions."
            ),
            "fieldwise": selections,
        },
        "audit": {
            "sample_id_order_exact": True,
            "video_uid_order_exact": True,
            "labels_exact": True,
            "epoch_0_fused_equals_visual_bit_exact": True,
            "p0a_raw_reconstruction_max_abs_error": reconstruction_error,
        },
        "validity_scope": {
            "inherited_validation_adaptivity": True,
            "source": (
                "The frozen visual source is direct next-action best.pt (epoch 3), selected "
                "earlier by Action Top-5 on a seed-42 2,000-sample subset of the same "
                "validation split. Outer-fold selection here cannot undo that prior choice."
            ),
            "bootstrap_scope": (
                "Paired video-cluster bootstrap conditions on the already selected OOF "
                "predictions; it does not repeat model/epoch/ensemble selection."
            ),
            "confirmatory_claim_allowed": False,
            "required_for_confirmatory_claim": (
                "A fresh held-out test set, or nested fold-specific Phase-1/2 training whose "
                "frozen visual recipe is chosen without the held-out fold."
            ),
        },
        "metrics_percent": metrics,
        "paired_action_top5": paired,
        "decisions": {
            "phase1_over_p0a": decision(paired["phase1_vs_p0a"]),
            "final_blend_over_p0a": decision(paired["final_blend_vs_p0a"]),
            "history_intervention_over_current_only": decision(
                paired["history_fused_vs_current_only"]
            ),
            "note": (
                "+1pp is reported as a descriptive practical-effect flag only; it is not a "
                "hard gate. The positive-delta/positive-CI rule is a provisional engineering "
                "decision only because the learned residual inherits earlier validation-based "
                "selection of its frozen epoch-3 visual source."
            ),
        },
    }
    score_artifact = {
        "format_version": FORMAT_VERSION,
        "kind": "goalstep_history_context_crossfit_oof_scores",
        "contract": CONTRACT,
        "sample_ids": p0a["sample_ids"],
        "video_uids": video_uids,
        "folds": p0a["folds"],
        "labels": labels,
        "num_classes": num_classes,
        "scores": outputs,
        "selections": selections,
        "metrics_percent": metrics,
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
        "--predictions-dir",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_k8_vna_ep10/val_predictions"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_k8_vna_ep10/"
            "history_context_vs_p0a_results.json"
        ),
    )
    parser.add_argument(
        "--output-scores",
        type=Path,
        default=Path(
            "outputs/goalstep/runs/z1_history_context_k8_vna_ep10/"
            "history_context_vs_p0a_oof_scores.pt"
        ),
    )
    parser.add_argument("--expected-last-epoch", type=int, default=10)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = evaluate_artifacts(
        endpoint_path=args.endpoint_logits,
        p0a_path=args.p0a_oof,
        predictions_dir=args.predictions_dir,
        output_json=args.output_json,
        output_scores=args.output_scores,
        expected_last_epoch=args.expected_last_epoch,
        alpha_step=args.alpha_step,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    action_metrics = results["metrics_percent"]
    phase1 = action_metrics["phase1"]["action"]
    blend = action_metrics["final_blend"]["action"]
    p0a = action_metrics["p0a"]["action"]
    print(
        "Action Top-5: "
        f"P0-a={p0a['top5']:.3f} Phase-1-OOF={phase1['top5']:.3f} "
        f"final-blend-OOF={blend['top5']:.3f}",
        flush=True,
    )
    print(f"wrote {args.output_json} and {args.output_scores}", flush=True)


if __name__ == "__main__":
    main()
