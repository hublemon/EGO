#!/usr/bin/env python3
"""Train the pre-registered GoalStep Phase-2 history-probe zoo.

The zoo reuses the Phase-1 compact history store and one shared DataLoader.
It trains the eleven LR/WD arms that are not the completed Phase-1 default
arm.  Every arm has independent model, optimizer, and scheduler state, while
each input batch is transferred to the device only once.

The production grid is deliberately closed rather than an arbitrary sweep::

    LR = {1e-4, 3e-4, 1e-3}
    WD = {1e-5, 1e-4, 1e-3, 1e-2}

The default ``(3e-4, 1e-4)`` arm must already exist as the Phase-1 run and is
therefore audited and skipped.  Model-only checkpoints and full-validation
prediction artifacts are retained for every epoch.  A single atomic latest
artifact holds all optimizer/scheduler/RNG state needed to resume the
synchronous zoo without silently mixing provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.exceptions import EgoConfigError  # noqa: E402
from ego.common.io import ensure_dir  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.common.seed import set_seed  # noqa: E402
from ego.step1_action_anticipation.goalstep.train_goalstep_history_context import (  # noqa: E402
    HEADS,
    MODES,
    HistoryContextDataset,
    PreloadedHistoryStore,
    _metric_block,
    _read_index,
    _save_val_predictions,
    _to_device,
    _warmup_cosine_scheduler,
    sigmoid_focal_loss,
)
from ego.step1_action_anticipation.models.history_context_head import (  # noqa: E402
    HistoryContextResidualHead,
)


PHASE = "TrainGoalStepHistoryZoo"
FORMAT_VERSION = 1
# Keep the complete Phase-1 prediction schema. The Phase-2 selector consumes
# fused/current_only; visual/history remain explicit controls and make the
# artifacts directly compatible with the Phase-1 evaluator.
STORED_PREDICTION_MODES = MODES
REGISTERED_LEARNING_RATES = (1e-4, 3e-4, 1e-3)
REGISTERED_WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3, 1e-2)
REGISTERED_DEFAULT = (3e-4, 1e-4)


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    learning_rate: float
    weight_decay: float
    grid_index: int


@dataclass
class ArmRuntime:
    spec: ArmSpec
    model: HistoryContextResidualHead
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _float_list(value: Any, *, location: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise EgoConfigError(f"{location} must be a non-empty YAML list")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0 for item in result):
        raise EgoConfigError(f"{location} values must be finite and non-negative")
    if len(set(result)) != len(result):
        raise EgoConfigError(f"{location} contains duplicate values")
    return result


def _arm_id(learning_rate: float, weight_decay: float) -> str:
    return f"lr_{learning_rate:.0e}__wd_{weight_decay:.0e}".replace("+", "")


def build_registered_grid(config: Mapping[str, Any]) -> tuple[list[ArmSpec], ArmSpec]:
    """Validate and return the eleven new arms plus the skipped default arm."""
    learning_rates = _float_list(
        get(dict(config), "zoo.learning_rates"), location="zoo.learning_rates"
    )
    weight_decays = _float_list(
        get(dict(config), "zoo.weight_decays"), location="zoo.weight_decays"
    )
    default_lr = float(require(dict(config), "zoo.default_phase1_learning_rate"))
    default_wd = float(require(dict(config), "zoo.default_phase1_weight_decay"))
    if int(require(dict(config), "zoo.epochs")) != 10:
        raise EgoConfigError("The pre-registered Phase-2 zoo must run for 10 epochs")
    if int(require(dict(config), "zoo.seed")) != 42:
        raise EgoConfigError("The pre-registered Phase-2 zoo must use seed 42")
    if set(learning_rates) != set(REGISTERED_LEARNING_RATES):
        raise EgoConfigError(
            "Phase-2 LR grid changed from the pre-registered values: "
            f"expected={REGISTERED_LEARNING_RATES}, got={learning_rates}"
        )
    if set(weight_decays) != set(REGISTERED_WEIGHT_DECAYS):
        raise EgoConfigError(
            "Phase-2 WD grid changed from the pre-registered values: "
            f"expected={REGISTERED_WEIGHT_DECAYS}, got={weight_decays}"
        )
    if (default_lr, default_wd) != REGISTERED_DEFAULT:
        raise EgoConfigError(
            "The skipped Phase-1 arm must be exactly LR=3e-4, WD=1e-4"
        )

    all_specs: list[ArmSpec] = []
    grid_index = 0
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            all_specs.append(
                ArmSpec(
                    arm_id=_arm_id(learning_rate, weight_decay),
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    grid_index=grid_index,
                )
            )
            grid_index += 1
    defaults = [
        spec
        for spec in all_specs
        if (spec.learning_rate, spec.weight_decay) == REGISTERED_DEFAULT
    ]
    if len(all_specs) != 12 or len(defaults) != 1:
        raise EgoConfigError("Registered grid must contain 12 arms and one default arm")
    default = defaults[0]
    train_specs = [spec for spec in all_specs if spec != default]
    if len(train_specs) != 11:
        raise EgoConfigError("Phase-2 must train exactly 11 non-default arms")
    return train_specs, default


def _validate_positive_int(value: Any, *, location: str) -> int:
    result = int(value)
    if result < 1:
        raise EgoConfigError(f"{location} must be >= 1")
    return result


def _validate_base_training_contract(base_config: dict[str, Any]) -> dict[str, Any]:
    epochs = _validate_positive_int(require(base_config, "training.epochs"), location="training.epochs")
    if epochs != 10:
        raise EgoConfigError(f"Phase-2 is pre-registered for 10 epochs, got {epochs}")
    seed = int(get(base_config, "experiment.seed", 42))
    if seed != 42:
        raise EgoConfigError(f"Phase-2 requires the Phase-1 seed 42, got {seed}")
    default_lr = float(require(base_config, "training.learning_rate"))
    default_wd = float(require(base_config, "training.weight_decay"))
    if (default_lr, default_wd) != REGISTERED_DEFAULT:
        raise EgoConfigError(
            "Phase-1 source does not match the skipped default arm: "
            f"LR={default_lr}, WD={default_wd}"
        )
    return {
        "epochs": epochs,
        "seed": seed,
        "default_learning_rate": default_lr,
        "default_weight_decay": default_wd,
    }


def _expected_default_prediction_paths(run_dir: Path, epochs: int) -> list[Path]:
    return [run_dir / "val_predictions" / f"epoch_{epoch:02d}.pt" for epoch in range(epochs + 1)]


def _default_phase1_inventory(run_dir: Path, epochs: int) -> dict[str, Any]:
    required_paths = [
        run_dir / "final_metrics.json",
        run_dir / "run_metadata.json",
        run_dir / "checkpoints" / "epoch_00_visual_fallback.pt",
        run_dir / "history_context_vs_p0a_results.json",
        run_dir / "history_context_vs_p0a_oof_scores.pt",
    ]
    required_paths.extend(_expected_default_prediction_paths(run_dir, epochs))
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise EgoConfigError(
            "The default Phase-1 arm must finish before Phase-2 starts; missing "
            f"{missing[0]}"
        )
    return {
        "run_dir": str(run_dir),
        "files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in required_paths
        ],
    }


def _validate_default_predictions(
    run_dir: Path,
    epochs: int,
    *,
    sample_ids: list[str],
    video_uids: list[str],
    labels: dict[str, torch.Tensor],
    num_classes: dict[str, int],
) -> None:
    for epoch, path in enumerate(_expected_default_prediction_paths(run_dir, epochs)):
        artifact = _torch_load(path)
        if artifact.get("format_version") != FORMAT_VERSION:
            raise EgoConfigError(f"{path}: unsupported prediction format")
        if artifact.get("kind") != "goalstep_history_context_val_predictions":
            raise EgoConfigError(f"{path}: unexpected prediction kind")
        if int(artifact.get("epoch", -1)) != epoch:
            raise EgoConfigError(f"{path}: epoch mismatch")
        if artifact.get("sample_ids") != sample_ids or artifact.get("video_uids") != video_uids:
            raise EgoConfigError(f"{path}: validation row alignment mismatch")
        if artifact.get("num_classes") != num_classes:
            raise EgoConfigError(f"{path}: taxonomy mismatch")
        artifact_labels = artifact.get("labels", {})
        artifact_logits = artifact.get("logits", {})
        if (
            set(artifact_labels) != set(HEADS)
            or set(artifact_logits) != set(STORED_PREDICTION_MODES)
        ):
            raise EgoConfigError(f"{path}: label/logit head schema mismatch")
        for head in HEADS:
            if not torch.equal(artifact_labels[head], labels[head]):
                raise EgoConfigError(f"{path}: {head} labels differ from the history index")
        for mode in STORED_PREDICTION_MODES:
            if set(artifact_logits[mode]) != set(HEADS):
                raise EgoConfigError(f"{path}: logits[{mode}] head schema mismatch")
            for head in HEADS:
                value = artifact_logits[mode][head]
                expected = (len(sample_ids), num_classes[head])
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != expected
                    or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                ):
                    raise EgoConfigError(f"{path}: logits[{mode}][{head}] must be fp32 {expected}")


def _validate_default_crossfit(
    run_dir: Path,
    *,
    sample_ids: list[str],
    video_uids: list[str],
    labels: dict[str, torch.Tensor],
) -> None:
    results_path = run_dir / "history_context_vs_p0a_results.json"
    scores_path = run_dir / "history_context_vs_p0a_oof_scores.pt"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("phase") != "Phase-1 crossfit selection and P0-a-aware final ensemble":
        raise EgoConfigError(f"Unexpected Phase-1 crossfit result: {results_path}")
    if int(results.get("sample_count", -1)) != len(sample_ids):
        raise EgoConfigError(f"Phase-1 crossfit sample count mismatch: {results_path}")
    scores = _torch_load(scores_path)
    if scores.get("kind") != "goalstep_history_context_crossfit_oof_scores":
        raise EgoConfigError(f"Unexpected Phase-1 crossfit score kind: {scores_path}")
    if scores.get("sample_ids") != sample_ids or scores.get("video_uids") != video_uids:
        raise EgoConfigError(f"Phase-1 crossfit row alignment mismatch: {scores_path}")
    score_labels = scores.get("labels", {})
    if set(score_labels) != set(HEADS):
        raise EgoConfigError(f"Phase-1 crossfit label schema mismatch: {scores_path}")
    for head in HEADS:
        if not torch.equal(score_labels[head], labels[head]):
            raise EgoConfigError(f"Phase-1 crossfit {head} labels mismatch: {scores_path}")


def _build_model(
    base_config: dict[str, Any],
    *,
    num_classes: dict[str, int],
    embed_dim: int,
    max_history: int,
    device: torch.device,
) -> HistoryContextResidualHead:
    model_config = get(base_config, "model.history", {})
    return HistoryContextResidualHead(
        num_classes=num_classes,
        embed_dim=embed_dim,
        max_history=max_history,
        segment_pooler_heads=int(model_config.get("segment_pooler_heads", 16)),
        transformer_heads=int(model_config.get("transformer_heads", 16)),
        transformer_layers=int(model_config.get("transformer_layers", 2)),
        transformer_mlp_ratio=float(model_config.get("transformer_mlp_ratio", 4.0)),
        transformer_dropout=float(model_config.get("transformer_dropout", 0.1)),
        segment_dropout=float(model_config.get("segment_dropout", 0.3)),
        recency_scale_sec=float(model_config.get("recency_scale_sec", 300.0)),
    ).to(device)


def _make_arms(
    specs: Iterable[ArmSpec],
    base_config: dict[str, Any],
    *,
    num_classes: dict[str, int],
    embed_dim: int,
    max_history: int,
    device: torch.device,
    steps_per_epoch: int,
) -> dict[str, ArmRuntime]:
    epochs = int(require(base_config, "training.epochs"))
    warmup_steps = int(float(get(base_config, "training.warmup_epochs", 1.0)) * steps_per_epoch)
    final_lr = float(get(base_config, "training.final_lr", 0.0))
    specs = list(specs)
    if not specs:
        raise EgoConfigError("Phase-2 has no non-default arms to train")

    # Build one seeded template and copy it to every arm. This makes LR/WD the
    # only between-arm training variables instead of silently confounding the
    # sweep with eleven different random initializations.
    template = _build_model(
        base_config,
        num_classes=num_classes,
        embed_dim=embed_dim,
        max_history=max_history,
        device=device,
    )
    # Creating the ten copies below invokes their random initializers before
    # the template state is loaded. Preserve the post-template RNG position so
    # the zoo's first training dropout mask matches a standalone seeded arm.
    post_template_cpu_rng = torch.get_rng_state()
    post_template_cuda_rng = (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    )
    template_state = {
        key: value.detach().clone() for key, value in template.state_dict().items()
    }
    arms: dict[str, ArmRuntime] = {}
    for position, spec in enumerate(specs):
        if position == 0:
            model = template
        else:
            model = _build_model(
                base_config,
                num_classes=num_classes,
                embed_dim=embed_dim,
                max_history=max_history,
                device=device,
            )
            model.load_state_dict(template_state, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
        )
        final_lr_ratio = final_lr / spec.learning_rate if spec.learning_rate > 0 else 0.0
        scheduler = _warmup_cosine_scheduler(
            optimizer,
            total_steps=max(1, epochs * steps_per_epoch),
            warmup_steps=warmup_steps,
            final_lr_ratio=final_lr_ratio,
        )
        arms[spec.arm_id] = ArmRuntime(spec, model, optimizer, scheduler)
    torch.set_rng_state(post_template_cpu_rng)
    if device.type == "cuda" and post_template_cuda_rng:
        torch.cuda.set_rng_state_all(post_template_cuda_rng)
    return arms


def _model_state_sha256(model: HistoryContextResidualHead) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def audit_identical_initial_states(
    arms: Mapping[str, ArmRuntime],
    *,
    default_phase1_epoch0_checkpoint: Path,
) -> dict[str, Any]:
    """Fail closed if any arm starts from weights different from arm zero."""
    if not arms:
        raise EgoConfigError("Cannot audit an empty arm collection")
    first_arm_id = next(iter(arms))
    reference = arms[first_arm_id].model.state_dict()
    for arm_id, arm in arms.items():
        candidate = arm.model.state_dict()
        if candidate.keys() != reference.keys():
            raise EgoConfigError(f"Initial state keys differ for {arm_id}")
        for key in reference:
            if not torch.equal(reference[key], candidate[key]):
                raise EgoConfigError(
                    f"Initial weight confound detected: {arm_id} differs at {key}"
                )
    default_checkpoint = _torch_load(default_phase1_epoch0_checkpoint)
    default_state = default_checkpoint.get("model_state", {})
    if default_state.keys() != reference.keys():
        raise EgoConfigError(
            "The default Phase-1 epoch-0 state keys differ from the zoo architecture"
        )
    for key in reference:
        if not torch.equal(reference[key].detach().cpu(), default_state[key].detach().cpu()):
            raise EgoConfigError(
                "Zoo initialization differs from the reused default Phase-1 arm at "
                f"{key}"
            )
    return {
        "identical": True,
        "matches_default_phase1_epoch0": True,
        "default_phase1_epoch0_checkpoint": str(default_phase1_epoch0_checkpoint),
        "default_phase1_epoch0_checkpoint_sha256": _sha256(
            default_phase1_epoch0_checkpoint
        ),
        "reference_arm_id": first_arm_id,
        "arm_count": len(arms),
        "state_sha256": _model_state_sha256(arms[first_arm_id].model),
        "confounded_variables": [],
        "varied_variables": ["learning_rate", "weight_decay"],
        "shared_per_batch_stochastic_masks": True,
    }


def _forward(
    model: HistoryContextResidualHead, inputs: dict[str, Any]
) -> dict[str, dict[str, torch.Tensor]]:
    return model(
        inputs["summaries"],
        inputs["history_mask"],
        inputs["history_delta_t_sec"],
        inputs["history_level_id"],
        inputs["visual_logits"],
    )


def train_shared_epoch(
    arms: Mapping[str, ArmRuntime],
    loader: DataLoader,
    device: torch.device,
    *,
    focal_alpha: float,
    focal_gamma: float,
    history_aux_weight: float,
    amp_dtype: torch.dtype | None,
    gradient_clip_norm: float | None,
) -> dict[str, dict[str, float]]:
    """Train all arms from each shared host batch before loading the next one."""
    for arm in arms.values():
        arm.model.train()
    totals = {
        arm_id: {"loss": 0.0, "fused_loss": 0.0, "history_aux_loss": 0.0, "samples": 0}
        for arm_id in arms
    }
    for batch in loader:
        batch_size = len(batch["sample_id"])
        inputs = _to_device(batch, device)
        # Give every arm the same stochastic dropout/segment-dropout masks.
        # Restore the state advanced by one arm afterward, matching a single
        # standalone run while avoiding stochasticity as an LR/WD confound.
        shared_cpu_rng = torch.get_rng_state()
        shared_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        advanced_cpu_rng: torch.Tensor | None = None
        advanced_cuda_rng: list[torch.Tensor] | None = None
        for arm_position, (arm_id, arm) in enumerate(arms.items()):
            torch.set_rng_state(shared_cpu_rng)
            if device.type == "cuda" and shared_cuda_rng:
                torch.cuda.set_rng_state_all(shared_cuda_rng)
            arm.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.bfloat16,
                enabled=device.type == "cuda" and amp_dtype is not None,
            ):
                outputs = _forward(arm.model, inputs)
                fused_loss = sum(
                    sigmoid_focal_loss(
                        outputs["fused"][head],
                        inputs["labels"][head],
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                    for head in HEADS
                )
                history_aux_loss = sum(
                    sigmoid_focal_loss(
                        outputs["history"][head],
                        inputs["labels"][head],
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                    for head in HEADS
                )
                loss = fused_loss + history_aux_weight * history_aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm_id}: non-finite training loss")
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(arm.model.parameters(), gradient_clip_norm)
            arm.optimizer.step()
            arm.scheduler.step()
            if arm_position == 0:
                advanced_cpu_rng = torch.get_rng_state()
                advanced_cuda_rng = (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                )
            totals[arm_id]["loss"] += float(loss.detach().cpu()) * batch_size
            totals[arm_id]["fused_loss"] += float(fused_loss.detach().cpu()) * batch_size
            totals[arm_id]["history_aux_loss"] += (
                float(history_aux_loss.detach().cpu()) * batch_size
            )
            totals[arm_id]["samples"] += batch_size
        if advanced_cpu_rng is None:
            raise RuntimeError("Shared zoo batch contained no arms")
        torch.set_rng_state(advanced_cpu_rng)
        if device.type == "cuda" and advanced_cuda_rng:
            torch.cuda.set_rng_state_all(advanced_cuda_rng)
    return {
        arm_id: {
            key: values[key] / max(1, int(values["samples"]))
            for key in ("loss", "fused_loss", "history_aux_loss")
        }
        for arm_id, values in totals.items()
    }


@torch.inference_mode()
def evaluate_shared(
    arms: Mapping[str, ArmRuntime],
    loader: DataLoader,
    device: torch.device,
    num_classes: dict[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate every arm while iterating over the validation loader once."""
    for arm in arms.values():
        arm.model.eval()
    collected = {
        arm_id: {
            mode: {head: [] for head in HEADS}
            for mode in STORED_PREDICTION_MODES
        }
        for arm_id in arms
    }
    labels_all = {head: [] for head in HEADS}
    sample_ids: list[str] = []
    video_uids: list[str] = []
    history_lengths: list[int] = []
    for batch in loader:
        inputs = _to_device(batch, device)
        for arm_id, arm in arms.items():
            outputs = _forward(arm.model, inputs)
            for mode in STORED_PREDICTION_MODES:
                for head in HEADS:
                    collected[arm_id][mode][head].append(outputs[mode][head].float().cpu())
        for head in HEADS:
            labels_all[head].append(inputs["labels"][head].cpu())
        sample_ids.extend(str(value) for value in batch["sample_id"])
        video_uids.extend(str(value) for value in batch["video_uid"])
        history_lengths.extend(int(value) for value in batch["history_length"])

    labels = {head: torch.cat(parts) for head, parts in labels_all.items()}
    predictions: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    for arm_id, arm in arms.items():
        logits = {
            mode: {head: torch.cat(collected[arm_id][mode][head]) for head in HEADS}
            for mode in STORED_PREDICTION_MODES
        }
        predictions[arm_id] = {
            "sample_ids": list(sample_ids),
            "video_uids": list(video_uids),
            "history_lengths": torch.tensor(history_lengths, dtype=torch.long),
            "labels": {head: value.clone() for head, value in labels.items()},
            "logits": logits,
        }
        metrics[arm_id] = {
            "size": len(sample_ids),
            "overall": {
                mode: {
                    head: _metric_block(logits[mode][head], labels[head], num_classes[head])
                    for head in HEADS
                }
                for mode in STORED_PREDICTION_MODES
            },
            "gate_values": arm.model.gate_values(),
        }
    return metrics, predictions


def _model_checkpoint(
    arm: ArmRuntime,
    *,
    epoch: int,
    metrics: dict[str, Any],
    provenance_fingerprint: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": "goalstep_history_probe_zoo_model_only",
        "epoch": int(epoch),
        "arm_id": arm.spec.arm_id,
        "learning_rate": arm.spec.learning_rate,
        "weight_decay": arm.spec.weight_decay,
        "model_state": arm.model.state_dict(),
        "metrics": metrics,
        "gate_values": arm.model.gate_values(),
        "provenance_fingerprint": provenance_fingerprint,
        "optimizer_state_included": False,
    }


def _capture_rng_state(
    train_generator: torch.Generator, *, include_cuda: bool
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Keep the resume artifact compatible with torch's restricted
        # ``weights_only`` loader: do not pickle a NumPy ndarray object.
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if include_cuda else [],
        "train_generator": train_generator.get_state(),
    }


def _restore_rng_state(state: Mapping[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])


def _latest_resume_state(
    arms: Mapping[str, ArmRuntime],
    *,
    epoch: int,
    provenance_fingerprint: str,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    first_device = next(iter(arms.values())).model.field_gates["action"].device
    return {
        "format_version": FORMAT_VERSION,
        "kind": "goalstep_history_probe_zoo_latest_resume",
        "epoch": int(epoch),
        "provenance_fingerprint": provenance_fingerprint,
        "arms": {
            arm_id: {
                "model_state": arm.model.state_dict(),
                "optimizer_state": arm.optimizer.state_dict(),
                "scheduler_state": arm.scheduler.state_dict(),
            }
            for arm_id, arm in arms.items()
        },
        "rng_state": _capture_rng_state(
            train_generator, include_cuda=first_device.type == "cuda"
        ),
    }


def _load_resume(
    path: Path,
    arms: Mapping[str, ArmRuntime],
    *,
    provenance_fingerprint: str,
    train_generator: torch.Generator,
) -> int:
    state = _torch_load(path)
    if state.get("kind") != "goalstep_history_probe_zoo_latest_resume":
        raise EgoConfigError(f"Unexpected resume artifact kind: {path}")
    if state.get("provenance_fingerprint") != provenance_fingerprint:
        raise EgoConfigError("Zoo resume provenance differs from the current inputs/config")
    if set(state.get("arms", {})) != set(arms):
        raise EgoConfigError("Zoo resume arm set differs from the registered grid")
    epoch = int(state.get("epoch", -1))
    if epoch < 0:
        raise EgoConfigError(f"Invalid resume epoch in {path}")
    for arm_id, arm in arms.items():
        arm_state = state["arms"][arm_id]
        arm.model.load_state_dict(arm_state["model_state"], strict=True)
        arm.optimizer.load_state_dict(arm_state["optimizer_state"])
        arm.scheduler.load_state_dict(arm_state["scheduler_state"])
    _restore_rng_state(state["rng_state"], train_generator)
    return epoch


def _validate_committed_epoch(
    run_dir: Path,
    arms: Mapping[str, ArmRuntime],
    *,
    epoch: int,
    provenance_fingerprint: str,
) -> None:
    if epoch == 0:
        return
    for arm_id in arms:
        checkpoint_path = run_dir / "arms" / arm_id / "checkpoints" / f"epoch_{epoch:02d}.pt"
        prediction_path = run_dir / "arms" / arm_id / "val_predictions" / f"epoch_{epoch:02d}.pt"
        if not checkpoint_path.is_file() or not prediction_path.is_file():
            raise EgoConfigError(
                f"Resume epoch {epoch} is not fully committed for {arm_id}"
            )
        checkpoint = _torch_load(checkpoint_path)
        if (
            checkpoint.get("kind") != "goalstep_history_probe_zoo_model_only"
            or checkpoint.get("provenance_fingerprint") != provenance_fingerprint
            or int(checkpoint.get("epoch", -1)) != epoch
        ):
            raise EgoConfigError(f"Invalid committed checkpoint: {checkpoint_path}")


def _write_history_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fields = [
        "epoch",
        "arm_id",
        "learning_rate",
        "weight_decay",
        "train_loss",
        "fused_action_cmr@5",
        "fused_action_top1",
        "fused_action_top5",
        "fused_action_top10",
        "fused_action_top15",
        "current_only_action_top5",
        "gate_verb",
        "gate_noun",
        "gate_action",
        "epoch_seconds",
    ]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({field: record[field] for field in fields})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_zoo_val_predictions(
    path: Path,
    *,
    arm: ArmRuntime,
    epoch: int,
    predictions: dict[str, Any],
    num_classes: dict[str, int],
    gate_values: dict[str, Any],
    provenance_fingerprint: str,
) -> None:
    """Save the Phase-1-compatible prediction schema plus zoo provenance."""
    staging = path.with_name(f".{path.name}.schema.tmp.{os.getpid()}")
    try:
        _save_val_predictions(
            staging,
            epoch=epoch,
            predictions=predictions,
            num_classes=num_classes,
            gate_values=gate_values,
        )
        artifact = _torch_load(staging)
        artifact.update(
            {
                "phase": "P2",
                "arm_id": arm.spec.arm_id,
                "learning_rate": arm.spec.learning_rate,
                "weight_decay": arm.spec.weight_decay,
                "provenance_fingerprint": provenance_fingerprint,
            }
        )
        _atomic_torch_save(artifact, path)
    finally:
        staging.unlink(missing_ok=True)


def _records_for_resume(path: Path, resume_epoch: int) -> list[dict[str, Any]]:
    if resume_epoch == 0:
        return []
    if not path.is_file():
        raise EgoConfigError(f"Resume artifact exists but training history is missing: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise EgoConfigError(f"Invalid training-history JSON: {path}")
    retained = [record for record in records if int(record.get("epoch", -1)) <= resume_epoch]
    expected = resume_epoch * 11
    if len(retained) != expected:
        raise EgoConfigError(
            f"Resume history has {len(retained)} committed arm-epochs; expected {expected}"
        )
    return retained


def run_zoo(
    zoo_config: dict[str, Any],
    *,
    config_path: Path,
    allow_resume: bool,
    _test_stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    specs, default_spec = build_registered_grid(zoo_config)
    base_config_path = expand_path(require(zoo_config, "source.phase1_config"))
    base_config = load_config(base_config_path)
    base_contract = _validate_base_training_contract(base_config)
    epochs = int(base_contract["epochs"])
    stop_epoch = epochs if _test_stop_after_epoch is None else int(_test_stop_after_epoch)
    if not 1 <= stop_epoch <= epochs:
        raise EgoConfigError("Internal stop epoch must be within the registered 10 epochs")

    seed = int(base_contract["seed"])
    set_seed(seed)
    device_name = str(get(zoo_config, "experiment.device", get(base_config, "experiment.device", "cuda")))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    run_dir = expand_path(require(zoo_config, "experiment.output_dir"))
    ensure_dir(run_dir)

    store_root = expand_path(require(base_config, "dataset.derived_store_dir"))
    index_dir = expand_path(require(base_config, "dataset.history_index_dir"))
    store_manifest_path = store_root / "manifest.json"
    train_frame, train_index_path = _read_index(index_dir, "train")
    val_frame, val_index_path = _read_index(index_dir, "val")
    if not store_manifest_path.is_file():
        raise FileNotFoundError(store_manifest_path)
    verify_hashes = bool(get(base_config, "dataset.verify_shard_hashes", False))
    train_store = PreloadedHistoryStore(store_root, "train", verify_shard_hashes=verify_hashes)
    val_store = PreloadedHistoryStore(store_root, "val", verify_shard_hashes=verify_hashes)
    if train_store.num_classes != val_store.num_classes:
        raise EgoConfigError("Train/val derived-store taxonomy mismatch")
    if (train_store.summary_tokens, train_store.embed_dim) != (
        val_store.summary_tokens,
        val_store.embed_dim,
    ):
        raise EgoConfigError("Train/val derived-store summary shape mismatch")
    max_history = int(get(base_config, "dataset.max_history", 8))
    train_dataset = HistoryContextDataset(train_frame, train_store, max_history)
    val_dataset = HistoryContextDataset(val_frame, val_store, max_history)
    expected_train = int(get(zoo_config, "dataset.expected_train_rows", 29293))
    expected_val = int(get(zoo_config, "dataset.expected_val_rows", 6960))
    if len(train_dataset) != expected_train or len(val_dataset) != expected_val:
        raise EgoConfigError(
            "Phase-2 cohort count mismatch: "
            f"train={len(train_dataset)} (expected {expected_train}), "
            f"val={len(val_dataset)} (expected {expected_val})"
        )

    default_run_dir = expand_path(require(zoo_config, "source.default_phase1_run_dir"))
    default_inventory = _default_phase1_inventory(default_run_dir, epochs)
    val_labels = {
        head: torch.tensor(val_frame[f"{head}_id"].to_numpy(), dtype=torch.long)
        for head in HEADS
    }
    val_sample_ids = val_frame["sample_id"].astype(str).tolist()
    val_video_uids = val_frame["video_uid"].astype(str).tolist()
    _validate_default_predictions(
        default_run_dir,
        epochs,
        sample_ids=val_sample_ids,
        video_uids=val_video_uids,
        labels=val_labels,
        num_classes=val_store.num_classes,
    )
    _validate_default_crossfit(
        default_run_dir,
        sample_ids=val_sample_ids,
        video_uids=val_video_uids,
        labels=val_labels,
    )

    provenance = {
        "format_version": FORMAT_VERSION,
        "kind": "goalstep_history_probe_zoo_provenance",
        "contract": "A2.end-1s plus K=8 completed visual history -> strict same-level A3",
        "feature_reextraction": False,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "phase1_config": {"path": str(base_config_path), "sha256": _sha256(base_config_path)},
        "store_manifest": {
            "path": str(store_manifest_path),
            "sha256": _sha256(store_manifest_path),
        },
        "indices": {
            "train": {"path": str(train_index_path), "sha256": _sha256(train_index_path)},
            "val": {"path": str(val_index_path), "sha256": _sha256(val_index_path)},
        },
        "default_phase1": default_inventory,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "num_classes": val_store.num_classes,
        "summary_shape": [train_store.summary_tokens, train_store.embed_dim],
        "max_history": max_history,
        "seed": seed,
        "epochs": epochs,
        "registered_grid": [spec.__dict__ for spec in [*specs, default_spec]],
        "skipped_default_arm": default_spec.__dict__,
    }
    provenance_fingerprint = _fingerprint(provenance)
    manifest = {
        **provenance,
        "provenance_fingerprint": provenance_fingerprint,
        "trained_arm_count": len(specs),
        "total_grid_arm_count": len(specs) + 1,
        "shared_store": True,
        "shared_dataloader": True,
        "checkpoint_policy": "model-only every epoch; optimizer/scheduler/RNG in atomic latest",
        "default_arm_reused_not_retrained": True,
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("provenance_fingerprint") != provenance_fingerprint:
            raise EgoConfigError(
                "Existing Phase-2 run has different provenance; choose a fresh output directory"
            )
    else:
        unexplained = [
            path
            for path in run_dir.iterdir()
            if path.name not in {"logs"} and path != manifest_path
        ]
        if unexplained:
            raise EgoConfigError(
                "Phase-2 output has artifacts but no run_manifest.json: "
                f"{unexplained[0]}"
            )
        _atomic_json(manifest, manifest_path)
    ensure_dir(run_dir / "arms")

    batch_size = _validate_positive_int(
        require(base_config, "training.batch_size"), location="training.batch_size"
    )
    eval_batch_size = _validate_positive_int(
        get(base_config, "training.eval_batch_size", batch_size),
        location="training.eval_batch_size",
    )
    num_workers = int(get(base_config, "dataset.num_workers", 0))
    if num_workers < 0:
        raise EgoConfigError("dataset.num_workers must be >= 0")
    train_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    arms = _make_arms(
        specs,
        base_config,
        num_classes=train_store.num_classes,
        embed_dim=train_store.embed_dim,
        max_history=max_history,
        device=device,
        steps_per_epoch=len(train_loader),
    )
    initialization_audit = {
        **audit_identical_initial_states(
            arms,
            default_phase1_epoch0_checkpoint=(
                default_run_dir / "checkpoints" / "epoch_00_visual_fallback.pt"
            ),
        ),
        "provenance_fingerprint": provenance_fingerprint,
        "seed": seed,
        "note": (
            "All arms copied from one seeded template before optimizer construction; "
            "training restores a shared per-batch RNG state for identical dropout masks"
        ),
    }
    initialization_audit_path = run_dir / "initialization_audit.json"
    if initialization_audit_path.is_file():
        existing_initialization = json.loads(
            initialization_audit_path.read_text(encoding="utf-8")
        )
        if existing_initialization != initialization_audit:
            raise EgoConfigError(
                "Initial-state audit changed for an existing zoo run; fail closed"
            )
    else:
        _atomic_json(initialization_audit, initialization_audit_path)

    precision = str(get(base_config, "training.precision", "bf16" if device.type == "cuda" else "fp32"))
    if precision not in ("fp32", "bf16"):
        raise EgoConfigError("training.precision must be fp32 or bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" and device.type == "cuda" else None
    focal_alpha = float(get(base_config, "training.focal_alpha", 0.25))
    focal_gamma = float(get(base_config, "training.focal_gamma", 2.0))
    history_aux_weight = float(get(base_config, "training.history_aux_weight", 0.25))
    gradient_clip_raw = get(base_config, "training.gradient_clip_norm", 1.0)
    gradient_clip_norm = None if gradient_clip_raw is None else float(gradient_clip_raw)

    latest_path = run_dir / "latest_resume.pt"
    history_json_path = run_dir / "metrics_per_arm_epoch.json"
    history_csv_path = run_dir / "training_history.csv"
    if latest_path.is_file():
        if not allow_resume:
            raise EgoConfigError(
                f"Existing resume state found at {latest_path}; rerun with --resume"
            )
        resume_epoch = _load_resume(
            latest_path,
            arms,
            provenance_fingerprint=provenance_fingerprint,
            train_generator=train_generator,
        )
        if resume_epoch > epochs:
            raise EgoConfigError(f"Resume epoch {resume_epoch} exceeds configured epochs {epochs}")
        _validate_committed_epoch(
            run_dir,
            arms,
            epoch=resume_epoch,
            provenance_fingerprint=provenance_fingerprint,
        )
        records = _records_for_resume(history_json_path, resume_epoch)
    else:
        existing_training_artifacts = [
            path
            for pattern in (
                "arms/*/checkpoints/epoch_*.pt",
                "arms/*/val_predictions/epoch_*.pt",
                "metrics_per_arm_epoch.json",
                "training_history.csv",
                "final_metrics.json",
            )
            for path in run_dir.glob(pattern)
        ]
        if existing_training_artifacts:
            raise EgoConfigError(
                "Training artifacts exist without an atomic latest resume artifact; "
                "use a fresh output directory"
            )
        resume_epoch = 0
        records = []

    if resume_epoch >= stop_epoch:
        status = "complete" if resume_epoch == epochs else "stopped_for_smoke"
        return {
            "status": status,
            "completed_epoch": resume_epoch,
            "epochs": epochs,
            "trained_arm_count": len(arms),
            "provenance_fingerprint": provenance_fingerprint,
        }

    for epoch in range(resume_epoch + 1, stop_epoch + 1):
        started = time.time()
        train_metrics = train_shared_epoch(
            arms,
            train_loader,
            device,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            history_aux_weight=history_aux_weight,
            amp_dtype=amp_dtype,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_metrics, val_predictions = evaluate_shared(
            arms, val_loader, device, train_store.num_classes
        )
        elapsed = time.time() - started
        for arm_id, arm in arms.items():
            arm_dir = run_dir / "arms" / arm_id
            _atomic_torch_save(
                _model_checkpoint(
                    arm,
                    epoch=epoch,
                    metrics=val_metrics[arm_id],
                    provenance_fingerprint=provenance_fingerprint,
                ),
                arm_dir / "checkpoints" / f"epoch_{epoch:02d}.pt",
            )
            _save_zoo_val_predictions(
                arm_dir / "val_predictions" / f"epoch_{epoch:02d}.pt",
                arm=arm,
                epoch=epoch,
                predictions=val_predictions[arm_id],
                num_classes=train_store.num_classes,
                gate_values=val_metrics[arm_id]["gate_values"],
                provenance_fingerprint=provenance_fingerprint,
            )
            action = val_metrics[arm_id]["overall"]["fused"]["action"]
            records.append(
                {
                    "epoch": epoch,
                    "arm_id": arm_id,
                    "learning_rate": arm.spec.learning_rate,
                    "weight_decay": arm.spec.weight_decay,
                    "train_loss": train_metrics[arm_id]["loss"],
                    "train_fused_loss": train_metrics[arm_id]["fused_loss"],
                    "train_history_aux_loss": train_metrics[arm_id]["history_aux_loss"],
                    "fused_action_cmr@5": action["cmr@5"],
                    "fused_action_top1": action["top1"],
                    "fused_action_top5": action["top5"],
                    "fused_action_top10": action["top10"],
                    "fused_action_top15": action["top15"],
                    "current_only_action_top5": val_metrics[arm_id]["overall"]["current_only"]["action"]["top5"],
                    "gate_verb": val_metrics[arm_id]["gate_values"]["verb"]["tanh"],
                    "gate_noun": val_metrics[arm_id]["gate_values"]["noun"]["tanh"],
                    "gate_action": val_metrics[arm_id]["gate_values"]["action"]["tanh"],
                    "epoch_seconds": elapsed,
                }
            )
        _atomic_json(records, history_json_path)
        _write_history_csv(history_csv_path, records)
        # This atomic artifact is the commit marker for the synchronous epoch.
        # Any model/prediction files from an interrupted later epoch are safe
        # to overwrite after restoring this state.
        _atomic_torch_save(
            _latest_resume_state(
                arms,
                epoch=epoch,
                provenance_fingerprint=provenance_fingerprint,
                train_generator=train_generator,
            ),
            latest_path,
        )
        summary = ", ".join(
            f"{arm_id}={val_metrics[arm_id]['overall']['fused']['action']['top5']:.2f}"
            for arm_id in arms
        )
        print(
            f"[{PHASE}] epoch {epoch}/{epochs} seconds={elapsed:.1f} action_top5: {summary}",
            flush=True,
        )

    status = "complete" if stop_epoch == epochs else "stopped_for_smoke"
    final = {
        "format_version": FORMAT_VERSION,
        "kind": "goalstep_history_probe_zoo_result",
        "status": status,
        "completed_epoch": stop_epoch,
        "epochs": epochs,
        "trained_arm_count": len(arms),
        "skipped_default_arm": default_spec.__dict__,
        "registered_grid": [spec.__dict__ for spec in [*specs, default_spec]],
        "provenance_fingerprint": provenance_fingerprint,
        "selection_status": "deferred_to_cross_fitted_champion_evaluator",
        "adoption_rule": "delta_top5_pp > 0 and video_bootstrap_95ci_lower_pp > 0",
        "material_gain_pp_is_descriptive_only": 1.0,
    }
    if status == "complete":
        _atomic_json(final, run_dir / "final_metrics.json")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/step1/goalstep/z1_history_context_probe_zoo_ep10.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the atomic latest artifact has identical provenance",
    )
    args = parser.parse_args()
    config_path = expand_path(args.config)
    result = run_zoo(
        load_config(config_path),
        config_path=config_path,
        allow_resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
