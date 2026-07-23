#!/usr/bin/env python3
"""Train the GoalStep Phase-1 visual-history probe.

This trainer consumes a compact *derived* store.  It never decodes video and
never runs V-JEPA.  See :class:`HistoryContextDataset` for the exact store and
history-index contracts enforced before training begins.

Usage:
    PYTHONPATH=src python \
      src/ego/step1_action_anticipation/goalstep/train_goalstep_history_context.py \
      --config configs/step1/goalstep/z1_history_context_k8_vna_ep10.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Permit direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.exceptions import EgoConfigError  # noqa: E402
from ego.common.io import ensure_dir, write_json, write_yaml  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.common.seed import set_seed  # noqa: E402
from ego.step1_action_anticipation.metrics import (  # noqa: E402
    class_mean_recall,
    top_k_recall,
)
from ego.step1_action_anticipation.models.history_context_head import (  # noqa: E402
    HEADS,
    HistoryContextResidualHead,
)


PHASE = "TrainGoalStepHistory"
MODES = ("visual", "history", "current_only", "fused")
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    """Use the restricted tensor loader when supported by this PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old PyTorch fallback
        return torch.load(path, map_location="cpu")


def _read_index(index_dir: Path, split: str) -> tuple[pd.DataFrame, Path]:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = index_dir / f"{split}{suffix}"
        if path.is_file():
            return reader(path).reset_index(drop=True), path
    raise FileNotFoundError(f"No {split}.parquet or {split}.csv under {index_dir}")


def _strict_bool(value: Any, *, location: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise EgoConfigError(f"{location}: expected a boolean mask, got {value!r}")


def _load_phase0_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    """Audit P0-b provenance without using its outcome to block Phase 1.

    The original recipe treated ``observed >= 27.7`` as a hard go/no-go
    gate.  That threshold was a hand-set compute heuristic, not a statistical
    dependency of the history model.  P0-b is therefore retained as an
    immutable diagnostic artifact, while Phase 1 runs regardless of PASS or
    FAIL.  A malformed or missing artifact still fails closed because that is
    a provenance problem rather than a performance decision.
    """
    policy = str(get(config, "phase0.policy", "diagnostic_only"))
    if policy != "diagnostic_only":
        raise EgoConfigError(
            "phase0.policy must be 'diagnostic_only'; P0-b is no longer a Phase-1 gate"
        )
    historical_threshold = float(
        get(config, "phase0.historical_gate_threshold_action_top5", 27.7)
    )
    artifact_path = expand_path(require(config, "phase0.gate_results_path"))
    if not artifact_path.is_file():
        raise EgoConfigError(
            f"P0-b diagnostic artifact does not exist: {artifact_path}"
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    gate = artifact.get("gate", {})
    observed = float(gate.get("observed_percent", float("nan")))
    artifact_threshold = float(gate.get("threshold_percent", float("nan")))
    artifact_passed = gate.get("passed") is True
    if artifact.get("phase") != "P0-b":
        raise EgoConfigError(f"Unexpected Phase-0 gate artifact phase in {artifact_path}")
    if not math.isfinite(observed) or not math.isfinite(artifact_threshold):
        raise EgoConfigError(f"Non-finite Phase-0 gate metric in {artifact_path}")
    if abs(artifact_threshold - historical_threshold) > 1e-9:
        raise EgoConfigError(
            f"Phase-0 registered threshold mismatch: artifact={artifact_threshold}, "
            f"config={historical_threshold}"
        )
    computed_pass = observed >= artifact_threshold
    if artifact_passed != computed_pass:
        raise EgoConfigError(
            "P0-b artifact has an internally inconsistent PASS/FAIL value: "
            f"observed={observed:.3f}, threshold={artifact_threshold:.3f}, "
            f"artifact_passed={artifact_passed}"
        )
    return {
        "policy": "diagnostic_only",
        "blocks_phase1": False,
        "historical_gate_passed": artifact_passed,
        "metric": str(gate.get("metric", "Action OOF instance Top-5 accuracy")),
        "observed_oof_action_top5": observed,
        "historical_gate_threshold_action_top5": historical_threshold,
        "artifact": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
    }


def _atomic_torch_save(record: dict[str, Any], path: Path) -> None:
    """Publish a torch artifact atomically so readers never see partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(record, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PreloadedHistoryStore:
    """Read a sharded derived store into compact CPU tensors once per split.

    Required root ``manifest.json`` fields::

        schema_version: 1
        kind: goalstep_history_context_derived_store
        summary_shape: [17, 1024]
        num_classes: {verb: V, noun: N, action: A}
        splits.<split>.rows: N
        splits.<split>.shards[*].path: <root-relative shard path>

    Every shard is a ``torch.save`` dictionary containing ``sample_ids``,
    fp16 ``summaries [N,17,1024]``, and fp32 dictionaries
    ``visual_logits`` / ``recognition_logits`` keyed by V/N/A.  Recognition
    logits are schema-validated but deliberately not exposed to Phase 1.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        verify_shard_hashes: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.manifest_path = root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise EgoConfigError(
                f"Unsupported history-store schema_version={self.manifest.get('schema_version')!r}"
            )
        if self.manifest.get("kind") != "goalstep_history_context_derived_store":
            raise EgoConfigError(f"Unexpected derived-store kind in {self.manifest_path}")
        if self.manifest.get("backbone_reextraction") is not False:
            raise EgoConfigError(
                "Phase-1 store must declare backbone_reextraction=false; it should derive from existing cache"
            )
        if split not in self.manifest.get("splits", {}):
            raise EgoConfigError(f"Derived store has no {split!r} split")

        summary_shape = tuple(int(value) for value in self.manifest["summary_shape"])
        if len(summary_shape) != 2 or any(value <= 0 for value in summary_shape):
            raise EgoConfigError(f"Invalid summary_shape={summary_shape}")
        self.summary_tokens, self.embed_dim = summary_shape
        raw_classes = self.manifest.get("num_classes", {})
        if set(raw_classes) != set(HEADS):
            raise EgoConfigError(f"Store num_classes must contain exactly {HEADS}")
        self.num_classes = {head: int(raw_classes[head]) for head in HEADS}

        split_manifest = self.manifest["splits"][split]
        top_provenance_fingerprint = self.manifest.get("provenance_base_fingerprint")
        if not top_provenance_fingerprint:
            raise EgoConfigError("Derived store is missing provenance_base_fingerprint")
        if split_manifest.get("provenance_base_fingerprint") != top_provenance_fingerprint:
            raise EgoConfigError(
                f"Derived-store {split} provenance differs from the top-level manifest"
            )
        expected_rows = int(split_manifest["rows"])
        shard_entries = split_manifest.get("shards", [])
        if not shard_entries:
            raise EgoConfigError(f"No shards listed for derived-store split {split!r}")

        self.summaries = torch.empty(
            expected_rows,
            self.summary_tokens,
            self.embed_dim,
            dtype=torch.float16,
        )
        self.visual_logits = {
            head: torch.empty(expected_rows, classes, dtype=torch.float32)
            for head, classes in self.num_classes.items()
        }
        sample_ids: list[str] = []
        cursor = 0
        for shard_entry in shard_entries:
            shard_path = root / shard_entry["path"]
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if verify_shard_hashes and shard_entry.get("sha256"):
                actual_hash = _sha256(shard_path)
                if actual_hash != shard_entry["sha256"]:
                    raise EgoConfigError(f"Shard hash mismatch: {shard_path}")
            record = _torch_load(shard_path)
            expected_shard_fingerprint = shard_entry.get("provenance_fingerprint")
            if not expected_shard_fingerprint or (
                record.get("provenance_fingerprint") != expected_shard_fingerprint
            ):
                raise EgoConfigError(f"Shard provenance mismatch: {shard_path}")
            if record.get("schema_version") != SCHEMA_VERSION:
                raise EgoConfigError(f"{shard_path}: unsupported shard schema version")
            ids = [str(value) for value in record.get("sample_ids", [])]
            summaries = record.get("summaries")
            count = len(ids)
            if int(shard_entry.get("rows", count)) != count:
                raise EgoConfigError(f"{shard_path}: shard manifest row count mismatch")
            if "start" in shard_entry and int(shard_entry["start"]) != cursor:
                raise EgoConfigError(f"{shard_path}: shard start offset mismatch")
            if "stop" in shard_entry and int(shard_entry["stop"]) != cursor + count:
                raise EgoConfigError(f"{shard_path}: shard stop offset mismatch")
            expected_summary_shape = (count, self.summary_tokens, self.embed_dim)
            if not torch.is_tensor(summaries) or tuple(summaries.shape) != expected_summary_shape:
                raise EgoConfigError(
                    f"{shard_path}: summaries must be {expected_summary_shape}"
                )
            if summaries.dtype != torch.float16:
                raise EgoConfigError(f"{shard_path}: summaries must be fp16")
            if not torch.isfinite(summaries).all():
                raise EgoConfigError(f"{shard_path}: non-finite summaries")
            if cursor + count > expected_rows:
                raise EgoConfigError(f"{shard_path}: shard rows exceed manifest row count")

            for dictionary_name in ("visual_logits", "recognition_logits"):
                dictionary = record.get(dictionary_name, {})
                if set(dictionary) != set(HEADS):
                    raise EgoConfigError(
                        f"{shard_path}: {dictionary_name} must contain exactly {HEADS}"
                    )
                for head in HEADS:
                    value = dictionary[head]
                    shape = (count, self.num_classes[head])
                    if not torch.is_tensor(value) or tuple(value.shape) != shape:
                        raise EgoConfigError(
                            f"{shard_path}: {dictionary_name}[{head}] must be {shape}"
                        )
                    if value.dtype != torch.float32:
                        raise EgoConfigError(
                            f"{shard_path}: {dictionary_name}[{head}] must be fp32"
                        )
                    if not torch.isfinite(value).all():
                        raise EgoConfigError(
                            f"{shard_path}: non-finite {dictionary_name}[{head}]"
                        )

            stop = cursor + count
            self.summaries[cursor:stop].copy_(summaries)
            for head in HEADS:
                self.visual_logits[head][cursor:stop].copy_(record["visual_logits"][head])
            sample_ids.extend(ids)
            cursor = stop

        if cursor != expected_rows:
            raise EgoConfigError(
                f"Derived-store {split} row mismatch: manifest={expected_rows}, loaded={cursor}"
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise EgoConfigError(f"Derived-store {split} contains duplicate sample IDs")
        self.sample_ids = sample_ids
        self.row_by_sample_id = {sample_id: row for row, sample_id in enumerate(sample_ids)}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def has(self, sample_id: str) -> bool:
        return sample_id in self.row_by_sample_id

    def summary(self, sample_id: str) -> torch.Tensor:
        return self.summaries[self.row_by_sample_id[sample_id]]

    def frozen_logits(self, sample_id: str) -> dict[str, torch.Tensor]:
        row = self.row_by_sample_id[sample_id]
        return {head: self.visual_logits[head][row] for head in HEADS}


class HistoryContextDataset(Dataset):
    """Join the audited history index to the derived visual store.

    Required index columns are ``sample_id``, ``current_cache_sample_id``,
    target ``verb_id/noun_id/action_id``, ``history_length``, and for every
    1-based slot ``i`` through ``K``:

    * ``history_i_cache_sample_id``
    * ``history_i_mask``
    * ``history_i_delta_t_sec``
    * ``history_i_level_id`` (step=0, substep=1, padding=-1)

    Slots are left padded, then chronological oldest-to-newest.  The audit
    columns ``audit_current_observation_end_sec`` and
    ``audit_target_start_sec`` are mandatory and assert strict anticipation.
    No column that resembles a history GT label is permitted.
    """

    def __init__(self, frame: pd.DataFrame, store: PreloadedHistoryStore, max_history: int) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.store = store
        self.max_history = int(max_history)
        required = {
            "video_uid",
            "sample_id",
            "current_cache_sample_id",
            "verb_id",
            "noun_id",
            "action_id",
            "history_length",
            "audit_current_observation_end_sec",
            "audit_target_start_sec",
        }
        for slot in range(1, self.max_history + 1):
            required.update(
                {
                    f"history_{slot}_cache_sample_id",
                    f"history_{slot}_mask",
                    f"history_{slot}_delta_t_sec",
                    f"history_{slot}_level_id",
                }
            )
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise EgoConfigError(f"History index is missing required columns: {missing}")
        if self.frame["sample_id"].astype(str).duplicated().any():
            raise EgoConfigError("History index sample_id values must be unique")

        forbidden = [
            column
            for column in self.frame.columns
            if column.startswith("history_")
            and any(token in column for token in ("verb", "noun", "action_label", "label_id"))
        ]
        if forbidden:
            raise EgoConfigError(f"History GT-label columns are forbidden: {forbidden}")

        self.rows: list[dict[str, Any]] = []
        referenced_ids: set[str] = set()
        for row_number, row in self.frame.iterrows():
            video_uid = str(row["video_uid"])
            sample_id = str(row["sample_id"])
            current_id = str(row["current_cache_sample_id"])
            if not video_uid or video_uid.lower() == "nan":
                raise EgoConfigError(f"row {row_number}: missing video_uid")
            if not current_id or current_id.lower() == "nan":
                raise EgoConfigError(f"row {row_number}: missing current cache ID")
            masks: list[bool] = []
            history_ids: list[str] = []
            deltas: list[float] = []
            levels: list[int] = []
            for slot in range(1, self.max_history + 1):
                mask = _strict_bool(
                    row[f"history_{slot}_mask"],
                    location=f"row {row_number}, slot {slot}",
                )
                raw_id = row[f"history_{slot}_cache_sample_id"]
                history_id = "" if pd.isna(raw_id) else str(raw_id)
                delta = float(row[f"history_{slot}_delta_t_sec"])
                level = int(row[f"history_{slot}_level_id"])
                if not math.isfinite(delta):
                    raise EgoConfigError(f"row {row_number}, slot {slot}: non-finite delta")
                if mask:
                    if not history_id or history_id.lower() == "nan":
                        raise EgoConfigError(f"row {row_number}, slot {slot}: valid slot has no ID")
                    if delta <= 0 or level not in (0, 1):
                        raise EgoConfigError(
                            f"row {row_number}, slot {slot}: valid slot needs delta>0 and level 0/1"
                        )
                    referenced_ids.add(history_id)
                elif history_id or delta != 0.0 or level != -1:
                    raise EgoConfigError(
                        f"row {row_number}, slot {slot}: padding must be id='', delta=0, level=-1"
                    )
                masks.append(mask)
                history_ids.append(history_id)
                deltas.append(delta)
                levels.append(level)

            # The builder contract is [padding..., oldest, ..., newest].
            if masks != sorted(masks):
                raise EgoConfigError(f"row {row_number}: history mask is not left padded")
            if sum(masks) != int(row["history_length"]):
                raise EgoConfigError(f"row {row_number}: history_length disagrees with masks")
            valid_history_ids = [value for value, mask in zip(history_ids, masks) if mask]
            if current_id in valid_history_ids:
                raise EgoConfigError(f"row {row_number}: current segment appears in its own history")
            if len(valid_history_ids) != len(set(valid_history_ids)):
                raise EgoConfigError(f"row {row_number}: duplicate history segment IDs")
            valid_deltas = [delta for delta, mask in zip(deltas, masks) if mask]
            if any(a < b for a, b in zip(valid_deltas, valid_deltas[1:])):
                raise EgoConfigError(
                    f"row {row_number}: deltas must be non-increasing oldest-to-newest"
                )
            if "annotation_level" in row:
                expected_level = {"step": 0, "substep": 1}.get(str(row["annotation_level"]))
                if expected_level is None:
                    raise EgoConfigError(
                        f"row {row_number}: unknown annotation_level={row['annotation_level']!r}"
                    )
                if any(level != expected_level for level, mask in zip(levels, masks) if mask):
                    raise EgoConfigError(f"row {row_number}: history is not same-level")
            current_obs_end = float(row["audit_current_observation_end_sec"])
            target_start = float(row["audit_target_start_sec"])
            if not math.isfinite(current_obs_end) or not math.isfinite(target_start):
                raise EgoConfigError(f"row {row_number}: non-finite audit boundary")
            if not current_obs_end < target_start:
                raise EgoConfigError(
                    f"row {row_number}: leakage contract failed: "
                    f"current_obs_end={current_obs_end} >= target_start={target_start}"
                )

            labels = {head: int(row[f"{head}_id"]) for head in HEADS}
            for head in HEADS:
                if not 0 <= labels[head] < store.num_classes[head]:
                    raise EgoConfigError(
                        f"row {row_number}: {head}_id={labels[head]} outside "
                        f"[0,{store.num_classes[head]})"
                    )
            referenced_ids.add(current_id)
            self.rows.append(
                {
                    "video_uid": video_uid,
                    "sample_id": sample_id,
                    "current_id": current_id,
                    "history_ids": history_ids,
                    "history_mask": masks,
                    "history_delta_t_sec": deltas,
                    "history_level_id": levels,
                    "labels": labels,
                }
            )

        missing_store_ids = sorted(sample_id for sample_id in referenced_ids if not store.has(sample_id))
        if missing_store_ids:
            raise EgoConfigError(
                f"History index references {len(missing_store_ids)} IDs absent from {store.split} store; "
                f"first={missing_store_ids[0]}"
            )
        self.zero_summary = torch.zeros(
            store.summary_tokens, store.embed_dim, dtype=torch.float16
        )

    @property
    def sample_ids(self) -> list[str]:
        return [row["sample_id"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        summaries = [self.store.summary(row["current_id"])]
        summaries.extend(
            self.store.summary(sample_id) if mask else self.zero_summary
            for sample_id, mask in zip(row["history_ids"], row["history_mask"])
        )
        result: dict[str, Any] = {
            "video_uid": row["video_uid"],
            "sample_id": row["sample_id"],
            "summaries": torch.stack(summaries),
            "history_mask": torch.tensor(row["history_mask"], dtype=torch.bool),
            "history_delta_t_sec": torch.tensor(row["history_delta_t_sec"], dtype=torch.float32),
            "history_level_id": torch.tensor(row["history_level_id"], dtype=torch.long),
            "history_length": int(sum(row["history_mask"])),
            "visual_logits": self.store.frozen_logits(row["current_id"]),
        }
        result.update({f"{head}_id": row["labels"][head] for head in HEADS})
        return result


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets_onehot = F.one_hot(targets, logits.shape[-1]).to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    p_t = probability * targets_onehot + (1.0 - probability) * (1.0 - targets_onehot)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets_onehot + (1.0 - alpha) * (1.0 - targets_onehot)
        loss = alpha_t * loss
    return loss.sum(dim=-1).mean()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "summaries": batch["summaries"].to(device=device, dtype=torch.float32, non_blocking=True),
        "history_mask": batch["history_mask"].to(device=device, non_blocking=True),
        "history_delta_t_sec": batch["history_delta_t_sec"].to(device=device, non_blocking=True),
        "history_level_id": batch["history_level_id"].to(device=device, non_blocking=True),
        "visual_logits": {
            head: batch["visual_logits"][head].to(device=device, dtype=torch.float32, non_blocking=True)
            for head in HEADS
        },
        "labels": {
            head: batch[f"{head}_id"].to(device=device, dtype=torch.long, non_blocking=True)
            for head in HEADS
        },
    }


def _model_forward(model: HistoryContextResidualHead, batch: dict[str, Any], device: torch.device):
    inputs = _to_device(batch, device)
    outputs = model(
        inputs["summaries"],
        inputs["history_mask"],
        inputs["history_delta_t_sec"],
        inputs["history_level_id"],
        inputs["visual_logits"],
    )
    return outputs, inputs["labels"]


def train_one_epoch(
    model: HistoryContextResidualHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    *,
    focal_alpha: float,
    focal_gamma: float,
    history_aux_weight: float,
    amp_dtype: torch.dtype | None,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "fused_loss": 0.0, "history_aux_loss": 0.0}
    samples = 0
    for batch in loader:
        batch_size = len(batch["sample_id"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype or torch.bfloat16,
            enabled=device.type == "cuda" and amp_dtype is not None,
        ):
            outputs, labels = _model_forward(model, batch, device)
            fused_loss = sum(
                sigmoid_focal_loss(
                    outputs["fused"][head], labels[head], alpha=focal_alpha, gamma=focal_gamma
                )
                for head in HEADS
            )
            history_aux_loss = sum(
                sigmoid_focal_loss(
                    outputs["history"][head], labels[head], alpha=focal_alpha, gamma=focal_gamma
                )
                for head in HEADS
            )
            loss = fused_loss + history_aux_weight * history_aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["fused_loss"] += float(fused_loss.detach().cpu()) * batch_size
        totals["history_aux_loss"] += float(history_aux_loss.detach().cpu()) * batch_size
        samples += batch_size
    return {key: value / max(1, samples) for key, value in totals.items()}


def _metric_block(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    return {
        "cmr@5": class_mean_recall(logits, labels, num_classes, k=5),
        "top1": top_k_recall(logits, labels, k=1),
        "top5": top_k_recall(logits, labels, k=5),
        "top10": top_k_recall(logits, labels, k=10),
        "top15": top_k_recall(logits, labels, k=15),
    }


def _history_bin(length: int, max_history: int) -> str:
    if length == 0:
        return "0"
    if length <= 2:
        return "1-2"
    if length <= 4:
        return "3-4"
    if length < max_history:
        return f"5-{max_history - 1}"
    return str(max_history)


@torch.inference_mode()
def evaluate(
    model: HistoryContextResidualHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: dict[str, int],
    *,
    require_visual_fallback_exact: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    collected = {
        mode: {head: [] for head in HEADS}
        for mode in MODES
    }
    labels_all = {head: [] for head in HEADS}
    history_lengths: list[int] = []
    sample_ids: list[str] = []
    video_uids: list[str] = []
    for batch in loader:
        outputs, labels = _model_forward(model, batch, device)
        if require_visual_fallback_exact:
            for head in HEADS:
                if not torch.equal(outputs["fused"][head], outputs["visual"][head]):
                    raise RuntimeError(
                        f"Epoch-0 fallback is not bit-exact for {head}; field gate must initialize at zero"
                    )
        for mode in MODES:
            for head in HEADS:
                collected[mode][head].append(outputs[mode][head].float().cpu())
        for head in HEADS:
            labels_all[head].append(labels[head].cpu())
        history_lengths.extend(int(value) for value in batch["history_length"])
        sample_ids.extend(str(value) for value in batch["sample_id"])
        video_uids.extend(str(value) for value in batch["video_uid"])

    logits = {
        mode: {head: torch.cat(collected[mode][head]) for head in HEADS}
        for mode in MODES
    }
    labels = {head: torch.cat(labels_all[head]) for head in HEADS}
    overall = {
        mode: {
            head: _metric_block(logits[mode][head], labels[head], num_classes[head])
            for head in HEADS
        }
        for mode in MODES
    }

    bin_positions: dict[str, list[int]] = {}
    for position, length in enumerate(history_lengths):
        bin_positions.setdefault(_history_bin(length, model.max_history), []).append(position)
    history_length_bins: dict[str, Any] = {}
    for name, positions in bin_positions.items():
        index = torch.tensor(positions, dtype=torch.long)
        history_length_bins[name] = {
            "size": len(positions),
            "modes": {
                mode: {
                    head: _metric_block(
                        logits[mode][head][index], labels[head][index], num_classes[head]
                    )
                    for head in HEADS
                }
                for mode in MODES
            },
        }
    metrics = {
        "size": len(sample_ids),
        "overall": overall,
        "history_length_bins": history_length_bins,
        "gate_values": model.gate_values(),
        "sample_ids": sample_ids,
    }
    predictions = {
        "sample_ids": sample_ids,
        "video_uids": video_uids,
        "history_lengths": torch.tensor(history_lengths, dtype=torch.long),
        "labels": labels,
        "logits": logits,
    }
    return metrics, predictions


def _save_val_predictions(
    path: Path,
    *,
    epoch: int,
    predictions: dict[str, Any],
    num_classes: dict[str, int],
    gate_values: dict[str, Any],
) -> None:
    """Write the per-epoch full-val logits needed for leakage-safe OOF selection."""
    sample_ids = predictions["sample_ids"]
    video_uids = predictions["video_uids"]
    history_lengths = predictions["history_lengths"]
    labels = predictions["labels"]
    logits = predictions["logits"]
    row_count = len(sample_ids)
    if row_count == 0 or len(video_uids) != row_count:
        raise RuntimeError("Validation prediction artifact has inconsistent row metadata")
    if len(set(sample_ids)) != row_count:
        raise RuntimeError("Validation prediction artifact contains duplicate sample IDs")
    if not torch.is_tensor(history_lengths) or tuple(history_lengths.shape) != (row_count,):
        raise RuntimeError("Validation prediction artifact has invalid history_lengths")
    if set(labels) != set(HEADS) or set(logits) != set(MODES):
        raise RuntimeError("Validation prediction artifact has invalid labels or modes")
    for head in HEADS:
        if tuple(labels[head].shape) != (row_count,) or labels[head].dtype != torch.long:
            raise RuntimeError(f"Validation labels[{head}] have an invalid schema")
    for mode in MODES:
        if set(logits[mode]) != set(HEADS):
            raise RuntimeError(f"Validation logits[{mode}] have an invalid head set")
        for head in HEADS:
            expected_shape = (row_count, int(num_classes[head]))
            value = logits[mode][head]
            if tuple(value.shape) != expected_shape or value.dtype != torch.float32:
                raise RuntimeError(
                    f"Validation logits[{mode}][{head}] must be fp32 {expected_shape}"
                )
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Validation logits[{mode}][{head}] are non-finite")
    _atomic_torch_save(
        {
            "format_version": SCHEMA_VERSION,
            "kind": "goalstep_history_context_val_predictions",
            "epoch": int(epoch),
            "contract": "A2.end-1s -> strict same-level A3",
            "sample_ids": sample_ids,
            "video_uids": video_uids,
            "history_lengths": history_lengths,
            "labels": labels,
            "logits": logits,
            "num_classes": {head: int(num_classes[head]) for head in HEADS},
            "gate_values": gate_values,
        },
        path,
    )


def _log_metrics(prefix: str, result: dict[str, Any]) -> None:
    for mode in MODES:
        for head in HEADS:
            metric = result["overall"][mode][head]
            step_log(
                1,
                PHASE,
                f"{prefix} {mode}/{head}: CMR@5={metric['cmr@5']:.2f} "
                f"top1={metric['top1']:.2f} top5={metric['top5']:.2f} "
                f"top10={metric['top10']:.2f} top15={metric['top15']:.2f}",
            )
    step_log(1, PHASE, f"{prefix} gates={result['gate_values']}")


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    final_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _checkpoint_state(
    *,
    epoch: int,
    model: HistoryContextResidualHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "metric_name": "fused.action.top5",
        "metric": metrics["overall"]["fused"]["action"]["top5"],
        "gate_values": model.gate_values(),
        "metadata": metadata,
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(get(config, "experiment.seed", 42))
    set_seed(seed)
    phase0_diagnostic = _load_phase0_diagnostic(config)
    device_name = str(get(config, "experiment.device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    run_dir_path = expand_path(require(config, "experiment.output_dir"))
    existing_artifacts = [
        run_dir_path / filename
        for filename in ("training_history.csv", "best.pt", "latest.pt", "final_metrics.json")
        if (run_dir_path / filename).exists()
    ]
    prediction_dir_path = run_dir_path / "val_predictions"
    if prediction_dir_path.is_dir() and any(prediction_dir_path.glob("epoch_*.pt")):
        existing_artifacts.append(prediction_dir_path)
    if existing_artifacts:
        raise EgoConfigError(
            "History training intentionally has no resume/overwrite semantics in v1. "
            "Choose a fresh experiment.output_dir; existing artifact: "
            f"{existing_artifacts[0]}"
        )
    run_dir = ensure_dir(run_dir_path)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    prediction_dir = ensure_dir(run_dir / "val_predictions")

    store_root = expand_path(require(config, "dataset.derived_store_dir"))
    index_dir = expand_path(require(config, "dataset.history_index_dir"))
    max_history = int(get(config, "dataset.max_history", 8))
    verify_hashes = bool(get(config, "dataset.verify_shard_hashes", False))
    step_log(1, PHASE, f"Loading compact derived stores from {store_root}")
    train_store = PreloadedHistoryStore(store_root, "train", verify_shard_hashes=verify_hashes)
    val_store = PreloadedHistoryStore(store_root, "val", verify_shard_hashes=verify_hashes)
    if train_store.num_classes != val_store.num_classes:
        raise EgoConfigError("Train/val derived-store taxonomy mismatch")
    if (train_store.summary_tokens, train_store.embed_dim) != (
        val_store.summary_tokens,
        val_store.embed_dim,
    ):
        raise EgoConfigError("Train/val summary-shape mismatch")
    expected_tokens = int(get(config, "dataset.expected_summary_tokens", 17))
    expected_embed_dim = int(get(config, "dataset.expected_embed_dim", 1024))
    if (train_store.summary_tokens, train_store.embed_dim) != (
        expected_tokens,
        expected_embed_dim,
    ):
        raise EgoConfigError(
            "Derived-store summary contract mismatch: expected "
            f"[{expected_tokens},{expected_embed_dim}], got "
            f"[{train_store.summary_tokens},{train_store.embed_dim}]"
        )
    train_frame, train_index_path = _read_index(index_dir, "train")
    val_frame, val_index_path = _read_index(index_dir, "val")
    train_dataset = HistoryContextDataset(train_frame, train_store, max_history)
    val_dataset = HistoryContextDataset(val_frame, val_store, max_history)
    if not len(train_dataset) or not len(val_dataset):
        raise EgoConfigError("History train and val datasets must both be non-empty")

    batch_size = int(get(config, "training.batch_size", 32))
    eval_batch_size = int(get(config, "training.eval_batch_size", batch_size))
    num_workers = int(get(config, "dataset.num_workers", 0))
    if batch_size < 1 or eval_batch_size < 1 or num_workers < 0:
        raise EgoConfigError("Batch sizes must be positive and num_workers non-negative")
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model_config = get(config, "model.history", {})
    model = HistoryContextResidualHead(
        num_classes=train_store.num_classes,
        embed_dim=train_store.embed_dim,
        max_history=max_history,
        segment_pooler_heads=int(model_config.get("segment_pooler_heads", 16)),
        transformer_heads=int(model_config.get("transformer_heads", 16)),
        transformer_layers=int(model_config.get("transformer_layers", 2)),
        transformer_mlp_ratio=float(model_config.get("transformer_mlp_ratio", 4.0)),
        transformer_dropout=float(model_config.get("transformer_dropout", 0.1)),
        segment_dropout=float(model_config.get("segment_dropout", 0.3)),
        recency_scale_sec=float(model_config.get("recency_scale_sec", 300.0)),
    ).to(device)

    learning_rate = float(require(config, "training.learning_rate"))
    weight_decay = float(get(config, "training.weight_decay", 1e-4))
    epochs = int(require(config, "training.epochs"))
    if epochs < 1:
        raise EgoConfigError("training.epochs must be >= 1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(float(get(config, "training.warmup_epochs", 1.0)) * len(train_loader))
    final_lr = float(get(config, "training.final_lr", 0.0))
    final_lr_ratio = final_lr / learning_rate if learning_rate > 0 else 0.0
    scheduler = _warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        final_lr_ratio=final_lr_ratio,
    )
    precision = str(get(config, "training.precision", "bf16" if device.type == "cuda" else "fp32"))
    if precision not in ("fp32", "bf16"):
        raise EgoConfigError("training.precision must be fp32 or bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" and device.type == "cuda" else None
    focal_alpha = float(get(config, "training.focal_alpha", 0.25))
    focal_gamma = float(get(config, "training.focal_gamma", 2.0))
    history_aux_weight = float(get(config, "training.history_aux_weight", 0.25))
    gradient_clip_raw = get(config, "training.gradient_clip_norm", 1.0)
    gradient_clip_norm = None if gradient_clip_raw is None else float(gradient_clip_raw)

    provenance = {
        "derived_store_manifest": str(train_store.manifest_path),
        "derived_store_manifest_sha256": _sha256(train_store.manifest_path),
        "history_train_index": str(train_index_path),
        "history_train_index_sha256": _sha256(train_index_path),
        "history_val_index": str(val_index_path),
        "history_val_index_sha256": _sha256(val_index_path),
        "source_cache_dir": train_store.manifest.get("source_cache_dir"),
        "visual_checkpoint": train_store.manifest.get("visual_checkpoint"),
        "visual_checkpoint_sha256": train_store.manifest.get("visual_checkpoint_sha256"),
        "recognition_checkpoint": train_store.manifest.get("recognition_checkpoint"),
        "recognition_checkpoint_sha256": train_store.manifest.get("recognition_checkpoint_sha256"),
    }
    checkpoint_metadata = {
        "task": "goalstep_same_level_next_action_A3",
        "feature_reextraction": False,
        "derived_store_schema_version": SCHEMA_VERSION,
        "history_index_contract": (
            "left-padded completed same-video same-level visual segments, oldest-to-newest; "
            "no history GT labels; current obs_end < target start"
        ),
        "summary_shape": [train_store.summary_tokens, train_store.embed_dim],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "architecture": model.architecture_metadata(),
        "provenance": provenance,
        "phase0_diagnostic": phase0_diagnostic,
        "checkpoint_selection_semantics": (
            "best.pt and best_action_top5.pt are legacy aliases for the exploratory "
            "full-validation maximum; they are not the authoritative OOF champion"
        ),
        "deployability_limitations": [
            (
                "History membership and same-level chains are built with oracle GoalStep action "
                "boundaries and annotation_level. No history class labels enter the model, but "
                "online deployment needs an upstream boundary/level estimator."
            ),
            (
                "The current observation uses the audited A2.end-1s anchor; the prediction target "
                "is the strict same-level next action A3."
            ),
        ],
    }
    run_metadata = {
        **checkpoint_metadata,
        "seed": seed,
        "device": str(device),
        "precision": precision,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "history_aux_weight": history_aux_weight,
        "full_val_checkpoint_selection_metric": "fused.action.top5",
        "full_val_checkpoint_selection_semantics": (
            "exploratory only; authoritative model comparison uses video-disjoint "
            "cross-fitted per-epoch logits"
        ),
        "authoritative_champion_reference": "P0-a same-decision OOF ensemble",
        "epoch_0_contract": "bit-exact fused == frozen visual because every field gate is zero",
        "resume_supported": False,
        "existing_output_policy": "fail_closed",
    }
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_json(run_dir / "run_metadata.json", run_metadata)
    step_log(
        1,
        PHASE,
        f"Train={len(train_dataset)} Val={len(val_dataset)} summary="
        f"[{train_store.summary_tokens},{train_store.embed_dim}] device={device}",
    )

    # Epoch 0 is a mandatory, bit-exact visual fallback checkpoint.
    epoch_zero, epoch_zero_predictions = evaluate(
        model,
        val_loader,
        device,
        train_store.num_classes,
        require_visual_fallback_exact=True,
    )
    _save_val_predictions(
        prediction_dir / "epoch_00.pt",
        epoch=0,
        predictions=epoch_zero_predictions,
        num_classes=train_store.num_classes,
        gate_values=epoch_zero["gate_values"],
    )
    _log_metrics("Val[FULL] epoch 0 visual fallback", epoch_zero)
    initial_state = _checkpoint_state(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=epoch_zero,
        metadata=checkpoint_metadata,
    )
    torch.save(initial_state, checkpoint_dir / "epoch_00_visual_fallback.pt")
    torch.save(initial_state, run_dir / "best.pt")
    torch.save(initial_state, run_dir / "best_action_top5.pt")
    torch.save(initial_state, run_dir / "best_fullval_exploratory.pt")
    best_metric = float(epoch_zero["overall"]["fused"]["action"]["top5"])
    best_epoch = 0

    records: list[dict[str, Any]] = [
        {"epoch": 0, "train": None, "val": epoch_zero, "seconds": 0.0}
    ]
    write_json(run_dir / "metrics_per_epoch.json", records)
    history_csv = run_dir / "training_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "fused_action_cmr@5",
                "fused_action_top1",
                "fused_action_top5",
                "fused_action_top10",
                "fused_action_top15",
                "history_action_top5",
                "current_only_action_top5",
                "visual_action_top5",
                "gate_verb",
                "gate_noun",
                "gate_action",
                "seconds",
            ]
        )
        action = epoch_zero["overall"]["fused"]["action"]
        writer.writerow(
            [
                0,
                "",
                action["cmr@5"],
                action["top1"],
                action["top5"],
                action["top10"],
                action["top15"],
                epoch_zero["overall"]["history"]["action"]["top5"],
                epoch_zero["overall"]["current_only"]["action"]["top5"],
                epoch_zero["overall"]["visual"]["action"]["top5"],
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    for epoch in range(1, epochs + 1):
        step_log(1, PHASE, f"Epoch {epoch}/{epochs}")
        started = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            history_aux_weight=history_aux_weight,
            amp_dtype=amp_dtype,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_metrics, val_predictions = evaluate(
            model, val_loader, device, train_store.num_classes
        )
        _save_val_predictions(
            prediction_dir / f"epoch_{epoch:02d}.pt",
            epoch=epoch,
            predictions=val_predictions,
            num_classes=train_store.num_classes,
            gate_values=val_metrics["gate_values"],
        )
        elapsed = time.time() - started
        _log_metrics(f"Val[FULL] epoch {epoch}", val_metrics)
        step_log(1, PHASE, f"Epoch {epoch} train={train_metrics} seconds={elapsed:.1f}")
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": elapsed,
        }
        records.append(record)
        write_json(run_dir / "metrics_per_epoch.json", records)

        state = _checkpoint_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=val_metrics,
            metadata=checkpoint_metadata,
        )
        torch.save(state, checkpoint_dir / f"epoch_{epoch:02d}.pt")
        torch.save(state, run_dir / "latest.pt")
        selection_metric = float(val_metrics["overall"]["fused"]["action"]["top5"])
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            torch.save(state, run_dir / "best.pt")
            torch.save(state, run_dir / "best_action_top5.pt")
            torch.save(state, run_dir / "best_fullval_exploratory.pt")
            step_log(
                1,
                PHASE,
                "Exploratory full-val best updated: "
                f"epoch={epoch} fused action top5={selection_metric:.2f}",
            )

        action = val_metrics["overall"]["fused"]["action"]
        with history_csv.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    action["cmr@5"],
                    action["top1"],
                    action["top5"],
                    action["top10"],
                    action["top15"],
                    val_metrics["overall"]["history"]["action"]["top5"],
                    val_metrics["overall"]["current_only"]["action"]["top5"],
                    val_metrics["overall"]["visual"]["action"]["top5"],
                    val_metrics["gate_values"]["verb"]["tanh"],
                    val_metrics["gate_values"]["noun"]["tanh"],
                    val_metrics["gate_values"]["action"]["tanh"],
                    elapsed,
                ]
            )

    visual_reference = float(epoch_zero["overall"]["visual"]["action"]["top5"])
    final = {
        "best_epoch": best_epoch,
        "best_fused_action_top5": best_metric,
        "checkpoint_selection_metric": "fused.action.top5",
        "checkpoint_selection_semantics": "exploratory_full_validation",
        "epoch_0_visual_fallback": epoch_zero,
        "best_val": records[best_epoch]["val"],
        "per_epoch": records,
        "lower_bound_preserved": best_metric >= float(
            epoch_zero["overall"]["visual"]["action"]["top5"]
        ),
        "phase1_decision": {
            "status": "deferred_to_cross_fitted_champion_evaluator",
            "authoritative_metric": "paired Action instance Top-5 accuracy",
            "reference": "P0-a same-decision OOF ensemble",
            "visual_reference_percent": visual_reference,
            "p0b_is_gate": False,
            "adoption_rule": "delta_top5_pp > 0 and video_bootstrap_95ci_lower_pp > 0",
            "material_gain_pp_is_descriptive_only": float(
                get(config, "champion.material_gain_pp_descriptive", 1.0)
            ),
            "full_val_exploratory_best_percent": best_metric,
            "full_val_exploratory_history_increment_over_current_only_pp": best_metric
            - float(records[best_epoch]["val"]["overall"]["current_only"]["action"]["top5"]),
        },
    }
    write_json(run_dir / "final_metrics.json", final)
    step_log(1, PHASE, f"Done. best_epoch={best_epoch} top5={best_metric:.2f} run={run_dir}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
