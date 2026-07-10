"""Training scaffold for Step 1 action anticipation.

Trains verb/noun/action :class:`~ego.step1_action_anticipation.models.anticipation_head.AnticipationHead`
classifiers on top of a frozen V-JEPA2 backbone. Verb, noun, and action often
reach their best validation score at different epochs, so this saves
``best_verb.pt`` / ``best_noun.pt`` / ``best_action.pt`` independently rather
than tracking a single "best" checkpoint.
"""

from __future__ import annotations

import csv
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ego.common.config import get, load_config, require
from ego.common.exceptions import EgoConfigError
from ego.common.io import ensure_dir, write_yaml
from ego.common.logging import step_log
from ego.common.paths import expand_path
from ego.common.seed import set_seed
from ego.step1_action_anticipation.data.build_samples import build_step1_datasets
from ego.step1_action_anticipation.data.collator import anticipation_collate
from ego.step1_action_anticipation.data.feature_cache import (
    FeatureCacheDataset,
    extract_and_cache_features,
)
from ego.step1_action_anticipation.metrics import class_mean_recall
from ego.step1_action_anticipation.models import AnticipationHead, load_vjepa2_backbone

HEADS = ("verb", "noun", "action")


def sigmoid_focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """RetinaNet-style focal loss (https://arxiv.org/abs/1708.02002), ported from the prototype."""
    num_classes = inputs.size(-1)
    targets_onehot = F.one_hot(targets, num_classes).float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_onehot, reduction="none")
    p_t = p * targets_onehot + (1 - p) * (1 - targets_onehot)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
        loss = alpha_t * loss
    return loss.sum(dim=-1).mean()


class _WarmupCosineLR:
    def __init__(self, optimizer, ref_lr, start_lr, final_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.ref_lr, self.start_lr, self.final_lr = ref_lr, start_lr, final_lr
        self.warmup_steps, self.total_steps = warmup_steps, total_steps
        self._step = 0

    def step(self) -> None:
        self._step += 1
        if self._step < self.warmup_steps:
            progress = self._step / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * progress)),
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class _CosineWD:
    def __init__(self, optimizer, ref_wd, final_wd, total_steps):
        self.optimizer = optimizer
        self.ref_wd, self.final_wd, self.total_steps = ref_wd, final_wd, total_steps
        self._step = 0

    def step(self) -> None:
        self._step += 1
        progress = self._step / max(1, self.total_steps)
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1 + math.cos(math.pi * progress))
        wd = max(self.final_wd, wd) if self.final_wd <= self.ref_wd else min(self.final_wd, wd)
        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd


def _forward_logits(
    head: AnticipationHead, backbone: torch.nn.Module | None, batch: dict, device: torch.device
) -> tuple[dict, dict]:
    clips = batch["video"].to(device)
    labels = {h: batch[f"{h}_id"].to(device) for h in HEADS}
    if backbone is not None:
        with torch.no_grad():
            features = backbone(clips, batch["anticipation_time_sec"].to(device))
    else:
        features = clips  # already-extracted feature cache
    logits = head(features)
    return logits, labels


def train_one_epoch(
    head: AnticipationHead,
    backbone: torch.nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: _WarmupCosineLR,
    wd_scheduler: _CosineWD,
    device: torch.device,
    use_focal_loss: bool,
) -> float:
    head.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        logits, labels = _forward_logits(head, backbone, batch, device)
        loss = 0.0
        for h in HEADS:
            if h not in logits:
                continue
            loss = loss + (
                sigmoid_focal_loss(logits[h], labels[h])
                if use_focal_loss
                else F.cross_entropy(logits[h], labels[h])
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        wd_scheduler.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    head: AnticipationHead,
    backbone: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    num_classes: dict[str, int],
    k: int = 5,
) -> dict[str, float]:
    head.eval()
    all_logits: dict[str, list[torch.Tensor]] = {h: [] for h in HEADS}
    all_labels: dict[str, list[torch.Tensor]] = {h: [] for h in HEADS}

    for batch in loader:
        logits, labels = _forward_logits(head, backbone, batch, device)
        for h in HEADS:
            if h not in logits:
                continue
            all_logits[h].append(logits[h].cpu())
            all_labels[h].append(labels[h].cpu())

    metrics = {}
    for h in HEADS:
        if not all_logits[h]:
            continue
        metrics[f"{h}_cmr@{k}"] = class_mean_recall(
            torch.cat(all_logits[h]), torch.cat(all_labels[h]), num_classes[h], k=k
        )
    return metrics


def _build_dataloaders(config: dict, datasets, device: torch.device):
    """Build train/val DataLoaders, transparently using a feature cache if configured."""
    use_feature_cache = get(config, "training.use_feature_cache", False)
    num_workers = get(config, "dataset.num_workers", 2)
    batch_size = require(config, "training.batch_size")
    backbone = None

    if use_feature_cache:
        cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
        train_cache, val_cache = cache_dir / "train", cache_dir / "val"
        if not (train_cache.exists() and any(train_cache.glob("*.pt"))):
            step_log(1, "Train", f"Feature cache empty at {cache_dir}; extracting now (one-time cost)")
            extractor = load_vjepa2_backbone(
                frames_per_clip=require(config, "dataset.frames_per_clip"),
                frames_per_second=require(config, "dataset.frames_per_second"),
                resolution=require(config, "dataset.resolution"),
                checkpoint=expand_path(require(config, "model.checkpoint")),
                model_kwargs=require(config, "model.model_kwargs"),
                wrapper_kwargs=get(config, "model.wrapper_kwargs", {}),
                repository_dir=get(config, "model.repository_dir"),
                device=device,
            )
            extract_batch_size = get(config, "training.extract_batch_size", batch_size)
            extract_and_cache_features(
                datasets.train, extractor, train_cache, device, extract_batch_size, num_workers
            )
            if datasets.val is not None:
                extract_and_cache_features(
                    datasets.val, extractor, val_cache, device, extract_batch_size, num_workers
                )
            del extractor
        train_dataset = FeatureCacheDataset.from_cache_dir(train_cache)
        val_dataset = FeatureCacheDataset.from_cache_dir(val_cache) if val_cache.exists() else None
    else:
        backbone = load_vjepa2_backbone(
            frames_per_clip=require(config, "dataset.frames_per_clip"),
            frames_per_second=require(config, "dataset.frames_per_second"),
            resolution=require(config, "dataset.resolution"),
            checkpoint=expand_path(require(config, "model.checkpoint")),
            model_kwargs=require(config, "model.model_kwargs"),
            wrapper_kwargs=get(config, "model.wrapper_kwargs", {}),
            repository_dir=get(config, "model.repository_dir"),
            device=device,
        )
        train_dataset, val_dataset = datasets.train, datasets.val

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=anticipation_collate,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=anticipation_collate,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    return train_loader, val_loader, backbone


def train(config_path: str) -> dict:
    config = load_config(config_path)
    seed = get(config, "experiment.seed", 42)
    set_seed(seed)
    step_log(1, "Train", "Config loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_log(1, "Train", f"Device: {device}")

    datasets = build_step1_datasets(config)
    mapping = datasets.label_mapping
    if datasets.val is None:
        raise EgoConfigError("No validation samples available; cannot train without a validation split.")
    step_log(1, "Train", f"Train samples: {len(datasets.train)}  Val samples: {len(datasets.val)}")
    step_log(
        1,
        "Train",
        f"Verb classes: {mapping.num_verbs}  Noun classes: {mapping.num_nouns}  "
        f"Action classes: {mapping.num_actions}",
    )

    train_loader, val_loader, backbone = _build_dataloaders(config, datasets, device)
    if val_loader is None:
        raise EgoConfigError("Validation split resolved to zero usable samples after dataset preparation.")

    embed_dim = backbone.embed_dim if backbone is not None else next(iter(train_loader))["video"].shape[-1]
    classifier_cfg = get(config, "model.classifier", {})
    head = AnticipationHead(
        num_verb_classes=mapping.num_verbs,
        num_noun_classes=mapping.num_nouns,
        num_action_classes=mapping.num_actions,
        embed_dim=embed_dim,
        num_heads=classifier_cfg.get("num_heads", 16),
        depth=classifier_cfg.get("num_probe_blocks", 4),
        repository_dir=get(config, "model.repository_dir"),
    ).to(device)

    num_epochs = require(config, "training.epochs")
    iterations_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * iterations_per_epoch
    warmup_epochs = get(config, "training.warmup_epochs", 0)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=require(config, "training.learning_rate"),
        weight_decay=get(config, "training.weight_decay", 0.0001),
    )
    lr_scheduler = _WarmupCosineLR(
        optimizer,
        ref_lr=require(config, "training.learning_rate"),
        start_lr=get(config, "training.start_lr", 0.0),
        final_lr=get(config, "training.final_lr", 0.0),
        warmup_steps=int(warmup_epochs * iterations_per_epoch),
        total_steps=total_steps,
    )
    wd_scheduler = _CosineWD(
        optimizer,
        ref_wd=get(config, "training.weight_decay", 0.0001),
        final_wd=get(config, "training.final_weight_decay", 0.0001),
        total_steps=total_steps,
    )
    use_focal_loss = get(config, "training.use_focal_loss", True)

    checkpoint_dir = ensure_dir(expand_path(require(config, "checkpoint.output_dir")))
    output_dir = ensure_dir(expand_path(require(config, "experiment.output_dir")))
    write_yaml(output_dir / "config_resolved.yaml", config)

    num_classes = {"verb": mapping.num_verbs, "noun": mapping.num_nouns, "action": mapping.num_actions}
    best_metrics = {h: float("-inf") for h in HEADS}
    history_path = output_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "verb_cmr@5", "noun_cmr@5", "action_cmr@5", "seconds"])

    for epoch in range(1, num_epochs + 1):
        step_log(1, "Train", f"Epoch {epoch}/{num_epochs}")
        t0 = time.time()
        train_loss = train_one_epoch(
            head, backbone, train_loader, optimizer, lr_scheduler, wd_scheduler, device, use_focal_loss
        )
        step_log(1, "Train", f"Train loss: {train_loss:.4f}")

        val_metrics = validate(head, backbone, val_loader, device, num_classes)
        for h in HEADS:
            key = f"{h}_cmr@5"
            if key in val_metrics:
                step_log(1, "Train", f"Validation {h} CMR@5: {val_metrics[key]:.2f}")

        elapsed = time.time() - t0
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    val_metrics.get("verb_cmr@5", ""),
                    val_metrics.get("noun_cmr@5", ""),
                    val_metrics.get("action_cmr@5", ""),
                    f"{elapsed:.1f}",
                ]
            )

        torch.save(
            {"epoch": epoch, "model_state": head.state_dict(), "optimizer_state": optimizer.state_dict()},
            checkpoint_dir / "latest.pt",
        )
        for h in HEADS:
            key = f"{h}_cmr@5"
            if key in val_metrics and val_metrics[key] > best_metrics[h]:
                best_metrics[h] = val_metrics[key]
                best_path = checkpoint_dir / f"best_{h}.pt"
                torch.save(
                    {"epoch": epoch, "model_state": head.state_dict(), "metric": val_metrics[key]}, best_path
                )
                step_log(1, "Train", f"Best checkpoint updated: {best_path} ({key}={val_metrics[key]:.2f})")

    return {"best_metrics": best_metrics, "history_path": str(history_path)}
