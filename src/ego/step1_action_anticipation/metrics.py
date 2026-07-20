"""Metric scaffold for Step 1 action anticipation.

Pure functions only: every function here takes accumulated logits/labels
tensors and returns a number (or a per-class tensor). Training loops, file
saving, and dataset-specific logic live in ``train.py`` / ``evaluate.py``.
"""

from __future__ import annotations

import torch


def top_k_recall(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Instance-level top-k hit rate: % of samples whose true label is in the top-k logits."""
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    hits = (topk == labels.unsqueeze(1)).any(dim=1)
    return 100.0 * hits.float().mean().item()


def per_class_recall(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int, k: int = 5
) -> torch.Tensor:
    """Per-class top-k recall, shape ``[num_classes]``. Classes with zero support are ``NaN``."""
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    hits = (topk == labels.unsqueeze(1)).any(dim=1).float()

    tp = torch.zeros(num_classes, dtype=torch.float64)
    support = torch.zeros(num_classes, dtype=torch.float64)
    tp.scatter_add_(0, labels, hits.double())
    support.scatter_add_(0, labels, torch.ones_like(hits, dtype=torch.float64))

    recall = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    has_support = support > 0
    recall[has_support] = tp[has_support] / support[has_support]
    return recall


def class_mean_recall(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int, k: int = 5
) -> float:
    """Macro-averaged top-k recall across classes that have at least one sample."""
    recall = per_class_recall(logits, labels, num_classes, k=k)
    valid = recall[~torch.isnan(recall)]
    if valid.numel() == 0:
        return float("nan")
    return 100.0 * valid.mean().item()


def verb_noun_joint_recall(
    verb_logits: torch.Tensor,
    noun_logits: torch.Tensor,
    verb_labels: torch.Tensor,
    noun_labels: torch.Tensor,
    k: int = 5,
) -> float:
    """% of samples where the true verb AND the true noun are each independently in their top-k.

    This is the looser "both heads correct" rate, distinct from the strict
    joint metric obtained by calling :func:`top_k_recall` on the action head
    (which requires the exact (verb, noun) pair, not just each head
    independently landing in its own top-k).
    """
    kv = min(k, verb_logits.size(-1))
    kn = min(k, noun_logits.size(-1))
    verb_hit = (verb_logits.topk(kv, dim=-1).indices == verb_labels.unsqueeze(1)).any(dim=1)
    noun_hit = (noun_logits.topk(kn, dim=-1).indices == noun_labels.unsqueeze(1)).any(dim=1)
    return 100.0 * (verb_hit & noun_hit).float().mean().item()


def candidate_coverage(logits: torch.Tensor, num_classes: int, k: int = 5) -> float:
    """% of the label vocabulary that appears at least once across all samples' top-k predictions.

    Low coverage signals the model collapsing onto a few head classes even if
    its recall looks reasonable.
    """
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    covered = torch.zeros(num_classes, dtype=torch.bool)
    covered[topk.reshape(-1).unique()] = True
    return 100.0 * covered.float().mean().item()


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of the softmax distribution over ``logits``'s last dim.

    ``logits`` may be ``[C]`` (single sample, returns a 0-d tensor) or
    ``[N, C]`` (batch, returns ``[N]``).
    """
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
