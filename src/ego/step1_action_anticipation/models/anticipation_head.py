"""Verb, noun, and action prediction heads for Step 1 action anticipation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ego.step1_action_anticipation.models.attentive_probe import AttentiveProbe


class AnticipationHead(nn.Module):
    """Attentive probe + independent linear heads producing verb/noun/action logits.

    ``num_verb_classes == 0`` switches to action-only mode (single pooled
    query, no separate verb/noun heads) for datasets without verb/noun labels.
    """

    def __init__(
        self,
        num_verb_classes: int,
        num_noun_classes: int,
        num_action_classes: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        repository_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.action_only = num_verb_classes == 0
        self.probe = AttentiveProbe(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            num_queries=1 if self.action_only else 3,
            repository_dir=repository_dir,
        )
        if not self.action_only:
            self.verb_classifier = nn.Linear(embed_dim, num_verb_classes, bias=True)
            self.noun_classifier = nn.Linear(embed_dim, num_noun_classes, bias=True)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes, bias=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if torch.isnan(x).any():
            raise ValueError("NaN detected in backbone features passed to AnticipationHead.")

        pooled = self.probe(x)
        if self.action_only:
            return {"action": self.action_classifier(pooled[:, 0, :])}

        verb, noun, action = pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]
        return {
            "verb": self.verb_classifier(verb),
            "noun": self.noun_classifier(noun),
            "action": self.action_classifier(action),
        }
