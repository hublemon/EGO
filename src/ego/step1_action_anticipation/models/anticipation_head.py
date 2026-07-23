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
        use_temporal_metadata: bool = False,
        temporal_duration_scale_sec: float = 32.0,
    ) -> None:
        super().__init__()
        self.action_only = num_verb_classes == 0
        self.use_temporal_metadata = use_temporal_metadata
        self.temporal_duration_scale_sec = float(temporal_duration_scale_sec)
        if self.temporal_duration_scale_sec <= 0:
            raise ValueError("temporal_duration_scale_sec must be > 0")
        if use_temporal_metadata:
            self.temporal_metadata_projection = nn.Sequential(
                nn.Linear(4, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            self.temporal_metadata_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.annotation_level_embedding = nn.Embedding(2, embed_dim)
            self.annotation_level_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
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

    def forward(
        self,
        x: torch.Tensor,
        observation_duration_sec: torch.Tensor | None = None,
        observed_action_duration_sec: torch.Tensor | None = None,
        frame_time_positions: torch.Tensor | None = None,
        frame_terminal_mask: torch.Tensor | None = None,
        annotation_level_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if torch.isnan(x).any():
            raise ValueError("NaN detected in backbone features passed to AnticipationHead.")

        if self.use_temporal_metadata:
            if (
                observation_duration_sec is None
                or observed_action_duration_sec is None
                or frame_time_positions is None
                or frame_terminal_mask is None
                or annotation_level_id is None
            ):
                raise ValueError(
                    "Temporal-metadata head requires observation_duration_sec, "
                    "observed_action_duration_sec, frame_time_positions, "
                    "frame_terminal_mask, and annotation_level_id"
                )
            frame_time_positions = frame_time_positions.to(device=x.device, dtype=x.dtype)
            frame_terminal_mask = frame_terminal_mask.to(device=x.device, dtype=x.dtype)
            observation_duration = observation_duration_sec.to(
                device=x.device, dtype=x.dtype
            ).reshape(-1, 1)
            observation_duration = torch.log1p(observation_duration.clamp(min=0.0)) / torch.log(
                x.new_tensor(1.0 + self.temporal_duration_scale_sec)
            )
            observation_duration = observation_duration.expand_as(frame_time_positions)
            action_duration = observed_action_duration_sec.to(
                device=x.device, dtype=x.dtype
            ).reshape(-1, 1)
            action_duration = torch.log1p(action_duration.clamp(min=0.0)) / torch.log(
                x.new_tensor(1.0 + self.temporal_duration_scale_sec)
            )
            action_duration = action_duration.expand_as(frame_time_positions)
            metadata = torch.stack(
                [
                    frame_time_positions,
                    frame_terminal_mask,
                    observation_duration,
                    action_duration,
                ],
                dim=-1,
            )
            metadata_tokens = self.temporal_metadata_projection(metadata) + self.temporal_metadata_type
            level_ids = annotation_level_id.to(device=x.device, dtype=torch.long).reshape(-1)
            if ((level_ids < 0) | (level_ids > 1)).any():
                raise ValueError("annotation_level_id must encode step=0 or substep=1")
            level_token = self.annotation_level_embedding(level_ids).unsqueeze(1)
            level_token = level_token + self.annotation_level_type
            x = torch.cat([x, metadata_tokens, level_token], dim=1)

        pooled = self.probe(x)
        if self.action_only:
            return {"action": self.action_classifier(pooled[:, 0, :])}

        verb, noun, action = pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]
        return {
            "verb": self.verb_classifier(verb),
            "noun": self.noun_classifier(noun),
            "action": self.action_classifier(action),
        }
