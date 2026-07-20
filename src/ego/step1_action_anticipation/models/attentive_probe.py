"""Attentive probe: pools frozen backbone tokens into per-task query vectors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ego.step1_action_anticipation.models.vjepa2_backbone import (
    default_repository_dir,
    ensure_vjepa2_on_path,
)


class AttentiveProbe(nn.Module):
    """Cross-attention pooling: backbone tokens ``[B, N, D]`` -> ``[B, num_queries, D]``.

    Wraps V-JEPA2's ``AttentivePooler`` (vendored under ``third_party/vjepa2``).
    ``num_queries=3`` yields independent (verb, noun, action) representations;
    ``num_queries=1`` is used for action-only label spaces.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        num_queries: int = 3,
        repository_dir: str | Path | None = None,
        use_activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        ensure_vjepa2_on_path(repository_dir or default_repository_dir())
        from src.models.attentive_pooler import AttentivePooler

        self.num_queries = num_queries
        self.pooler = AttentivePooler(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooler(x)
