"""Leakage-safe visual-history residual head for GoalStep next-action prediction.

The current observation is represented by frozen, precomputed visual logits.
The history-only ablation receives *only* summaries of earlier action
segments.  A second contextual pass receives the pooled current segment plus
those history tokens and produces the gated residual.  This separation makes
the three required ablations exact:

``visual``
    The immutable next-action probe logits.
``history``
    A learned prediction from prior visual segments; it cannot see the current
    segment or its GT label.
``fused``
    ``visual + tanh(g_field) * contextual_history`` with one zero-initialized
    gate for each of verb, noun, and action.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


HEADS = ("verb", "noun", "action")


class AttentiveSegmentPooler(nn.Module):
    """One shared cross-attention block pooling ``T`` tokens into one token."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, segment_tokens: torch.Tensor) -> torch.Tensor:
        """Pool ``[B, S, T, D]`` summaries to ``[B, S, D]``."""
        if segment_tokens.ndim != 4:
            raise ValueError(
                "segment_tokens must have shape [batch, segments, tokens, dim]; "
                f"got {tuple(segment_tokens.shape)}"
            )
        batch, segments, tokens, dim = segment_tokens.shape
        if tokens < 1:
            raise ValueError("Each segment must contain at least one token")
        flattened = segment_tokens.reshape(batch * segments, tokens, dim)
        query = self.query.expand(batch * segments, -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(query),
            self.token_norm(flattened),
            self.token_norm(flattened),
            need_weights=False,
        )
        pooled = query + attended
        pooled = pooled + self.mlp(self.output_norm(pooled))
        return pooled[:, 0].reshape(batch, segments, dim)


class HistoryContextResidualHead(nn.Module):
    """Predict from previous visual segments and gate into frozen visual logits.

    ``current_and_history_summaries`` is deliberately shaped ``[B, 1+K, T,
    D]`` so the data contract can be audited in one tensor.  Position zero is
    the current observation.  The returned ``outputs["history"]`` comes from a
    separate transformer pass that excludes position zero.  The residual used
    by ``outputs["fused"]`` comes from a current+history contextual pass.
    """

    def __init__(
        self,
        *,
        num_classes: Mapping[str, int],
        embed_dim: int = 1024,
        max_history: int = 8,
        segment_pooler_heads: int = 16,
        transformer_heads: int = 16,
        transformer_layers: int = 2,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        segment_dropout: float = 0.3,
        recency_scale_sec: float = 300.0,
    ) -> None:
        super().__init__()
        if set(num_classes) != set(HEADS):
            raise ValueError(f"num_classes must contain exactly {HEADS}; got {sorted(num_classes)}")
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        if not 0.0 <= segment_dropout < 1.0:
            raise ValueError("segment_dropout must be in [0, 1)")
        if recency_scale_sec <= 0:
            raise ValueError("recency_scale_sec must be > 0")

        self.num_classes = {head: int(num_classes[head]) for head in HEADS}
        if any(value <= 0 for value in self.num_classes.values()):
            raise ValueError("Every output class count must be positive")
        self.embed_dim = int(embed_dim)
        self.max_history = int(max_history)
        self.segment_dropout = float(segment_dropout)
        self.recency_scale_sec = float(recency_scale_sec)

        self.segment_pooler = AttentiveSegmentPooler(
            embed_dim=self.embed_dim,
            num_heads=segment_pooler_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
        )
        self.recency_mlp = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.level_embedding = nn.Embedding(2, self.embed_dim)
        self.slot_embedding = nn.Parameter(torch.zeros(1, self.max_history, self.embed_dim))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        self.history_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.history_query, std=0.02)
        self.current_token_type = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.current_token_type, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=transformer_heads,
            dim_feedforward=int(round(self.embed_dim * transformer_mlp_ratio)),
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.embed_dim),
        )
        self.history_classifiers = nn.ModuleDict(
            {head: nn.Linear(self.embed_dim, self.num_classes[head]) for head in HEADS}
        )
        self.field_gates = nn.ParameterDict(
            {head: nn.Parameter(torch.zeros(())) for head in HEADS}
        )

    def _validate_inputs(
        self,
        current_and_history_summaries: torch.Tensor,
        history_mask: torch.Tensor,
        history_delta_t_sec: torch.Tensor,
        history_level_id: torch.Tensor,
        visual_logits: Mapping[str, torch.Tensor],
    ) -> None:
        if current_and_history_summaries.ndim != 4:
            raise ValueError("summaries must be [B, 1+K, T, D]")
        batch, segments, _, dim = current_and_history_summaries.shape
        if segments != self.max_history + 1 or dim != self.embed_dim:
            raise ValueError(
                f"Expected summaries [B,{self.max_history + 1},T,{self.embed_dim}], "
                f"got {tuple(current_and_history_summaries.shape)}"
            )
        expected = (batch, self.max_history)
        for name, value in (
            ("history_mask", history_mask),
            ("history_delta_t_sec", history_delta_t_sec),
            ("history_level_id", history_level_id),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}; got {tuple(value.shape)}")
        mask = history_mask.bool()
        if (history_delta_t_sec[mask] <= 0).any():
            raise ValueError("Valid history slots require delta_t_sec > 0")
        if (history_delta_t_sec[~mask] != 0).any():
            raise ValueError("Padded history slots require delta_t_sec == 0")
        if mask.any() and ((history_level_id[mask] < 0) | (history_level_id[mask] > 1)).any():
            raise ValueError("Valid history level IDs must encode step=0 or substep=1")
        if set(visual_logits) != set(HEADS):
            raise ValueError(f"visual_logits must contain exactly {HEADS}")
        for head in HEADS:
            expected_logits = (batch, self.num_classes[head])
            if tuple(visual_logits[head].shape) != expected_logits:
                raise ValueError(
                    f"visual_logits[{head!r}] must be {expected_logits}; "
                    f"got {tuple(visual_logits[head].shape)}"
                )

    def _apply_segment_dropout(self, history_mask: torch.Tensor) -> torch.Tensor:
        mask = history_mask.bool()
        if not self.training or self.segment_dropout == 0.0:
            return mask
        keep = torch.rand(mask.shape, device=mask.device) >= self.segment_dropout
        dropped = mask & keep

        # Do not turn a sample with real history into an empty-history sample.
        # Restore its most recent valid slot (the manifest is chronological).
        had_history = mask.any(dim=1)
        lost_all = had_history & ~dropped.any(dim=1)
        if lost_all.any():
            positions = torch.arange(self.max_history, device=mask.device).expand_as(mask)
            most_recent = positions.masked_fill(~mask, -1).argmax(dim=1)
            rows = lost_all.nonzero(as_tuple=False).flatten()
            dropped[rows, most_recent[rows]] = True
        return dropped

    def forward(
        self,
        current_and_history_summaries: torch.Tensor,
        history_mask: torch.Tensor,
        history_delta_t_sec: torch.Tensor,
        history_level_id: torch.Tensor,
        visual_logits: Mapping[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        self._validate_inputs(
            current_and_history_summaries,
            history_mask,
            history_delta_t_sec,
            history_level_id,
            visual_logits,
        )
        if not torch.isfinite(current_and_history_summaries).all():
            raise ValueError("Non-finite segment summaries detected")

        # One shared attentive block pools both current and historical
        # segments.  The current token is then excluded from the history-only
        # pass and included only in the contextual residual pass.
        pooled_segments = self.segment_pooler(current_and_history_summaries)
        pooled_current = pooled_segments[:, :1]
        pooled_history = pooled_segments[:, 1:]
        effective_mask = self._apply_segment_dropout(history_mask)

        delta = history_delta_t_sec.to(dtype=pooled_history.dtype).clamp_min(0.0)
        delta = torch.log1p(delta) / torch.log1p(delta.new_tensor(self.recency_scale_sec))
        recency = self.recency_mlp(delta.unsqueeze(-1))
        safe_levels = history_level_id.clamp(min=0, max=1).long()
        history_tokens = (
            pooled_history
            + recency
            + self.level_embedding(safe_levels)
            + self.slot_embedding
        )
        history_tokens = history_tokens.masked_fill(~effective_mask.unsqueeze(-1), 0.0)

        batch = history_tokens.shape[0]
        query = self.history_query.expand(batch, -1, -1)
        transformer_input = torch.cat([query, history_tokens], dim=1)
        padding_mask = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=effective_mask.device), ~effective_mask],
            dim=1,
        )
        encoded = self.history_transformer(
            transformer_input,
            src_key_padding_mask=padding_mask,
        )[:, 0]
        history_outputs = {
            head: self.history_classifiers[head](encoded) for head in HEADS
        }

        contextual_input = torch.cat(
            [pooled_current + self.current_token_type, history_tokens], dim=1
        )
        contextual_padding_mask = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=effective_mask.device), ~effective_mask],
            dim=1,
        )
        contextual_encoded = self.history_transformer(
            contextual_input,
            src_key_padding_mask=contextual_padding_mask,
        )[:, 0]
        contextual_outputs = {
            head: self.history_classifiers[head](contextual_encoded) for head in HEADS
        }
        # Evaluation-only causal control: use the same contextual branch and
        # learned gates, but remove every history token. Any fused gain beyond
        # this control is attributable to history rather than to adding a
        # second (compressed) view of the current segment.
        visual_outputs = {head: visual_logits[head].detach() for head in HEADS}
        if self.training:
            # The control is never part of the optimization objective; avoid
            # a third Transformer pass during training.
            current_only_outputs = visual_outputs
        else:
            current_only_encoded = self.history_transformer(
                pooled_current + self.current_token_type
            )[:, 0]
            current_only_contextual_outputs = {
                head: self.history_classifiers[head](current_only_encoded) for head in HEADS
            }
            current_only_outputs = {
                head: visual_outputs[head]
                + torch.tanh(self.field_gates[head]).to(visual_outputs[head].dtype)
                * current_only_contextual_outputs[head].to(visual_outputs[head].dtype)
                for head in HEADS
            }
        fused_outputs = {
            head: visual_outputs[head]
            + torch.tanh(self.field_gates[head]).to(visual_outputs[head].dtype)
            * contextual_outputs[head].to(visual_outputs[head].dtype)
            for head in HEADS
        }
        return {
            "visual": visual_outputs,
            "history": history_outputs,
            "current_only": current_only_outputs,
            "contextual_history": contextual_outputs,
            "fused": fused_outputs,
        }

    @torch.no_grad()
    def gate_values(self) -> dict[str, dict[str, float]]:
        return {
            head: {
                "raw": float(self.field_gates[head].detach().cpu()),
                "tanh": float(torch.tanh(self.field_gates[head]).detach().cpu()),
            }
            for head in HEADS
        }

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "embed_dim": self.embed_dim,
            "max_history": self.max_history,
            "history_only_sees_current": False,
            "causal_control": "current_only uses identical contextual branch with all history removed",
            "fused_residual_context": "current pooled token + history tokens",
            "visual_path": "frozen_precomputed_logits",
            "fusion": "visual + tanh(field_gate) * contextual_history",
            "field_gate_initialization": 0.0,
            "segment_dropout": self.segment_dropout,
            "recency_transform": f"log1p(delta_t_sec)/log1p({self.recency_scale_sec})",
            "num_classes": dict(self.num_classes),
        }
