"""V-JEPA2 backbone wrapper: frozen encoder + autoregressive predictor.

Ported from the validated V-JEPA2 action-anticipation prototype
(``evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py``
in the EvE/V-JEPA2 repo). The encoder/predictor source itself is vendored
under ``third_party/vjepa2`` (see ``third_party/versions.yaml``); this module
only adds it to ``sys.path`` and wraps it for EGO's config/logging conventions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from ego.common.exceptions import EgoCheckpointError
from ego.common.logging import step_log
from ego.common.paths import third_party_dir


def default_repository_dir() -> Path:
    return third_party_dir() / "vjepa2"


def ensure_vjepa2_on_path(repository_dir: str | Path | None = None) -> Path:
    """Add the vendored (or externally configured) V-JEPA2 repo root to ``sys.path``."""
    repo = Path(repository_dir) if repository_dir else default_repository_dir()
    repo = repo.resolve()
    if not repo.is_dir():
        raise EgoCheckpointError(
            f"V-JEPA2 repository directory not found: {repo}. "
            "Set model.repository_dir in the config or VJEPA2_REPO_DIR in .env."
        )
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _get_model_modules(repository_dir: str | Path | None, use_v2_1: bool):
    ensure_vjepa2_on_path(repository_dir)
    if use_v2_1:
        import app.vjepa_2_1.models.predictor as vit_pred
        import app.vjepa_2_1.models.vision_transformer as vit
    else:
        import src.models.predictor as vit_pred
        import src.models.vision_transformer as vit
    return vit, vit_pred


class AnticipativeVJEPA2(nn.Module):
    """Frozen encoder + predictor, stepped forward to a future anticipation horizon.

    ``forward(x, anticipation_times)`` encodes the observed clip, then runs the
    predictor autoregressively (``num_steps`` chunks) so its target mask-token
    positions reach ``anticipation_times`` seconds past the last observed
    frame. The concatenation of encoder tokens and predicted future tokens is
    what gets pooled by :class:`~ego.step1_action_anticipation.models.attentive_probe.AttentiveProbe`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        frames_per_second: int,
        crop_size: int,
        patch_size: int,
        tubelet_size: int,
        no_predictor: bool = False,
        num_output_frames: int = 2,
        num_steps: int = 1,
        no_encoder: bool = False,
    ) -> None:
        super().__init__()
        if no_predictor and no_encoder:
            raise ValueError("AnticipativeVJEPA2 must use the predictor or the encoder (or both).")
        self.encoder = encoder
        self.predictor = predictor
        self.grid_size = crop_size // patch_size
        self.tubelet_size = tubelet_size
        self.no_predictor = no_predictor
        self.num_output_frames = max(num_output_frames, tubelet_size)
        self.frames_per_second = frames_per_second
        self.num_steps = num_steps
        self.no_encoder = no_encoder
        self.embed_dim = encoder.embed_dim

    def forward(self, x: torch.Tensor, anticipation_times: torch.Tensor) -> torch.Tensor:
        """
        :param x: video clip, shape [B, C, T, H, W]
        :param anticipation_times: seconds into the future to predict, shape [B]
        """
        x_full = self.encoder(x)
        if self.no_predictor:
            return x_full

        B, N, D_full = x_full.size()
        embed_dim = self.encoder.embed_dim
        use_hierarchical = D_full > embed_dim
        x = x_full[:, :, -embed_dim:] if use_hierarchical else x_full

        x_accumulate = torch.rand(B, 0, embed_dim, device=x.device) if self.no_encoder else x.clone()

        ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

        anticipation_steps = (anticipation_times * self.frames_per_second / self.tubelet_size).to(torch.int64)
        skip_positions = N + int(self.grid_size**2) * anticipation_steps

        N_pred = int(self.grid_size**2 * (self.num_output_frames // self.tubelet_size))
        tgt_positions = torch.arange(N_pred, device=x.device).unsqueeze(0).repeat(B, 1)
        tgt_positions = tgt_positions + skip_positions.unsqueeze(1).repeat(1, N_pred)

        x_pred_input = x_full
        for _ in range(self.num_steps):
            pred_out = self.predictor(x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions)
            x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out

            x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)

            x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
            x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)

        return x_accumulate


def _load_state_dict(module: nn.Module, pretrained_dict: dict, name: str) -> None:
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    own = module.state_dict()
    missing, mismatched = [], []
    for k, v in own.items():
        if k not in pretrained_dict:
            missing.append(k)
        elif pretrained_dict[k].shape != v.shape:
            mismatched.append(k)
            pretrained_dict[k] = v
    msg = module.load_state_dict(pretrained_dict, strict=False)
    step_log(
        1,
        "Model",
        f"{name} loaded ({len(missing)} missing keys, {len(mismatched)} shape-mismatched keys): {msg}",
    )


def load_vjepa2_backbone(
    frames_per_clip: int,
    frames_per_second: int,
    resolution: int,
    checkpoint: str | Path,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    repository_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> AnticipativeVJEPA2:
    """Build the frozen V-JEPA2 encoder+predictor from a checkpoint and freeze it for inference."""
    repository_dir = ensure_vjepa2_on_path(repository_dir)
    step_log(1, "Model", f"V-JEPA2 repository: {repository_dir}")
    step_log(1, "Model", f"Checkpoint: {checkpoint}")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise EgoCheckpointError(f"V-JEPA2 checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    use_v2_1 = bool(model_kwargs.get("use_v2_1", False))
    vit, vit_pred = _get_model_modules(repository_dir, use_v2_1)

    enc_kwargs = model_kwargs["encoder"]
    encoder = vit.__dict__[enc_kwargs["model_name"]](
        img_size=resolution, num_frames=frames_per_clip, **enc_kwargs
    )
    _load_state_dict(encoder, ckpt[enc_kwargs["checkpoint_key"]], "Encoder")

    prd_kwargs = model_kwargs["predictor"]
    teacher_embed_dim = prd_kwargs.get("teacher_embed_dim")
    n_output_distillation = prd_kwargs.get("n_output_distillation", 4)
    prd_out_embed_dim = teacher_embed_dim // n_output_distillation if teacher_embed_dim is not None else None
    predictor = vit_pred.__dict__[prd_kwargs["model_name"]](
        img_size=resolution,
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        out_embed_dim=prd_out_embed_dim,
        **prd_kwargs,
    )
    _load_state_dict(predictor, ckpt[prd_kwargs["checkpoint_key"]], "Predictor")

    model = AnticipativeVJEPA2(
        encoder=encoder,
        predictor=predictor,
        frames_per_second=frames_per_second,
        crop_size=resolution,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        **wrapper_kwargs,
    )

    if hasattr(predictor, "hierarchical_layers") and len(predictor.hierarchical_layers) > 1:
        encoder.return_hierarchical = True

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    step_log(1, "Model", "Backbone frozen: True")
    step_log(
        1,
        "Model",
        f"Input shape: [B, 3, {frames_per_clip}, {resolution}, {resolution}] @ {frames_per_second} fps",
    )
    return model
