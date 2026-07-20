"""Video preprocessing transforms for V-JEPA2 input clips.

Ported from the validated V-JEPA2 action-anticipation prototype's
``evals/action_anticipation_frozen/dataloader.py``. The underlying transform
primitives (resize/crop/autoaugment/erase) are vendored under
``third_party/vjepa2/src/datasets/utils/video`` since they must exactly match
what the backbone was trained/evaluated with.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as tv_transforms

from ego.step1_action_anticipation.models.vjepa2_backbone import (
    default_repository_dir,
    ensure_vjepa2_on_path,
)

IMAGENET_NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def tensor_normalize(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)
    return (tensor - mean) / std


class VideoTransform:
    """Train-time augmentation or eval-time resize+center-crop, ending in a normalized C,T,H,W tensor."""

    def __init__(
        self,
        training: bool = True,
        random_horizontal_flip: bool = True,
        random_resize_aspect_ratio: tuple[float, float] = (3 / 4, 4 / 3),
        random_resize_scale: tuple[float, float] = (0.3, 1.0),
        reprob: float = 0.0,
        auto_augment: bool = False,
        motion_shift: bool = False,
        crop_size: int = 224,
        normalize: tuple[tuple[float, ...], tuple[float, ...]] = IMAGENET_NORMALIZE,
        repository_dir: str | Path | None = None,
    ) -> None:
        ensure_vjepa2_on_path(repository_dir or default_repository_dir())
        import src.datasets.utils.video.transforms as video_transforms
        import src.datasets.utils.video.volume_transforms as volume_transforms
        from src.datasets.utils.video.randerase import RandomErasing

        self._video_transforms = video_transforms
        self.training = training

        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(crop_size, crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )
        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift
            if motion_shift
            else video_transforms.random_resized_crop
        )
        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob, mode="pixel", max_count=1, num_splits=1, device="cpu"
        )

    def __call__(self, buffer):
        if not self.training:
            return self.eval_transform(buffer)

        buffer = [tv_transforms.ToPILImage()(frame) for frame in buffer]
        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        buffer = [tv_transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])
        buffer = buffer.permute(3, 0, 1, 2)  # C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = self._video_transforms.horizontal_flip(0.5, buffer)

        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


def build_transform(
    training: bool,
    crop_size: int = 224,
    random_resize_scale: tuple[float, float] = (0.3, 1.0),
    reprob: float = 0.0,
    auto_augment: bool = False,
    motion_shift: bool = False,
    repository_dir: str | Path | None = None,
) -> VideoTransform:
    return VideoTransform(
        training=training,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        repository_dir=repository_dir,
    )
