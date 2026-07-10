"""Dataset-agnostic Step 1 sample builders.

Dispatches to the right dataset adapter (``ego.datasets.ek100`` /
``ego.datasets.assembly101``) based on ``dataset.name`` in the resolved
config, and returns ready-to-use train/val ``Dataset`` instances sharing one
train-fit :class:`~ego.datasets.label_mapping.LabelMapping`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ego.common.config import get, require
from ego.common.exceptions import EgoConfigError
from ego.common.paths import expand_path
from ego.datasets.assembly101 import Assembly101Dataset, build_assembly101_manifest
from ego.datasets.ek100 import EK100Dataset, build_ek100_manifest
from ego.datasets.label_mapping import LabelMapping
from ego.step1_action_anticipation.data.transforms import build_transform


@dataclass
class Step1Datasets:
    train: Any
    val: Any | None
    label_mapping: LabelMapping
    missing_videos: list[str]
    num_train_videos: int
    num_val_videos: int


def _resolve(value: str) -> str:
    return str(expand_path(value))


def build_step1_datasets(config: dict) -> Step1Datasets:
    name = require(config, "dataset.name").lower()
    frames_per_clip = require(config, "dataset.frames_per_clip")
    frames_per_second = require(config, "dataset.frames_per_second")
    resolution = require(config, "dataset.resolution")
    anticipation_time_sec = tuple(require(config, "dataset.anticipation_time_sec"))
    train_anticipation_time_sec = tuple(
        get(config, "dataset.train_anticipation_time_sec", anticipation_time_sec)
    )
    repository_dir = get(config, "model.repository_dir")
    if repository_dir:
        repository_dir = _resolve(repository_dir)

    train_transform = build_transform(
        training=True,
        crop_size=resolution,
        random_resize_scale=tuple(get(config, "dataset.random_resize_scale", (0.3, 1.0))),
        reprob=get(config, "dataset.reprob", 0.0),
        auto_augment=get(config, "dataset.auto_augment", False),
        motion_shift=get(config, "dataset.motion_shift", False),
        repository_dir=repository_dir,
    )
    eval_transform = build_transform(training=False, crop_size=resolution, repository_dir=repository_dir)

    if "ek100" in name:
        manifest = build_ek100_manifest(
            base_path=_resolve(require(config, "dataset.video_root")),
            annotation_train_path=_resolve(require(config, "dataset.annotation_train")),
            annotation_val_path=_resolve(require(config, "dataset.annotation_val")),
            file_format=get(config, "dataset.file_format", 0),
            verb_classes_csv=_resolve(v) if (v := get(config, "dataset.verb_classes_csv")) else None,
            noun_classes_csv=_resolve(v) if (v := get(config, "dataset.noun_classes_csv")) else None,
        )
        dataset_cls = EK100Dataset
    elif "assembly101" in name:
        manifest = build_assembly101_manifest(
            base_path=_resolve(require(config, "dataset.video_root")),
            annotation_train_path=_resolve(require(config, "dataset.annotation_train")),
            annotation_val_path=_resolve(require(config, "dataset.annotation_val")),
            views=get(config, "dataset.views", ["e1"]),
        )
        dataset_cls = Assembly101Dataset
    else:
        raise EgoConfigError(f"Unsupported dataset.name: {name!r} (expected 'ek100' or 'assembly101').")

    train_dataset = dataset_cls(
        df=manifest.train_df,
        label_mapping=manifest.label_mapping,
        split="train",
        frames_per_clip=frames_per_clip,
        frames_per_second=frames_per_second,
        resolution=resolution,
        anticipation_time_range=train_anticipation_time_sec,
        transform=train_transform,
    )

    val_dataset = None
    if len(manifest.val_df) > 0:
        val_dataset = dataset_cls(
            df=manifest.val_df,
            label_mapping=manifest.label_mapping,
            split="val",
            frames_per_clip=frames_per_clip,
            frames_per_second=frames_per_second,
            resolution=resolution,
            anticipation_time_range=anticipation_time_sec,
            transform=eval_transform,
        )

    return Step1Datasets(
        train=train_dataset,
        val=val_dataset,
        label_mapping=manifest.label_mapping,
        missing_videos=manifest.missing_videos,
        num_train_videos=manifest.num_train_videos,
        num_val_videos=manifest.num_val_videos,
    )
