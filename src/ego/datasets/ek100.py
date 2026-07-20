"""EPIC-KITCHENS-100 dataset adapter scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ego.common.exceptions import EgoDatasetError
from ego.contracts.observation import Observation
from ego.datasets.base import EgoActionAnticipationDataset
from ego.datasets.label_mapping import (
    LabelMapping,
    build_label_mapping,
    filter_to_known_pairs,
)
from ego.datasets.video_sampling import build_clip_window, sample_anticipation_time_sec

DATASET_NAME = "EK100"


def resolve_video_path(
    base_path: str | Path, participant_id: str, video_id: str, file_format: int = 0
) -> Path:
    """Resolve an EK100 video file path.

    ``file_format=0``: ``base_path/participant_id/videos/video_id.MP4``
    ``file_format=1``: ``base_path/participant_id/video_id.MP4``
    """
    base = Path(base_path)
    if file_format == 0:
        return base / participant_id / "videos" / f"{video_id}.MP4"
    return base / participant_id / f"{video_id}.MP4"


def _load_canonical_text(path: str | Path | None, id_col: str, text_col: str = "key") -> dict[int, str]:
    if path is None or not Path(path).is_file():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df[id_col], df[text_col]))


@dataclass
class EK100Manifest:
    """Resolved train/val annotation frames plus the label mapping fit on train."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    label_mapping: LabelMapping
    missing_videos: list[str]
    num_train_videos: int
    num_val_videos: int


def build_ek100_manifest(
    base_path: str | Path,
    annotation_train_path: str | Path,
    annotation_val_path: str | Path,
    file_format: int = 0,
    verb_classes_csv: str | Path | None = None,
    noun_classes_csv: str | Path | None = None,
) -> EK100Manifest:
    """Load, resolve, and filter EK100 train/val annotations.

    Val rows whose ``(verb_class, noun_class)`` pair was never seen in train
    are dropped (the model has no way to predict a class it wasn't fit on),
    and the returned :class:`LabelMapping` is deterministic (sorted ids).
    """
    train_df = pd.read_csv(annotation_train_path)
    val_df = pd.read_csv(annotation_val_path)

    def _resolve(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        paths, missing = [], []
        for video_id in df["video_id"]:
            pid = video_id.split("_")[0]
            fpath = resolve_video_path(base_path, pid, video_id, file_format)
            paths.append(str(fpath))
            if not fpath.is_file():
                missing.append(str(fpath))
        out = df.copy()
        out["video_path"] = paths
        return out, missing

    train_df, train_missing = _resolve(train_df)
    val_df, val_missing = _resolve(val_df)

    train_df = train_df[~train_df["video_path"].isin(set(train_missing))].reset_index(drop=True)
    val_df = val_df[~val_df["video_path"].isin(set(val_missing))].reset_index(drop=True)

    if len(train_df) == 0:
        raise EgoDatasetError(
            "No EK100 train videos found on disk after path resolution; "
            f"checked base_path={base_path} with file_format={file_format}."
        )

    train_pairs = list(zip(train_df["verb_class"].astype(int), train_df["noun_class"].astype(int)))
    verb_text = _load_canonical_text(verb_classes_csv, "verb_class") or dict(
        zip(train_df["verb_class"], train_df["verb"])
    )
    noun_text = _load_canonical_text(noun_classes_csv, "noun_class") or dict(
        zip(train_df["noun_class"], train_df["noun"])
    )
    label_mapping = build_label_mapping(train_pairs, verb_text=verb_text, noun_text=noun_text)

    known_pairs = set(train_pairs)
    val_rows = filter_to_known_pairs(
        val_df.to_dict("records"), known_pairs, verb_key="verb_class", noun_key="noun_class"
    )
    val_df = pd.DataFrame(val_rows, columns=val_df.columns) if val_rows else val_df.iloc[0:0]

    return EK100Manifest(
        train_df=train_df,
        val_df=val_df,
        label_mapping=label_mapping,
        missing_videos=sorted(set(train_missing) | set(val_missing)),
        num_train_videos=int(train_df["video_id"].nunique()),
        num_val_videos=int(val_df["video_id"].nunique()),
    )


class EK100Dataset(EgoActionAnticipationDataset):
    """One item == one annotated EK100 action segment's observation clip."""

    def __init__(
        self,
        df: pd.DataFrame,
        label_mapping: LabelMapping,
        split: str,
        frames_per_clip: int,
        frames_per_second: int,
        resolution: int,
        anticipation_time_range: tuple[float, float],
        transform: Any | None = None,
    ) -> None:
        if len(df) == 0:
            raise EgoDatasetError(f"EK100Dataset[{split}] built from an empty annotation frame.")
        self._rows = df.reset_index(drop=True)
        self._label_mapping = label_mapping
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.frames_per_second = frames_per_second
        self.resolution = resolution
        self.anticipation_time_range = anticipation_time_range
        self.transform = transform
        self._fps_cache: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def get_label_mapping(self) -> LabelMapping:
        return self._label_mapping

    def _video_fps(self, video_path: str) -> float:
        if video_path not in self._fps_cache:
            from decord import VideoReader, cpu

            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            self._fps_cache[video_path] = float(vr.get_avg_fps())
        return self._fps_cache[video_path]

    def _sample_id(self, row: pd.Series, index: int) -> str:
        narration_id = row.get("narration_id")
        return str(narration_id) if narration_id is not None else f"{row['video_id']}_{index}"

    def get_sample_metadata(self, index: int) -> Observation:
        row = self._rows.iloc[index]
        window = build_clip_window(
            target_start_frame=int(row["start_frame"]),
            video_fps=self._video_fps(row["video_path"]),
            frames_per_clip=self.frames_per_clip,
            frames_per_second=self.frames_per_second,
            anticipation_time_sec=sample_anticipation_time_sec(self.anticipation_time_range),
        )
        return Observation(
            sample_id=self._sample_id(row, index),
            dataset=DATASET_NAME,
            split=self.split,
            video_id=str(row["video_id"]),
            observation_start_sec=window.observation_start_sec,
            observation_end_sec=window.observation_end_sec,
            target_start_sec=window.target_start_sec,
            anticipation_time_sec=window.anticipation_time_sec,
            frames_per_clip=self.frames_per_clip,
            frames_per_second=self.frames_per_second,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        from decord import VideoReader, cpu

        row = self._rows.iloc[index]
        video_path = row["video_path"]
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        vfps = self._fps_cache.setdefault(video_path, float(vr.get_avg_fps()))

        window = build_clip_window(
            target_start_frame=int(row["start_frame"]),
            video_fps=vfps,
            frames_per_clip=self.frames_per_clip,
            frames_per_second=self.frames_per_second,
            anticipation_time_sec=sample_anticipation_time_sec(self.anticipation_time_range),
        )
        buffer = vr.get_batch(window.frame_indices.tolist()).asnumpy()
        video = self.transform(buffer) if self.transform is not None else torch.from_numpy(buffer)

        verb_raw = int(row["verb_class"])
        noun_raw = int(row["noun_class"])

        return {
            "video": video,
            "verb_id": self._label_mapping.encode_verb(verb_raw),
            "noun_id": self._label_mapping.encode_noun(noun_raw),
            "action_id": self._label_mapping.encode_action(verb_raw, noun_raw),
            "verb_id_raw": verb_raw,
            "noun_id_raw": noun_raw,
            "anticipation_time_sec": window.anticipation_time_sec,
            "observation_start_sec": window.observation_start_sec,
            "observation_end_sec": window.observation_end_sec,
            "target_start_sec": window.target_start_sec,
            "sample_id": self._sample_id(row, index),
            "video_id": str(row["video_id"]),
        }
