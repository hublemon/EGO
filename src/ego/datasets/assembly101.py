"""Assembly101 dataset adapter scaffold."""

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

DATASET_NAME = "Assembly101"

# Egocentric head-mounted-camera view short names -> filename keyword.
VIEW_MAP = {
    "e1": "HMC_21176875",
    "e2": "HMC_21110305",
    "e3": "HMC_21179183",
    "e4": "HMC_21108298",
}


@dataclass
class Assembly101Manifest:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    label_mapping: LabelMapping
    missing_videos: list[str]
    num_train_videos: int
    num_val_videos: int


def build_assembly101_manifest(
    base_path: str | Path,
    annotation_train_path: str | Path,
    annotation_val_path: str | Path,
    views: list[str] | None = None,
) -> Assembly101Manifest:
    """Load, view-filter, and resolve Assembly101 fine-grained annotations.

    Keeps only monochrome egocentric HMC views (``is_RGB == 0``, filename
    matching one of ``views``, default ``["e1"]``) and, like EK100, drops val
    rows whose ``(verb_id, noun_id)`` pair was never seen in train.
    """
    if views is None:
        views = ["e1"]
    view_keywords = [VIEW_MAP.get(v, v) for v in views]

    def _load(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        matches_view = df["video"].apply(lambda v: any(kw in v for kw in view_keywords))
        return df[matches_view & (df["is_RGB"] == 0)].copy()

    train_df = _load(annotation_train_path)
    val_df = _load(annotation_val_path)

    def _resolve(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        paths, missing = [], []
        for rel_video in df["video"]:
            fpath = Path(base_path) / rel_video
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
            "No Assembly101 train videos found on disk for views="
            f"{views} under base_path={base_path}."
        )

    train_df = train_df.sort_values(["video", "start_frame"]).reset_index(drop=True)
    val_df = val_df.sort_values(["video", "start_frame"]).reset_index(drop=True)
    train_df["narration_id"] = [
        f"{Path(v).parent.name}_{i}" for i, v in enumerate(train_df["video"])
    ]
    val_df["narration_id"] = [f"{Path(v).parent.name}_{i}" for i, v in enumerate(val_df["video"])]

    train_pairs = list(zip(train_df["verb_id"].astype(int), train_df["noun_id"].astype(int)))
    verb_text = dict(zip(train_df["verb_id"], train_df["verb_cls"]))
    noun_text = dict(zip(train_df["noun_id"], train_df["noun_cls"]))
    label_mapping = build_label_mapping(train_pairs, verb_text=verb_text, noun_text=noun_text)

    known_pairs = set(train_pairs)
    val_rows = filter_to_known_pairs(
        val_df.to_dict("records"), known_pairs, verb_key="verb_id", noun_key="noun_id"
    )
    val_df = pd.DataFrame(val_rows, columns=val_df.columns) if val_rows else val_df.iloc[0:0]

    return Assembly101Manifest(
        train_df=train_df,
        val_df=val_df,
        label_mapping=label_mapping,
        missing_videos=sorted(set(train_missing) | set(val_missing)),
        num_train_videos=int(train_df["video"].nunique()),
        num_val_videos=int(val_df["video"].nunique()),
    )


class Assembly101Dataset(EgoActionAnticipationDataset):
    """One item == one annotated Assembly101 fine-grained action segment's observation clip."""

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
            raise EgoDatasetError(f"Assembly101Dataset[{split}] built from an empty annotation frame.")
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
            sample_id=str(row["narration_id"]),
            dataset=DATASET_NAME,
            split=self.split,
            video_id=str(row["video"]),
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

        verb_raw = int(row["verb_id"])
        noun_raw = int(row["noun_id"])

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
            "sample_id": str(row["narration_id"]),
            "video_id": str(row["video"]),
        }
