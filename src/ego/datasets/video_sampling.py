"""Video sampling scaffold for egocentric clips.

Core principle (must hold for every sample this module produces): the
observation window ends at or before the target action's start time. Frames
from the target action (or anything after it) are never included in the
observation clip -- see ``ClipWindow`` and ``build_clip_window``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClipWindow:
    """Native-video frame indices plus the timing metadata for one observation clip."""

    frame_indices: np.ndarray
    observation_start_sec: float
    observation_end_sec: float
    target_start_sec: float
    anticipation_time_sec: float


def sample_anticipation_time_sec(time_range: tuple[float, float]) -> float:
    """Draw an anticipation horizon (seconds) from ``[lo, hi]``.

    ``lo == hi`` (as used for validation/inference) returns that fixed value.
    """
    lo, hi = time_range
    if lo == hi:
        return float(lo)
    return random.uniform(lo, hi)


def build_clip_window(
    target_start_frame: int,
    video_fps: float,
    frames_per_clip: int,
    frames_per_second: int,
    anticipation_time_sec: float,
) -> ClipWindow:
    """Build the frame-index window observed before ``target_start_frame``.

    The window spans ``frames_per_clip`` frames strided to approximate
    ``frames_per_second`` (native ``video_fps`` frames are subsampled), and
    ends ``anticipation_time_sec`` seconds before the target action starts.
    Indices are clamped into ``[0, target_start_frame - 1]`` so, even for
    actions near the start of a video or very short anticipation horizons,
    the window can never reach into the target action.
    """
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be >= 1")
    if video_fps <= 0:
        raise ValueError("video_fps must be > 0")

    target_start_sec = target_start_frame / video_fps
    native_stride = video_fps / frames_per_second

    end_frame = target_start_frame - anticipation_time_sec * video_fps
    end_frame = min(end_frame, target_start_frame - 1)
    end_frame = max(end_frame, 0.0)

    offsets = native_stride * np.arange(frames_per_clip - 1, -1, -1)
    frame_indices = end_frame - offsets
    frame_indices = np.clip(frame_indices, 0, max(target_start_frame - 1, 0))
    frame_indices = np.round(frame_indices).astype(np.int64)

    observation_start_sec = float(frame_indices[0]) / video_fps
    observation_end_sec = float(frame_indices[-1]) / video_fps

    return ClipWindow(
        frame_indices=frame_indices,
        observation_start_sec=observation_start_sec,
        observation_end_sec=observation_end_sec,
        target_start_sec=target_start_sec,
        anticipation_time_sec=anticipation_time_sec,
    )
