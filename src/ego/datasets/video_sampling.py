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


def sample_uniform_frame_indices(
    start_sec: float,
    end_sec: float,
    video_fps: float,
    num_frames: int,
) -> np.ndarray:
    """Evenly-spaced native-frame indices covering ``[start_sec, end_sec]``.

    Used for datasets (e.g. Ego4D LTA) whose observation window is already
    fixed by an upstream index-building step, rather than derived here from
    an anticipation horizon -- see ``build_clip_window`` for that case.
    ``end_sec < start_sec`` (a window that got clamped to zero width) still
    returns ``num_frames`` copies of ``start_sec``'s frame rather than
    raising, so callers can decide how to handle degenerate windows.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if video_fps <= 0:
        raise ValueError("video_fps must be > 0")

    start_frame = max(0, start_sec) * video_fps
    end_frame = max(start_sec, end_sec) * video_fps
    frame_indices = np.linspace(start_frame, end_frame, num=num_frames)
    return np.round(frame_indices).astype(np.int64)


def sample_adaptive_multirate_frame_indices(
    start_sec: float,
    end_sec: float,
    video_fps: float,
    global_frames: int = 24,
    terminal_frames: int = 8,
    terminal_window_sec: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a variable-duration action with global and terminal resolution.

    ``global_frames`` cover the full observation, while ``terminal_frames``
    densely cover its last ``terminal_window_sec`` seconds.  The combined
    indices are stably sorted to preserve chronological encoder input.  Frame
    duplication is intentional for short clips: it keeps the tensor length
    fixed without reaching beyond the causal observation boundary.

    Returns ``(frame_indices, normalized_times, terminal_mask)``.  Normalized
    times are in ``[0, 1]`` relative to this sample's observation window and
    are suitable as probe-only metadata; they contain no future gap value.
    """
    if global_frames < 1 or terminal_frames < 1:
        raise ValueError("global_frames and terminal_frames must both be >= 1")
    if video_fps <= 0:
        raise ValueError("video_fps must be > 0")
    if terminal_window_sec <= 0:
        raise ValueError("terminal_window_sec must be > 0")

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    terminal_start = max(start_sec, end_sec - terminal_window_sec)
    global_times = np.linspace(start_sec, end_sec, num=global_frames)
    terminal_times = np.linspace(terminal_start, end_sec, num=terminal_frames)
    times = np.concatenate([global_times, terminal_times])
    terminal_mask = np.concatenate([
        np.zeros(global_frames, dtype=np.bool_),
        np.ones(terminal_frames, dtype=np.bool_),
    ])
    order = np.argsort(times, kind="stable")
    times = times[order]
    terminal_mask = terminal_mask[order]

    # Convert requested timestamps into a fully causal discrete-frame window.
    # Rounding the endpoint can choose the first frame *after* ``end_sec``;
    # floor it instead and clamp every sample between the first frame at/after
    # ``start_sec`` and the last frame at/before ``end_sec``.  The adaptive
    # index guarantees windows far longer than one native frame, but keep a
    # defensive fallback for synthetic/very-low-FPS inputs.
    first_frame = int(np.ceil(start_sec * video_fps - 1e-9))
    last_frame = int(np.floor(end_sec * video_fps + 1e-9))
    if last_frame < first_frame:
        first_frame = last_frame
    frame_indices = np.floor(times * video_fps + 1e-9).astype(np.int64)
    frame_indices = np.clip(frame_indices, first_frame, last_frame)

    # Metadata describes the frames that will actually be decoded, including
    # native-FPS quantisation (rather than the ideal floating-point requests).
    duration = max(end_sec - start_sec, 1e-8)
    decoded_times = frame_indices.astype(np.float64) / video_fps
    normalized_times = np.clip(
        (decoded_times - start_sec) / duration, 0.0, 1.0
    ).astype(np.float32)
    return frame_indices, normalized_times, terminal_mask
