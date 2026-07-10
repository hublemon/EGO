"""Tests for observation-clip sampling invariants."""

from __future__ import annotations

from ego.datasets.video_sampling import build_clip_window, sample_anticipation_time_sec


def test_observation_ends_before_target_action_starts():
    window = build_clip_window(
        target_start_frame=300,
        video_fps=30.0,
        frames_per_clip=16,
        frames_per_second=8,
        anticipation_time_sec=1.0,
    )
    assert window.observation_end_sec <= window.target_start_sec
    assert window.frame_indices.max() < 300


def test_anticipation_horizon_matches_requested_value():
    window = build_clip_window(
        target_start_frame=1000,
        video_fps=30.0,
        frames_per_clip=16,
        frames_per_second=8,
        anticipation_time_sec=2.0,
    )
    assert window.anticipation_time_sec == 2.0
    assert abs((window.target_start_sec - window.observation_end_sec) - 2.0) < 1e-6


def test_clip_frame_count_matches_frames_per_clip_config():
    window = build_clip_window(
        target_start_frame=500,
        video_fps=30.0,
        frames_per_clip=32,
        frames_per_second=8,
        anticipation_time_sec=1.0,
    )
    assert len(window.frame_indices) == 32


def test_clamps_safely_when_target_is_near_the_start_of_the_video():
    # target_start_frame is small enough that end_frame - offsets would go
    # negative; every index must still stay within [0, target_start_frame).
    window = build_clip_window(
        target_start_frame=5,
        video_fps=30.0,
        frames_per_clip=16,
        frames_per_second=8,
        anticipation_time_sec=1.0,
    )
    assert (window.frame_indices >= 0).all()
    assert window.frame_indices.max() < 5
    assert window.observation_end_sec <= window.target_start_sec


def test_sample_anticipation_time_sec_fixed_range_returns_constant():
    assert sample_anticipation_time_sec((1.0, 1.0)) == 1.0


def test_sample_anticipation_time_sec_within_range():
    for _ in range(50):
        value = sample_anticipation_time_sec((0.25, 1.75))
        assert 0.25 <= value <= 1.75
