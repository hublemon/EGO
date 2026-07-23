from __future__ import annotations

import pandas as pd
import torch

from ego.datasets.video_sampling import sample_adaptive_multirate_frame_indices
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset
from ego.step1_action_anticipation.goalstep.build_goalstep_adaptive_transition_index import build_split


def test_adaptive_multirate_sampling_is_causal_chronological_and_fixed_length():
    indices, positions, terminal = sample_adaptive_multirate_frame_indices(
        start_sec=10.0,
        end_sec=22.0,
        video_fps=30.0,
        global_frames=24,
        terminal_frames=8,
        terminal_window_sec=2.0,
    )
    assert len(indices) == len(positions) == len(terminal) == 32
    assert (indices[1:] >= indices[:-1]).all()
    assert int(terminal.sum()) == 8
    assert float(positions.min()) == 0.0
    assert float(positions.max()) == 1.0
    assert int(indices.min()) >= int(10.0 * 30.0)
    assert int(indices.max()) <= int(22.0 * 30.0)


def test_adaptive_sampling_never_rounds_past_fractional_cutoff():
    _, end_sec, fps = 10.0, 21.017, 29.97
    indices, _, _ = sample_adaptive_multirate_frame_indices(
        start_sec=10.0,
        end_sec=end_sec,
        video_fps=fps,
        global_frames=24,
        terminal_frames=8,
        terminal_window_sec=2.0,
    )
    assert float(indices.max()) / fps <= end_sec + 1e-9


def test_adaptive_index_keeps_only_close_same_level_successor():
    labels = pd.DataFrame([
        {"video_uid": "v1", "split": "train", "level": "step", "start_time": 0.0,
         "end_time": 5.0, "verb_label": 1, "noun_label": 1},
        {"video_uid": "v1", "split": "train", "level": "step", "start_time": 5.5,
         "end_time": 8.0, "verb_label": 2, "noun_label": 2},
        {"video_uid": "v1", "split": "train", "level": "step", "start_time": 12.0,
         "end_time": 16.0, "verb_label": 3, "noun_label": 3},
        {"video_uid": "v1", "split": "train", "level": "step", "start_time": 16.2,
         "end_time": 20.0, "verb_label": 4, "noun_label": 4},
    ])
    actions = {(i, i): i for i in range(1, 5)}
    output, stats = build_split(
        labels,
        "train",
        {"v1": "scenario"},
        actions,
        gap_ratio=0.2,
        max_gap_sec=2.0,
        min_action_sec=1.0,
        guard_sec=0.25,
        max_observation_sec=32.0,
    )
    assert len(output) == 2
    assert output["action_label"].tolist() == [2, 4]
    assert output["observed_action_label"].tolist() == [1, 3]
    assert (output["target_start_sec"] > output["obs_end_sec"]).all()
    assert stats["excluded_adaptive_gap"] == 1


def test_adaptive_index_excludes_overlap_short_and_over_threshold_pairs():
    labels = pd.DataFrame([
        {"video_uid": "overlap", "split": "train", "level": "step", "start_time": 0.0,
         "end_time": 5.0, "verb_label": 1, "noun_label": 1},
        {"video_uid": "overlap", "split": "train", "level": "step", "start_time": 4.9,
         "end_time": 8.0, "verb_label": 2, "noun_label": 2},
        {"video_uid": "short", "split": "train", "level": "step", "start_time": 0.0,
         "end_time": 0.9, "verb_label": 1, "noun_label": 1},
        {"video_uid": "short", "split": "train", "level": "step", "start_time": 0.95,
         "end_time": 3.0, "verb_label": 2, "noun_label": 2},
        {"video_uid": "boundary", "split": "train", "level": "step", "start_time": 0.0,
         "end_time": 5.0, "verb_label": 1, "noun_label": 1},
        {"video_uid": "boundary", "split": "train", "level": "step", "start_time": 6.0,
         "end_time": 8.0, "verb_label": 2, "noun_label": 2},
        {"video_uid": "too_far", "split": "train", "level": "step", "start_time": 0.0,
         "end_time": 5.0, "verb_label": 1, "noun_label": 1},
        {"video_uid": "too_far", "split": "train", "level": "step", "start_time": 6.001,
         "end_time": 8.0, "verb_label": 2, "noun_label": 2},
        # Different levels never become a candidate pair.
        {"video_uid": "cross_level", "split": "train", "level": "step", "start_time": 0.0,
         "end_time": 5.0, "verb_label": 1, "noun_label": 1},
        {"video_uid": "cross_level", "split": "train", "level": "substep", "start_time": 5.0,
         "end_time": 8.0, "verb_label": 2, "noun_label": 2},
    ])
    output, stats = build_split(
        labels,
        "train",
        {video_uid: "scenario" for video_uid in labels["video_uid"].unique()},
        {(1, 1): 1, (2, 2): 2},
        gap_ratio=0.2,
        max_gap_sec=2.0,
        min_action_sec=1.0,
        guard_sec=0.25,
        max_observation_sec=32.0,
    )
    assert output["video_uid"].tolist() == ["boundary"]
    assert float(output.iloc[0]["inter_action_gap_sec"]) == 1.0
    assert stats["excluded_overlap"] == 1
    assert stats["excluded_short_observed_action"] == 1
    assert stats["excluded_adaptive_gap"] == 1


def test_feature_cache_roundtrips_adaptive_temporal_metadata(tmp_path):
    sample_id = "v1_0"
    torch.save({
        "features": torch.ones(4, 8, dtype=torch.float16),
        "verb_id": 1,
        "noun_id": 2,
        "action_id": 3,
        "anticipation_time_sec": 1.0,
        "sample_id": sample_id,
        "observation_duration_sec": torch.tensor(12.0),
        "observed_action_duration_sec": torch.tensor(18.0),
        "frame_time_positions": torch.linspace(0, 1, 32),
        "frame_terminal_mask": torch.tensor([False] * 24 + [True] * 8),
        "annotation_level_id": torch.tensor(1),
    }, tmp_path / f"{sample_id}.pt")

    item = FeatureCacheDataset([sample_id], tmp_path)[0]
    assert item["video"].dtype == torch.float32
    assert float(item["observation_duration_sec"]) == 12.0
    assert float(item["observed_action_duration_sec"]) == 18.0
    assert tuple(item["frame_time_positions"].shape) == (32,)
    assert int(item["frame_terminal_mask"].sum()) == 8
    assert int(item["annotation_level_id"]) == 1
    assert "inter_action_gap_sec" not in item
    assert "target_horizon_sec" not in item
