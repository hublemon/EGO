from __future__ import annotations

import pandas as pd
import pytest

from ego.step1_action_anticipation.goalstep.build_goalstep_history_index import build_split


def _endpoint() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "video_uid": "v1", "clip_uid": "v1", "obs_end_sec": 1.0,
            "target_start_sec": 0.0, "target_end_sec": 2.0, "matched_level": "step",
        },
        {
            "video_uid": "v1", "clip_uid": "v1", "obs_end_sec": 3.0,
            "target_start_sec": 2.0, "target_end_sec": 4.0, "matched_level": "substep",
        },
        {
            "video_uid": "v1", "clip_uid": "v1", "obs_end_sec": 4.0,
            "target_start_sec": 2.0, "target_end_sec": 5.0, "matched_level": "step",
        },
        {
            "video_uid": "v1", "clip_uid": "v1", "obs_end_sec": 7.0,
            "target_start_sec": 6.0, "target_end_sec": 8.0, "matched_level": "step",
        },
    ])


def _target() -> pd.DataFrame:
    return pd.DataFrame([{
        "video_uid": "v1",
        "clip_uid": "v1",
        "obs_end_sec": 7.0,
        "cache_sample_id": "v1_3",
        "observed_action_start_sec": 6.0,
        "observed_action_end_sec": 8.0,
        "target_start_sec": 9.0,
        "target_end_sec": 10.0,
        "annotation_level": "step",
        "verb_label": 1,
        "noun_label": 2,
        "action_label": 3,
    }])


def test_history_is_left_padded_same_level_and_chronological() -> None:
    targets = _target()
    output, stats = build_split(_endpoint(), targets, history_length=3)

    assert output[targets.columns].equals(targets)
    assert output.loc[0, "history_length"] == 2
    assert output.loc[0, "sample_id"] == "v1_3"
    assert output.loc[0, "current_cache_sample_id"] == "v1_3"
    assert output.loc[0, "history_1_mask"] == False  # noqa: E712
    assert output.loc[0, "history_1_cache_sample_id"] == ""
    assert output.loc[0, "history_2_cache_sample_id"] == "v1_0"
    assert output.loc[0, "history_3_cache_sample_id"] == "v1_2"
    assert output.loc[0, "history_2_delta_t_sec"] == 6.0
    assert output.loc[0, "history_3_delta_t_sec"] == 3.0
    assert output.loc[0, "history_2_level_id"] == 0
    assert output.loc[0, "history_1_level_id"] == -1
    assert output.loc[0, "verb_id"] == 1
    assert output.loc[0, "noun_id"] == 2
    assert output.loc[0, "action_id"] == 3
    assert stats["history_length_histogram"] == {"0": 0, "1": 0, "2": 1, "3": 0}
    assert not any(
        token in column
        for column in output.columns
        if column.startswith("history_")
        for token in ("verb", "noun", "action_label")
    )


def test_current_observation_must_be_strictly_before_next_action() -> None:
    targets = _target()
    targets.loc[0, "target_start_sec"] = targets.loc[0, "obs_end_sec"]
    with pytest.raises(RuntimeError, match="not strictly before A3"):
        build_split(_endpoint(), targets, history_length=3)
