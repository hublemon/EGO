from __future__ import annotations

import pandas as pd
import pytest

from ego.step1_action_anticipation.goalstep.build_goalstep_adaptive_history_index import (
    build_adaptive_split,
)


REGISTRY = {
    "verb_classes": {str(value): value for value in range(5)},
    "noun_classes": {str(value): value for value in range(8)},
    "action_classes": {
        "0|1": 0,
        "1|2": 1,
        "2|3": 2,
        "3|4": 3,
    },
}


def _adaptive_rows() -> pd.DataFrame:
    rows = [
        # Cached step A1_0 -> target A2_0.
        ("step", 0.0, 2.0, 1.75, 2.2, 3.0, 0, 1, 0),
        # Interleaved substep: never enters a step history.
        ("substep", 1.0, 2.5, 2.25, 2.8, 3.2, 1, 2, 1),
        # Cached step A1_1; row 0 is completed before this action starts.
        ("step", 3.0, 5.0, 4.75, 5.3, 6.0, 2, 3, 2),
        # Cached step A1_2; rows 0 and 2 are valid history.
        ("step", 6.0, 8.0, 7.75, 8.2, 9.0, 3, 4, 3),
    ]
    records = []
    for level, start, end, obs_end, target_start, target_end, verb, noun, action in rows:
        records.append(
            {
                "video_uid": "v1",
                "clip_uid": "v1",
                "obs_start_sec": start,
                "obs_end_sec": obs_end,
                "verb_label": verb,
                "noun_label": noun,
                "action_label": action,
                "scenario": "synthetic",
                "boundary_flag": False,
                "annotation_level": level,
                "observed_action_start_sec": start,
                "observed_action_end_sec": end,
                "observed_action_duration_sec": end - start,
                "observed_verb_label": 4,
                "observed_noun_label": 7,
                "observed_action_label": 99,
                "target_start_sec": target_start,
                "target_end_sec": target_end,
                "target_horizon_sec": target_start - obs_end,
                "inter_action_gap_sec": target_start - end,
                "allowed_gap_sec": 1.0,
                "observation_duration_sec": obs_end - start,
                "guard_sec": 0.25,
                "sampling_strategy": "adaptive_multirate_24_8",
            }
        )
    return pd.DataFrame(records)


def test_adaptive_history_reuses_a1_cache_and_preserves_a2_target() -> None:
    source = _adaptive_rows()
    output, stats = build_adaptive_split(
        source,
        history_length=3,
        action_registry=REGISTRY,
    )

    assert len(output) == len(source)
    assert output["cache_sample_id"].tolist() == ["v1_0", "v1_1", "v1_2", "v1_3"]
    assert output["sample_id"].equals(output["cache_sample_id"])
    assert output["current_cache_sample_id"].equals(output["cache_sample_id"])

    current = output.iloc[3]
    assert current["history_length"] == 2
    assert current["history_1_mask"] == False  # noqa: E712
    assert current["history_1_cache_sample_id"] == ""
    assert current["history_2_cache_sample_id"] == "v1_0"
    assert current["history_3_cache_sample_id"] == "v1_2"
    assert current["history_2_delta_t_sec"] == pytest.approx(6.0)
    assert current["history_3_delta_t_sec"] == pytest.approx(3.0)
    assert current["history_2_level_id"] == 0
    assert current["verb_id"] == 3
    assert current["noun_id"] == 4
    assert current["action_id"] == 3
    assert current["audit_current_observation_end_sec"] == pytest.approx(7.75)
    assert current["audit_target_start_sec"] == pytest.approx(8.2)

    assert stats["retained_samples"] == len(source)
    assert stats["history_length_histogram"] == {"0": 2, "1": 1, "2": 1, "3": 0}
    assert not any(
        token in column
        for column in output.columns
        if column.startswith("history_")
        for token in ("verb", "noun", "action_label", "label_id")
    )


def test_adaptive_history_rejects_noncausal_current_observation() -> None:
    source = _adaptive_rows()
    source.loc[3, "target_start_sec"] = source.loc[3, "obs_end_sec"]
    with pytest.raises(RuntimeError, match="not strictly before A3"):
        build_adaptive_split(source, history_length=3, action_registry=REGISTRY)


def test_adaptive_history_rejects_duplicate_observed_action() -> None:
    source = _adaptive_rows()
    duplicate = source.iloc[[0]].copy()
    source = pd.concat([source, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate observed actions"):
        build_adaptive_split(source, history_length=3, action_registry=REGISTRY)
