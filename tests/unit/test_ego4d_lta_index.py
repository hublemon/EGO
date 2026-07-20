"""Tests for the Ego4D LTA Z=1 index-building pipeline, against synthetic fixtures.

These validate the arithmetic, boundary handling, label registry reuse, and
split logic in isolation. Parsing against the *real* Ego4D LTA JSON schema is
not covered here -- see PILOT.md's "Known open item" and
``ego.datasets.ego4d``'s module docstring.
"""

from __future__ import annotations

import json

import pytest

from ego.datasets.ego4d import (
    build_z1_index,
    index_scenario_lookup,
    load_lta_taxonomy,
    load_video_scenarios,
    parse_lta_annotations,
    register_action_labels,
    split_dev_heldout,
    z1_sample_id,
)
from ego.datasets.label_mapping import build_label_mapping


def _action(clip_uid, video_uid, action_idx, start, end, verb, noun):
    return {
        "clip_uid": clip_uid,
        "video_uid": video_uid,
        "action_idx": action_idx,
        "clip_parent_start_sec": 0.0,
        "action_clip_start_sec": start,
        "action_clip_end_sec": end,
        "verb_label": verb,
        "noun_label": noun,
        "verb": f"verb{verb}",
        "noun": f"noun{noun}",
    }


@pytest.fixture
def annotations_path(tmp_path):
    records = [
        # clipA: 3 actions, none near the clip start after the first -> clean truncate-free case
        _action("clipA", "vidA", 0, 0.0, 2.0, 0, 0),
        _action("clipA", "vidA", 1, 5.0, 7.0, 1, 1),
        _action("clipA", "vidA", 2, 10.0, 12.0, 0, 1),
        # clipB: second action is early enough to truncate but still usable
        _action("clipB", "vidB", 0, 0.0, 1.0, 2, 2),
        _action("clipB", "vidB", 1, 2.0, 4.0, 1, 1),
        # clipC: second action is so early the truncated window is unusably short
        _action("clipC", "vidC", 0, 0.0, 1.0, 0, 0),
        _action("clipC", "vidC", 1, 1.0, 3.0, 1, 1),
    ]
    path = tmp_path / "fho_lta_train.json"
    path.write_text(json.dumps({"clips": records}))
    return path


@pytest.fixture
def taxonomy_path(tmp_path):
    path = tmp_path / "fho_lta_taxonomy.json"
    path.write_text(json.dumps({"verbs": ["v0", "v1", "v2"], "nouns": ["n0", "n1", "n2"]}))
    return path


@pytest.fixture
def ego4d_json_path(tmp_path):
    path = tmp_path / "ego4d.json"
    path.write_text(
        json.dumps(
            {
                "videos": [
                    {"video_uid": "vidA", "scenarios": ["Cooking"]},
                    {"video_uid": "vidB", "scenarios": ["Cleaning", "Cooking"]},
                ]
            }
        )
    )
    return path


def test_load_lta_taxonomy(taxonomy_path):
    tax = load_lta_taxonomy(taxonomy_path)
    assert tax.num_verbs == 3
    assert tax.num_nouns == 3
    assert tax.verb_text(1) == "v1"


def test_load_video_scenarios_takes_first_tag(ego4d_json_path):
    mapping = load_video_scenarios(ego4d_json_path)
    assert mapping["vidA"] == "Cooking"
    assert mapping["vidB"] == "Cleaning"  # first of multiple tags


def test_parse_lta_annotations_flattens_and_sorts(annotations_path):
    df = parse_lta_annotations(annotations_path)
    assert len(df) == 7
    assert list(df["clip_uid"]) == sorted(df["clip_uid"])
    # within a clip, sorted by action_idx
    clip_a = df[df["clip_uid"] == "clipA"]
    assert list(clip_a["action_idx"]) == [0, 1, 2]


def test_parse_lta_annotations_raises_clear_error_on_missing_field(tmp_path):
    path = tmp_path / "bad.json"
    bad_record = _action("clipX", "vidX", 0, 0.0, 1.0, 0, 0)
    del bad_record["verb_label"]
    path.write_text(json.dumps({"clips": [bad_record]}))
    with pytest.raises(Exception, match="verb_label"):
        parse_lta_annotations(path)


def test_build_z1_index_excludes_first_action_per_clip(annotations_path):
    df = parse_lta_annotations(annotations_path)
    index_df, stats = build_z1_index(df, tau_a=1.0, l_obs=3.5, min_obs_sec=0.5, boundary_policy="truncate")
    assert stats.excluded_first_action == 3  # one per clip
    assert set(index_df["clip_uid"]) <= {"clipA", "clipB", "clipC"}


def test_build_z1_index_truncate_policy(annotations_path):
    df = parse_lta_annotations(annotations_path)
    index_df, stats = build_z1_index(df, tau_a=1.0, l_obs=3.5, min_obs_sec=0.5, boundary_policy="truncate")

    # clipA action_idx=1 (start=5.0): obs_end=4.0, obs_start=0.5 -> no truncation needed
    row_a1 = index_df[(index_df.clip_uid == "clipA") & (index_df.obs_end_sec == 4.0)].iloc[0]
    assert row_a1.boundary_flag == False  # noqa: E712
    assert row_a1.obs_start_sec == pytest.approx(0.5)

    # clipB action_idx=1 (start=2.0): obs_end=1.0, obs_start=1.0-3.5=-2.5 -> truncated to 0.0, window=1.0s, kept
    row_b1 = index_df[index_df.clip_uid == "clipB"].iloc[0]
    assert row_b1.boundary_flag == True  # noqa: E712
    assert row_b1.obs_start_sec == 0.0
    assert row_b1.obs_end_sec == pytest.approx(1.0)

    # clipC action_idx=1 (start=1.0): obs_end=0.0, obs_start=-3.5 -> truncated to 0.0, window=0.0s < min_obs_sec -> excluded
    assert "clipC" not in set(index_df["clip_uid"])
    assert stats.excluded_min_obs == 1
    assert stats.truncated == 1  # only clipB's sample counted as a kept-but-truncated row
    assert stats.kept == 3  # clipA action_idx=1, clipA action_idx=2, clipB action_idx=1


def test_build_z1_index_exclude_policy_drops_any_boundary_case(annotations_path):
    df = parse_lta_annotations(annotations_path)
    index_df, stats = build_z1_index(df, tau_a=1.0, l_obs=3.5, min_obs_sec=0.5, boundary_policy="exclude")
    assert "clipB" not in set(index_df["clip_uid"])
    assert "clipC" not in set(index_df["clip_uid"])
    assert stats.truncated == 0
    assert set(index_df["clip_uid"]) == {"clipA"}


def test_build_z1_index_never_produces_target_before_observation_end(annotations_path):
    df = parse_lta_annotations(annotations_path)
    index_df, _ = build_z1_index(df, tau_a=1.0, l_obs=3.5, min_obs_sec=0.5, boundary_policy="truncate")
    for _, row in index_df.iterrows():
        assert row.obs_start_sec <= row.obs_end_sec


def test_build_z1_index_attaches_scenario(annotations_path, ego4d_json_path):
    df = parse_lta_annotations(annotations_path)
    scenario_map = load_video_scenarios(ego4d_json_path)
    index_df, _ = build_z1_index(df, tau_a=1.0, l_obs=3.5, boundary_policy="truncate", scenario_map=scenario_map)
    assert set(index_df[index_df.clip_uid == "clipA"]["scenario"]) == {"Cooking"}
    # vidC has no entry in ego4d.json -> falls back to "unknown"
    unknown_rows = index_df[index_df.clip_uid == "clipC"]
    if len(unknown_rows):
        assert set(unknown_rows["scenario"]) == {"unknown"}


def test_register_action_labels_matches_build_label_mapping(annotations_path):
    df = parse_lta_annotations(annotations_path)
    index_df, _ = build_z1_index(df, tau_a=1.0, l_obs=3.5, boundary_policy="truncate")
    mapping = register_action_labels(index_df)

    pairs = list(zip(index_df["verb_label"].astype(int), index_df["noun_label"].astype(int)))
    expected = build_label_mapping(pairs)
    assert mapping.verb_classes == expected.verb_classes
    assert mapping.noun_classes == expected.noun_classes
    assert mapping.action_classes == expected.action_classes


def test_split_dev_heldout_is_deterministic_and_clip_disjoint():
    import pandas as pd

    val_df = pd.DataFrame({"clip_uid": [f"clip{i}" for i in range(20) for _ in range(3)]})
    dev1, heldout1 = split_dev_heldout(val_df, dev_fraction=0.8, seed=7)
    dev2, heldout2 = split_dev_heldout(val_df, dev_fraction=0.8, seed=7)

    assert set(dev1["clip_uid"]) == set(dev2["clip_uid"])  # deterministic given the seed
    assert set(dev1["clip_uid"]).isdisjoint(set(heldout1["clip_uid"]))  # no clip leakage
    assert len(dev1) + len(heldout1) == len(val_df)
    assert abs(len(set(dev1["clip_uid"])) - 16) <= 1  # ~80% of 20 clips


def test_z1_sample_id_and_scenario_lookup_are_consistent():
    import pandas as pd

    index_df = pd.DataFrame(
        {"clip_uid": ["c0", "c0", "c1"], "scenario": ["Cooking", "Cooking", "Cleaning"]}
    )
    lookup = index_scenario_lookup(index_df)
    assert lookup[z1_sample_id("c0", 0)] == "Cooking"
    assert lookup[z1_sample_id("c0", 1)] == "Cooking"
    assert lookup[z1_sample_id("c1", 2)] == "Cleaning"
