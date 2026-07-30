"""Focused tests for the GoalStep-space LTA auxiliary index contract."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from ego.step1_action_anticipation.goalstep.build_lta_aux_index import (
    build_lta_aux_index,
    copy_registry_for_output,
    load_goalstep_val_videos,
    load_taxonomy_lookup,
    normalize_lta_text,
)


def _action(
    clip_uid: str,
    video_uid: str,
    action_idx: int,
    start: float,
    end: float,
    verb: str,
    noun: str,
    *,
    parent: float = 100.0,
) -> dict:
    return {
        "clip_uid": clip_uid,
        "video_uid": video_uid,
        "action_idx": action_idx,
        "clip_parent_start_sec": parent,
        "action_clip_start_sec": start,
        "action_clip_end_sec": end,
        "verb": verb,
        "noun": noun,
        "source_split": "train",
        "source_row": action_idx,
    }


@pytest.fixture
def registry() -> dict:
    # Deliberately non-identity dense IDs verify that raw taxonomy labels and
    # loss-head IDs are not accidentally conflated.
    return {
        "num_verbs": 3,
        "num_nouns": 4,
        "num_actions": 6,
        "verb_classes": {"7": 2},
        "noun_classes": {"9": 3},
        "action_classes": {"7|9": 5},
    }


@pytest.fixture
def lookups(tmp_path, registry):
    verb_path = tmp_path / "verb.csv"
    noun_path = tmp_path / "noun.csv"
    with verb_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_id", "class_key", "members", "segment_count"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "class_id": "7",
                "class_key": "take",
                "members": "fetch;get;take",
                "segment_count": "1",
            }
        )
    with noun_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_id", "class_key", "members", "segment_count"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "class_id": "9",
                "class_key": "tray",
                "members": "plate|tray",
                "segment_count": "1",
            }
        )
    return (
        load_taxonomy_lookup(verb_path, registry["verb_classes"]),
        load_taxonomy_lookup(noun_path, registry["noun_classes"]),
    )


def _synthetic_actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Retained both-match pair. Operational decoder fields stay
            # clip-relative; explicit audit fields become parent-video times.
            _action("keep", "video-keep", 0, 0.0, 5.0, "touch", "dough"),
            _action(
                "keep",
                "video-keep",
                1,
                6.0,
                8.0,
                "take_(pick,_grab,_get)",
                "tray",
            ),
            # Partial target: kept only under match_policy=any.
            _action("partial", "video-partial", 0, 0.0, 5.0, "touch", "dough"),
            _action("partial", "video-partial", 1, 6.0, 8.0, "take", "unknown"),
            # GoalStep validation leakage: must be removed even though mapped.
            _action("leak", "video-val", 0, 0.0, 5.0, "touch", "dough"),
            _action("leak", "video-val", 1, 6.0, 8.0, "take", "plate"),
            # A2.end-1 (=9) is after A3.start (=7): recognition, not future.
            _action("overlap", "video-overlap", 0, 0.0, 10.0, "touch", "dough"),
            _action("overlap", "video-overlap", 1, 7.0, 9.0, "get", "tray"),
            # First adjacent annotation overlaps A2, so the strict target is
            # the first later action_idx starting at/after A2.end.
            _action("skip", "video-skip", 0, 0.0, 10.0, "touch", "dough"),
            _action("skip", "video-skip", 1, 7.0, 12.0, "unknown", "unknown"),
            _action("skip", "video-skip", 2, 10.0, 12.0, "take", "tray"),
        ]
    )


def test_normalization_and_literal_handoff_alias_policy(lookups):
    verb_lookup, noun_lookup = lookups
    assert normalize_lta_text("take_(pick,_grab,_get)") == "take"
    assert verb_lookup["take"] == "7"
    # Literal §3.1 only splits comma/pipe. The real CSV's semicolon compound
    # remains conservative rather than silently expanding this alias.
    assert "fetch" not in verb_lookup
    assert noun_lookup["plate"] == "9"


def test_default_both_contract_uses_a2_endpoint_and_a3_target(registry, lookups):
    output, stats = build_lta_aux_index(
        _synthetic_actions(),
        verb_lookup=lookups[0],
        noun_lookup=lookups[1],
        registry=registry,
        goalstep_val_videos={"video-val"},
    )

    assert list(output["cache_sample_id"]) == [
        "ltaaux_keep_0_1",
        "ltaaux_skip_0_2",
    ]
    row = output[output["clip_uid"] == "keep"].iloc[0]
    assert row["obs_start_clip_sec"] == pytest.approx(0.0)
    assert row["obs_end_clip_sec"] == pytest.approx(4.0)
    assert row["obs_start_sec"] == pytest.approx(0.0)
    assert row["obs_end_sec"] == pytest.approx(4.0)
    assert row["target_start_sec"] == pytest.approx(6.0)
    assert row["obs_start_video_sec"] == pytest.approx(100.0)
    assert row["obs_end_video_sec"] == pytest.approx(104.0)
    assert row["target_start_video_sec"] == pytest.approx(106.0)
    assert row["target_horizon_sec"] == pytest.approx(2.0)
    assert row["verb_label"] == 7
    assert row["noun_label"] == 9
    assert row["verb_id"] == 2
    assert row["noun_id"] == 3
    assert row["action_id"] == 5
    assert row["action_label"] == 5
    assert bool(row["verb_mask"] and row["noun_mask"] and row["action_mask"])
    skipped = output[output["clip_uid"] == "skip"].iloc[0]
    assert skipped["target_action_idx"] == 2
    assert skipped["target_action_rank_gap"] == 2
    assert skipped["skipped_overlapping_actions"] == 1
    assert skipped["target_start_sec"] == skipped["observed_action_end_sec"]
    assert skipped["obs_end_sec"] < skipped["target_start_sec"]
    assert stats["output_rows"] == 2
    assert stats["mapped_any_candidates"] == 4
    assert stats["mapped_both_candidates"] == 3
    assert stats["excluded_no_strict_later_target"] == 2
    assert stats["skipped_overlapping_annotations_total"] == 1
    assert stats["excluded_goalstep_val_rows"] == 1


def test_any_policy_preserves_partial_ids_and_masks(registry, lookups):
    output, stats = build_lta_aux_index(
        _synthetic_actions(),
        verb_lookup=lookups[0],
        noun_lookup=lookups[1],
        registry=registry,
        goalstep_val_videos={"video-val"},
        match_policy="any",
    )

    assert set(output["clip_uid"]) == {"keep", "partial", "skip"}
    partial = output[output["clip_uid"] == "partial"].iloc[0]
    assert partial["verb_label"] == 7
    assert partial["verb_id"] == 2
    assert partial["noun_label"] == -1
    assert partial["noun_id"] == -1
    assert partial["action_id"] == -1
    assert bool(partial["verb_mask"])
    assert not bool(partial["noun_mask"])
    assert not bool(partial["action_mask"])
    assert stats["output_rows"] == 3


def test_goalstep_val_loader_fails_closed_on_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="validation-video set is empty"):
        load_goalstep_val_videos(empty)


def test_registry_copy_is_byte_identical(tmp_path):
    source = tmp_path / "registry.json"
    source.write_bytes(b'{\n  "num_actions": 293\n}\n')
    destination = copy_registry_for_output(source, tmp_path / "index")
    assert destination.read_bytes() == source.read_bytes()


def test_taxonomy_collision_fails_closed(tmp_path):
    path = tmp_path / "verb.csv"
    path.write_text(
        "class_id,class_key,members\n"
        "7,get,take\n"
        "8,take,take\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="maps to both"):
        load_taxonomy_lookup(path, {"7": 0, "8": 1})
