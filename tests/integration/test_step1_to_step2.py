"""Integration tests for the Step 1 -> Step 2 candidate hand-off artifact."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ego.common.io import write_jsonl
from ego.contracts.candidates import ActionCandidate, StepOneCandidateRecord

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipelines"))
from step1_to_step2 import load_step1_candidates  # noqa: E402


def _make_record(sample_id: str) -> dict:
    candidates = [
        ActionCandidate(rank=1, verb="take", noun="cup", verb_id=3, noun_id=12, action_id=81, probability=0.5),
        ActionCandidate(rank=2, verb="put-down", noun="cup", verb_id=1, noun_id=12, action_id=44, probability=0.3),
    ]
    record = StepOneCandidateRecord(
        sample_id=sample_id,
        dataset="EK100",
        split="validation",
        video_id="P01_01",
        observation_start_sec=31.0,
        observation_end_sec=33.0,
        target_start_sec=34.0,
        anticipation_time_sec=1.0,
        entropy=1.1,
        action_candidates=candidates,
        gt={"verb_id": 3, "verb": "take", "noun_id": 12, "noun": "cup", "action_id": 81},
    )
    return record.to_dict()


def test_step1_output_is_readable_and_schema_valid(tmp_path):
    path = tmp_path / "action_candidates.jsonl"
    write_jsonl(path, [_make_record("P01_01_0"), _make_record("P01_01_1")])

    records = load_step1_candidates(path)

    assert len(records) == 2
    for record in records:
        assert "sample_id" in record
        assert "candidates" in record
        probs = [c["probability"] for c in record["candidates"]]
        assert all(0.0 <= p <= 1.0 for p in probs)
        ranks = [c["rank"] for c in record["candidates"]]
        assert len(ranks) == len(set(ranks))  # no duplicate ranks


def test_step1_output_rejects_malformed_records(tmp_path):
    path = tmp_path / "bad_candidates.jsonl"
    bad_record = _make_record("bad")
    del bad_record["entropy"]  # required by schemas/step1_candidates.schema.json
    write_jsonl(path, [bad_record])

    with pytest.raises(Exception):
        load_step1_candidates(path)
