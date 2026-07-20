"""Smoke test for the tiny Step 1 pipeline: no real video, no real checkpoint.

Exercises the real (non-mocked) code path from a synthetic backbone output
through the classifier head, Top-K extraction, JSONL export, and schema
validation -- everything except video decoding and the pretrained V-JEPA2
weights, which are replaced with a random tensor and a randomly-initialized
head respectively.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from ego.common.io import write_jsonl
from ego.contracts.candidates import StepOneCandidateRecord
from ego.datasets.label_mapping import build_label_mapping
from ego.step1_action_anticipation.infer import _action_candidates, _head_candidates
from ego.step1_action_anticipation.metrics import prediction_entropy
from ego.step1_action_anticipation.models import AnticipationHead

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipelines"))
from step1_to_step2 import load_step1_candidates  # noqa: E402

TINY_ANNOTATIONS = [
    {"sample_id": "TINY_01_0", "video_id": "TINY_01", "verb_id": 0, "noun_id": 10},
    {"sample_id": "TINY_01_1", "video_id": "TINY_01", "verb_id": 1, "noun_id": 11},
    {"sample_id": "TINY_02_0", "video_id": "TINY_02", "verb_id": 2, "noun_id": 10},
]


def test_tiny_end_to_end_pipeline(tmp_path):
    # 1. Tiny annotation -> label mapping.
    pairs = [(row["verb_id"], row["noun_id"]) for row in TINY_ANNOTATIONS]
    mapping = build_label_mapping(pairs, verb_text={0: "take", 1: "put-down", 2: "open"}, noun_text={10: "cup", 11: "plate"})

    # 2. Tiny/mock feature (stand-in for the real V-JEPA2 backbone output).
    batch_size, num_tokens, embed_dim = len(TINY_ANNOTATIONS), 8, 16
    mock_features = torch.randn(batch_size, num_tokens, embed_dim)

    # 3. Prediction head (real architecture, randomly initialized).
    head = AnticipationHead(
        num_verb_classes=mapping.num_verbs,
        num_noun_classes=mapping.num_nouns,
        num_action_classes=mapping.num_actions,
        embed_dim=embed_dim,
        num_heads=2,
        depth=1,
    )
    head.eval()
    with torch.no_grad():
        logits = head(mock_features)

    verb_probs = torch.softmax(logits["verb"], dim=-1)
    noun_probs = torch.softmax(logits["noun"], dim=-1)
    action_probs = torch.softmax(logits["action"], dim=-1)
    entropy = prediction_entropy(logits["action"])

    # 4. Top-K extraction -> StepOneCandidateRecord.
    records = []
    for i, row in enumerate(TINY_ANNOTATIONS):
        gt = {
            "verb_id": mapping.verb_classes[row["verb_id"]],
            "verb": mapping.verb_text.get(row["verb_id"]),
            "noun_id": mapping.noun_classes[row["noun_id"]],
            "noun": mapping.noun_text.get(row["noun_id"]),
            "action_id": mapping.action_classes[(row["verb_id"], row["noun_id"])],
        }
        record = StepOneCandidateRecord(
            sample_id=row["sample_id"],
            dataset="TINY",
            split="smoke",
            video_id=row["video_id"],
            observation_start_sec=0.0,
            observation_end_sec=2.0,
            target_start_sec=3.0,
            anticipation_time_sec=1.0,
            entropy=float(entropy[i]),
            action_candidates=_action_candidates(action_probs[i], logits["action"][i], mapping, k=2),
            verb_candidates=_head_candidates(verb_probs[i], logits["verb"][i], mapping.decode_verb_text, 2, "verb_id"),
            noun_candidates=_head_candidates(noun_probs[i], logits["noun"][i], mapping.decode_noun_text, 2, "noun_id"),
            gt=gt,
        )
        records.append(record.to_dict())

    # 5. JSONL export.
    output_path = tmp_path / "action_candidates.jsonl"
    write_jsonl(output_path, records)

    # 6. Schema validation via the Step 1 -> Step 2 pipeline seam.
    validated = load_step1_candidates(output_path)
    assert len(validated) == len(TINY_ANNOTATIONS)
    for record in validated:
        assert len(record["candidates"]) == 2
        assert record["observation_end_sec"] <= record["target_start_sec"]
