from __future__ import annotations

from pathlib import Path

import torch

from ego.step1_action_anticipation.data.collator import anticipation_collate
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset
from ego.step1_action_anticipation.goalstep.train_goalstep_z1 import (
    GoalAuxBatchSampler,
    _SourceAnnotatedDataset,
    goal_aux_masked_loss,
)


def test_goal_aux_batch_sampler_uses_every_goalstep_row_once() -> None:
    sampler = GoalAuxBatchSampler(
        11,
        4,
        goalstep_per_batch=3,
        aux_per_batch=2,
        seed=7,
    )
    first = list(sampler)
    goalstep = [index for batch in first for index in batch if index < 11]
    assert sorted(goalstep) == list(range(11))
    assert all(
        sum(index < 11 for index in batch) >= 1
        and sum(index >= 11 for index in batch) == 2
        for batch in first
    )
    assert all(11 <= index < 15 for batch in first for index in batch if index >= 11)

    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_masked_aux_labels_do_not_change_joint_loss() -> None:
    generator = torch.Generator().manual_seed(3)
    logits = {
        "verb": torch.randn(4, 5, generator=generator),
        "noun": torch.randn(4, 7, generator=generator),
        "action": torch.randn(4, 9, generator=generator),
    }
    batch = {
        "is_aux": torch.tensor([False, False, True, True]),
        "verb_id": torch.tensor([1, 2, 3, 4]),
        "noun_id": torch.tensor([2, 3, 4, 5]),
        "action_id": torch.tensor([3, 4, 5, 6]),
        "verb_mask": torch.tensor([True, True, True, True]),
        "noun_mask": torch.tensor([True, True, True, False]),
        "action_mask": torch.tensor([True, True, False, True]),
    }
    first = goal_aux_masked_loss(
        logits, batch, alpha=0.25, gamma=2.0, aux_loss_weight=0.3
    )

    changed = {key: value.clone() for key, value in batch.items()}
    changed["action_id"][2] = 8
    changed["noun_id"][3] = 6
    second = goal_aux_masked_loss(
        logits, changed, alpha=0.25, gamma=2.0, aux_loss_weight=0.3
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.allclose(first[0], first[1] + 0.3 * first[2])


def test_rare_aux_head_is_weighted_by_coverage() -> None:
    logits = {
        "verb": torch.zeros(4, 5),
        "noun": torch.zeros(4, 7),
        "action": torch.zeros(4, 9),
    }
    batch = {
        "is_aux": torch.tensor([False, False, True, True]),
        "verb_id": torch.tensor([1, 2, 0, 0]),
        "noun_id": torch.tensor([2, 3, 0, 0]),
        "action_id": torch.tensor([3, 4, 5, 5]),
        "verb_mask": torch.tensor([True, True, False, False]),
        "noun_mask": torch.tensor([True, True, False, False]),
        "action_mask": torch.tensor([True, True, True, True]),
    }
    full = goal_aux_masked_loss(
        logits, batch, alpha=0.25, gamma=2.0, aux_loss_weight=0.3
    )
    half_batch = {key: value.clone() for key, value in batch.items()}
    half_batch["action_mask"][3] = False
    half = goal_aux_masked_loss(
        logits, half_batch, alpha=0.25, gamma=2.0, aux_loss_weight=0.3
    )
    assert torch.allclose(half[2], 0.5 * full[2])


def test_feature_cache_preserves_partial_supervision_metadata(tmp_path: Path) -> None:
    sample_id = "ltaaux_sample"
    torch.save(
        {
            "features": torch.randn(17, 8).half(),
            "verb_id": 2,
            "noun_id": 4,
            "action_id": 0,
            "anticipation_time_sec": 1.0,
            "sample_id": sample_id,
            "verb_mask": torch.tensor(True),
            "noun_mask": torch.tensor(True),
            "action_mask": torch.tensor(False),
            "supervision_source": "lta_aux",
        },
        tmp_path / f"{sample_id}.pt",
    )
    item = FeatureCacheDataset([sample_id], tmp_path)[0]
    assert bool(item["verb_mask"])
    assert bool(item["noun_mask"])
    assert not bool(item["action_mask"])
    assert item["supervision_source"] == "lta_aux"


def test_mixed_source_batch_has_identical_collation_schema(tmp_path: Path) -> None:
    records = {
        "goal": {
            "features": torch.randn(17, 8).half(),
            "verb_id": 1,
            "noun_id": 2,
            "action_id": 3,
            "anticipation_time_sec": 1.0,
            "sample_id": "goal",
        },
        "aux": {
            "features": torch.randn(17, 8).half(),
            "verb_id": 4,
            "noun_id": 5,
            "action_id": 0,
            "anticipation_time_sec": 1.0,
            "sample_id": "aux",
            "verb_mask": torch.tensor(True),
            "noun_mask": torch.tensor(True),
            "action_mask": torch.tensor(False),
            "supervision_source": "lta_aux",
        },
    }
    for sample_id, record in records.items():
        torch.save(record, tmp_path / f"{sample_id}.pt")

    goal = _SourceAnnotatedDataset(
        FeatureCacheDataset(["goal"], tmp_path), is_aux=False
    )[0]
    aux = _SourceAnnotatedDataset(
        FeatureCacheDataset(["aux"], tmp_path), is_aux=True
    )[0]
    for ordered in ([goal, aux], [aux, goal]):
        batch = anticipation_collate(list(ordered))
        assert batch["supervision_source"] in (
            ["goalstep", "lta_aux"],
            ["lta_aux", "goalstep"],
        )
        assert batch["is_aux"].dtype == torch.bool
        assert batch["action_mask"].tolist() in (
            [True, False],
            [False, True],
        )
