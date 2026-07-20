"""Inference scaffold for Step 1 action anticipation.

Runs a trained checkpoint over a validation/test split and writes the Top-K
verb/noun/action candidate distribution that Step 2 consumes, matching
``schemas/step1_candidates.schema.json``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ego.common.config import get, load_config, require
from ego.common.exceptions import EgoConfigError
from ego.common.io import write_jsonl
from ego.common.logging import step_log
from ego.common.paths import expand_path
from ego.common.seed import set_seed
from ego.contracts.candidates import ActionCandidate, StepOneCandidateRecord
from ego.datasets.label_mapping import LabelMapping
from ego.step1_action_anticipation.data.build_samples import build_step1_datasets
from ego.step1_action_anticipation.data.collator import anticipation_collate
from ego.step1_action_anticipation.metrics import prediction_entropy
from ego.step1_action_anticipation.models import AnticipationHead, load_vjepa2_backbone


def _action_candidates(
    probs: torch.Tensor, logits: torch.Tensor, mapping: LabelMapping, k: int
) -> list[ActionCandidate]:
    top_probs, top_ids = probs.topk(min(k, probs.numel()))
    out = []
    for rank, (p, uid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
        raw_verb, raw_noun = mapping.inv_action_classes[uid]
        out.append(
            ActionCandidate(
                rank=rank,
                verb=mapping.verb_text.get(raw_verb),
                noun=mapping.noun_text.get(raw_noun),
                verb_id=mapping.verb_classes.get(raw_verb),
                noun_id=mapping.noun_classes.get(raw_noun),
                action_id=uid,
                logit=logits[uid].item(),
                probability=p,
            )
        )
    return out


def _head_candidates(
    probs: torch.Tensor, logits: torch.Tensor, decode_text, k: int, id_field: str
) -> list[ActionCandidate]:
    top_probs, top_ids = probs.topk(min(k, probs.numel()))
    out = []
    for rank, (p, uid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
        kwargs = {"rank": rank, "verb": None, "noun": None, "probability": p, "logit": logits[uid].item()}
        kwargs[id_field] = uid
        kwargs["verb" if id_field == "verb_id" else "noun"] = decode_text(uid)
        out.append(ActionCandidate(**kwargs))
    return out


def infer(config_path: str) -> dict:
    config = load_config(config_path)
    set_seed(get(config, "experiment.seed", 42))
    step_log(1, "Infer", "Config loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_step1_datasets(config)
    mapping = datasets.label_mapping
    eval_dataset = datasets.val
    if eval_dataset is None or len(eval_dataset) == 0:
        raise EgoConfigError("Inference dataset resolved to zero samples.")

    top_k = require(config, "inference.top_k")
    batch_size = get(config, "inference.batch_size", 8)
    step_log(1, "Infer", f"Number of samples: {len(eval_dataset)}")
    step_log(1, "Infer", f"Top-K: {top_k}")

    backbone = load_vjepa2_backbone(
        frames_per_clip=require(config, "dataset.frames_per_clip"),
        frames_per_second=require(config, "dataset.frames_per_second"),
        resolution=require(config, "dataset.resolution"),
        checkpoint=expand_path(require(config, "model.checkpoint")),
        model_kwargs=require(config, "model.model_kwargs"),
        wrapper_kwargs=get(config, "model.wrapper_kwargs", {}),
        repository_dir=get(config, "model.repository_dir"),
        device=device,
    )
    classifier_cfg = get(config, "model.classifier", {})
    head = AnticipationHead(
        num_verb_classes=mapping.num_verbs,
        num_noun_classes=mapping.num_nouns,
        num_action_classes=mapping.num_actions,
        embed_dim=backbone.embed_dim,
        num_heads=classifier_cfg.get("num_heads", 16),
        depth=classifier_cfg.get("num_probe_blocks", 4),
        repository_dir=get(config, "model.repository_dir"),
    ).to(device)

    head_checkpoint = expand_path(require(config, "model.head_checkpoint"))
    state = torch.load(head_checkpoint, map_location=device)
    head.load_state_dict(state["model_state"])
    head.eval()
    step_log(1, "Infer", f"Head checkpoint: {head_checkpoint}")

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get(config, "dataset.num_workers", 2),
        collate_fn=anticipation_collate,
    )

    output_path = expand_path(require(config, "inference.output_path"))
    dataset_name = require(config, "dataset.name")
    checkpoint_str = str(head_checkpoint)
    config_str = str(Path(config_path).resolve())

    records: list[dict] = []
    example = None
    with torch.no_grad():
        for batch in loader:
            clips = batch["video"].to(device)
            ant_times = batch["anticipation_time_sec"].to(device)
            logits = head(backbone(clips, ant_times))

            verb_probs = torch.softmax(logits["verb"], dim=-1).cpu()
            noun_probs = torch.softmax(logits["noun"], dim=-1).cpu()
            action_probs = torch.softmax(logits["action"], dim=-1).cpu()
            verb_logits_cpu = logits["verb"].cpu()
            noun_logits_cpu = logits["noun"].cpu()
            action_logits_cpu = logits["action"].cpu()
            entropy = prediction_entropy(logits["action"]).cpu()

            for i in range(clips.size(0)):
                gt_verb_raw = int(batch["verb_id_raw"][i])
                gt_noun_raw = int(batch["noun_id_raw"][i])
                gt = {
                    "verb_id": mapping.verb_classes.get(gt_verb_raw),
                    "verb": mapping.verb_text.get(gt_verb_raw),
                    "noun_id": mapping.noun_classes.get(gt_noun_raw),
                    "noun": mapping.noun_text.get(gt_noun_raw),
                    "action_id": mapping.action_classes.get((gt_verb_raw, gt_noun_raw)),
                }
                record = StepOneCandidateRecord(
                    sample_id=batch["sample_id"][i],
                    dataset=dataset_name,
                    split="val",
                    video_id=batch["video_id"][i],
                    observation_start_sec=float(batch["observation_start_sec"][i]),
                    observation_end_sec=float(batch["observation_end_sec"][i]),
                    target_start_sec=float(batch["target_start_sec"][i]),
                    anticipation_time_sec=float(batch["anticipation_time_sec"][i]),
                    entropy=float(entropy[i]),
                    action_candidates=_action_candidates(
                        action_probs[i], action_logits_cpu[i], mapping, top_k
                    ),
                    verb_candidates=_head_candidates(
                        verb_probs[i], verb_logits_cpu[i], mapping.decode_verb_text, top_k, "verb_id"
                    ),
                    noun_candidates=_head_candidates(
                        noun_probs[i], noun_logits_cpu[i], mapping.decode_noun_text, top_k, "noun_id"
                    ),
                    gt=gt,
                    checkpoint=checkpoint_str,
                    config_path=config_str,
                )
                record_dict = record.to_dict()
                records.append(record_dict)
                if example is None:
                    example = record_dict

    write_jsonl(output_path, records)
    step_log(1, "Infer", f"Output path: {output_path}")
    step_log(1, "Infer", f"Example candidate output: {example}")
    return {"output_path": str(output_path), "num_samples": len(records)}
