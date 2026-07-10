"""Sliding-window Step 1 demo: run a trained checkpoint across whole videos.

Reproduces the old prototype's `outputs/<participant>/<video_id>_1sec.csv`
output (`sliding_window_anticipation.py` / `wm_output_pipeline.py` in
EvE/V-JEPA2): step through a video at fixed-second intervals, anticipate the
action `anticipation_sec` seconds ahead of each step, match it against the
nearest ground-truth annotation, and record Top-1/Top-5 correctness.

Unlike `ego step1 infer` (which scores pre-annotated action segments from a
val CSV), this walks the *entire* video at a fixed cadence -- useful for a
qualitative/accuracy demo on a couple of videos without needing a trained
EGO-native checkpoint first.

Usage:
    python scripts/step1/sliding_window_demo.py \
        --config configs/step1/ek100_vjepa2.yaml \
        --checkpoint checkpoints/step1/legacy_ek100_vitl256/best_action.pt \
        --videos P01_13 P02_13
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from decord import VideoReader, cpu

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.io import ensure_dir  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ek100 import resolve_video_path  # noqa: E402
from ego.datasets.video_sampling import build_clip_window  # noqa: E402
from ego.step1_action_anticipation.data.transforms import build_transform  # noqa: E402
from ego.step1_action_anticipation.legacy_checkpoint import (  # noqa: E402
    build_legacy_ek100_label_mapping,
    load_legacy_head_state_dict,
)
from ego.step1_action_anticipation.models import AnticipationHead, load_vjepa2_backbone  # noqa: E402


def find_gt(video_ann: pd.DataFrame, pred_frame: int):
    """3-tier ground-truth lookup: containing segment -> next segment -> most recent past segment."""
    within = video_ann[(video_ann.start_frame <= pred_frame) & (pred_frame <= video_ann.stop_frame)]
    if len(within):
        return within.iloc[0]
    future = video_ann[video_ann.start_frame > pred_frame]
    if len(future):
        return future.sort_values("start_frame").iloc[0]
    past = video_ann[video_ann.stop_frame < pred_frame]
    if len(past):
        return past.sort_values("stop_frame").iloc[-1]
    return None


def topk_ids_and_text(probs: torch.Tensor, inv_classes: dict, text_map: dict, k: int):
    top_probs, top_ids = probs.topk(min(k, probs.numel()))
    raw_ids = [inv_classes[i] for i in top_ids.tolist()]
    texts = [text_map.get(r, "?") for r in raw_ids]
    return raw_ids, texts, top_probs.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/step1/ek100_vjepa2.yaml")
    parser.add_argument(
        "--checkpoint", default="checkpoints/step1/legacy_ek100_vitl256/best_action.pt"
    )
    parser.add_argument("--videos", nargs="+", default=["P01_13", "P02_13"])
    parser.add_argument("--annotations", default="data/annotations/EPIC_100_validation.csv")
    parser.add_argument("--anticipation-sec", type=float, default=1.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/step1/legacy_demo")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_per_clip = require(config, "dataset.frames_per_clip")
    frames_per_second = require(config, "dataset.frames_per_second")
    resolution = require(config, "dataset.resolution")
    base_path = expand_path(require(config, "dataset.video_root"))
    file_format = get(config, "dataset.file_format", 0)

    mapping = build_legacy_ek100_label_mapping(
        train_annotations_path=expand_path(require(config, "dataset.annotation_train")),
        verb_classes_csv=expand_path(get(config, "dataset.verb_classes_csv")),
        noun_classes_csv=expand_path(get(config, "dataset.noun_classes_csv")),
    )
    step_log(1, "SlidingWindowDemo", f"Legacy label space: verb={mapping.num_verbs} noun={mapping.num_nouns} action={mapping.num_actions}")

    backbone = load_vjepa2_backbone(
        frames_per_clip=frames_per_clip,
        frames_per_second=frames_per_second,
        resolution=resolution,
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
    legacy_state = load_legacy_head_state_dict(expand_path(args.checkpoint))
    missing, unexpected = head.load_state_dict(legacy_state, strict=True)
    step_log(1, "SlidingWindowDemo", f"Loaded legacy checkpoint: {args.checkpoint} (missing={missing}, unexpected={unexpected})")
    head.eval()

    transform = build_transform(training=False, crop_size=resolution, repository_dir=get(config, "model.repository_dir"))
    ann = pd.read_csv(expand_path(args.annotations))

    output_dir = ensure_dir(expand_path(args.output_dir))
    overall = {"n": 0, "verb1": 0, "verb5": 0, "noun1": 0, "noun5": 0}

    for video_id in args.videos:
        pid = video_id.split("_")[0]
        video_path = resolve_video_path(base_path, pid, video_id, file_format)
        vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        vfps = vr.get_avg_fps()
        total_frames = len(vr)
        video_ann = ann[ann.video_id == video_id]
        step_log(1, "SlidingWindowDemo", f"{video_id}: {total_frames} frames @ {vfps:.2f}fps, {len(video_ann)} GT segments")

        out_dir = ensure_dir(output_dir / pid)
        out_path = out_dir / f"{video_id}_1sec.csv"
        stats = {"n": 0, "verb1": 0, "verb5": 0, "noun1": 0, "noun5": 0}

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "observation_sec", "observation_frame", "target_frame",
                    "true_verb_id", "true_verb", "true_noun_id", "true_noun",
                    "pred_verb_top1", "pred_verb_top5", "pred_verb_top5_prob",
                    "pred_noun_top1", "pred_noun_top5", "pred_noun_top5_prob",
                    "verb_correct_top1", "verb_correct_top5",
                    "noun_correct_top1", "noun_correct_top5",
                ]
            )

            n_steps = 0
            step_sec = 0.0
            while True:
                obs_frame = int(step_sec * vfps)
                target_frame = obs_frame + round(args.anticipation_sec * vfps)
                if target_frame >= total_frames:
                    break
                if args.max_steps is not None and n_steps >= args.max_steps:
                    break

                window = build_clip_window(
                    target_start_frame=target_frame,
                    video_fps=vfps,
                    frames_per_clip=frames_per_clip,
                    frames_per_second=frames_per_second,
                    anticipation_time_sec=args.anticipation_sec,
                )
                buffer = vr.get_batch(window.frame_indices.tolist()).asnumpy()
                clip = transform(buffer).unsqueeze(0).to(device)
                ant_time = torch.tensor([args.anticipation_sec], device=device)

                with torch.no_grad():
                    logits = head(backbone(clip, ant_time))
                verb_probs = torch.softmax(logits["verb"][0], dim=-1).cpu()
                noun_probs = torch.softmax(logits["noun"][0], dim=-1).cpu()

                verb_ids5, verb_txt5, verb_p5 = topk_ids_and_text(verb_probs, mapping.inv_verb_classes, mapping.verb_text, args.top_k)
                noun_ids5, noun_txt5, noun_p5 = topk_ids_and_text(noun_probs, mapping.inv_noun_classes, mapping.noun_text, args.top_k)

                gt = find_gt(video_ann, target_frame)
                if gt is not None:
                    true_verb_id, true_noun_id = int(gt.verb_class), int(gt.noun_class)
                    true_verb, true_noun = gt.verb, gt.noun
                    verb_c1 = verb_ids5[0] == true_verb_id
                    verb_c5 = true_verb_id in verb_ids5
                    noun_c1 = noun_ids5[0] == true_noun_id
                    noun_c5 = true_noun_id in noun_ids5
                    stats["n"] += 1
                    stats["verb1"] += verb_c1
                    stats["verb5"] += verb_c5
                    stats["noun1"] += noun_c1
                    stats["noun5"] += noun_c5
                else:
                    true_verb_id = true_noun_id = true_verb = true_noun = ""
                    verb_c1 = verb_c5 = noun_c1 = noun_c5 = ""

                writer.writerow(
                    [
                        f"{step_sec:.1f}", obs_frame, target_frame,
                        true_verb_id, true_verb, true_noun_id, true_noun,
                        verb_txt5[0], "|".join(verb_txt5), "|".join(f"{p:.4f}" for p in verb_p5),
                        noun_txt5[0], "|".join(noun_txt5), "|".join(f"{p:.4f}" for p in noun_p5),
                        verb_c1, verb_c5, noun_c1, noun_c5,
                    ]
                )
                n_steps += 1
                step_sec += args.step_sec

        step_log(1, "SlidingWindowDemo", f"{video_id}: wrote {n_steps} rows -> {out_path}")
        if stats["n"]:
            step_log(
                1,
                "SlidingWindowDemo",
                f"{video_id} accuracy (n={stats['n']}): "
                f"verb top1={100*stats['verb1']/stats['n']:.1f}% top5={100*stats['verb5']/stats['n']:.1f}%  "
                f"noun top1={100*stats['noun1']/stats['n']:.1f}% top5={100*stats['noun5']/stats['n']:.1f}%",
            )
        for k in overall:
            overall[k] += stats[k]

    if overall["n"]:
        step_log(
            1,
            "SlidingWindowDemo",
            f"OVERALL accuracy (n={overall['n']}): "
            f"verb top1={100*overall['verb1']/overall['n']:.1f}% top5={100*overall['verb5']/overall['n']:.1f}%  "
            f"noun top1={100*overall['noun1']/overall['n']:.1f}% top5={100*overall['noun5']/overall['n']:.1f}%",
        )


if __name__ == "__main__":
    main()
