#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vjepa_ek100_smoke.py — 공식 V-JEPA2 EK100 AC 체크포인트로 F0 후보 jsonl 생성 (합성 클립).

목적: colab_smoke_f0.py 의 `--wm real` 경로. 공식 facebookresearch/vjepa2 의
  - ViT-g/16 384 백본 (vitg-384.pt)
  - EK100 action-anticipation attentive probe (ek100-vitg-384.pt, 39.7 R@5)
를 실제로 GPU 에 로드하고, **합성 32프레임 클립**을 forward 시켜 verb/noun/action top-5 +
softmax likelihood 를 산출한다. 그 결과를 F0 v2 학습 jsonl 스키마로 조립한다.

핵심 사실 (검증된 것):
  - filter_annotations() 의 verb/noun/action 클래스 딕셔너리는 train CSV 전체에서
    파일 존재검사 **이전**에 구성된다 → EK100 원본 영상이 0개여도 probe 헤드 크기가
    정확히 맞아 체크포인트가 로드된다. 우리는 dataloader 를 우회하고 합성 클립을 직접 넣는다.
  - 산출 likelihood 의 '의미'는 무의미(입력이 합성) — 스모크의 목적은 "WM 추론 경로가 이 GPU 에서
    로드·forward 된다"의 검증이다. 학습 품질/정확도와는 무관.

공식 API (main 브랜치, 2026 기준):
  evals.action_anticipation_frozen.models.init_module / init_classifier
  evals.action_anticipation_frozen.dataloader.make_transforms
  evals.action_anticipation_frozen.epickitchens.filter_annotations
  model.forward(clip[B,C,T,H,W], anticipation_time[B]) -> feat[B,N,D]; clf(feat) -> {verb,noun,action}

이 파일은 GPU + 공식 repo 클론 + 체크포인트가 있어야 실행된다 (Colab 전용).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 공식 EK100 inference config (configs/inference/vitg-384/ek100.yaml) 의 값을 코드에 고정.
# yaml 파일 의존을 없애기 위해 in-memory 로 구성한다.
MODULE_NAME = "evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar"
FRAMES_PER_CLIP = 32
FRAMES_PER_SECOND = 8
RESOLUTION = 384
ANTICIPATION_SEC = 1.0
CLF_NUM_HEADS = 16
CLF_NUM_BLOCKS = 4
WRAPPER_KWARGS = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}
PRETRAIN_KWARGS = {
    "encoder": {
        "model_name": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
        "tubelet_size": 2,
        "patch_size": 16,
        "uniform_power": True,
        "use_rope": True,
    },
    "predictor": {
        "model_name": "vit_predictor",
        "checkpoint_key": "predictor",
        "num_frames": 64,
        "depth": 12,
        "num_heads": 12,
        "predictor_embed_dim": 384,
        "num_mask_tokens": 10,
        "uniform_power": True,
        "use_mask_tokens": True,
        "use_sdpa": True,
        "use_silu": False,
        "wide_silu": False,
        "use_rope": True,
    },
}
FRAME_OFFSETS_4 = [4.0, 2.67, 1.33, 0.0]


def strip_module_prefix(sd: dict) -> dict:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def build_id_to_key(csv: Path) -> dict[int, str]:
    df = pd.read_csv(csv)
    return {int(r["id"]): r["key"] for _, r in df.iterrows()}


def top5_with_prob(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    probs = F.softmax(logits.float(), dim=-1)
    vals, idx = probs.topk(5)
    return idx.tolist(), [round(float(v), 6) for v in vals.tolist()]


def decode_topk(out, b, inv_verb, inv_noun, inv_action, verb_id2key, noun_id2key):
    v_idx, v_prob = top5_with_prob(out["verb"][b])
    n_idx, n_prob = top5_with_prob(out["noun"][b])
    a_idx, a_prob = top5_with_prob(out["action"][b])
    top5_verb = [{"rank": r, "verb": verb_id2key[int(inv_verb[i])], "verb_class": int(inv_verb[i]),
                  "likelihood": p} for r, (i, p) in enumerate(zip(v_idx, v_prob), 1)]
    top5_noun = [{"rank": r, "noun": noun_id2key[int(inv_noun[i])], "noun_class": int(inv_noun[i]),
                  "likelihood": p} for r, (i, p) in enumerate(zip(n_idx, n_prob), 1)]
    top5_action = []
    for r, (aid, p) in enumerate(zip(a_idx, a_prob), 1):
        ov, on = inv_action[aid]
        ov, on = int(ov), int(on)
        top5_action.append({"rank": r, "action": f"{verb_id2key[ov]} {noun_id2key[on]}",
                            "verb": verb_id2key[ov], "noun": noun_id2key[on],
                            "verb_class": ov, "noun_class": on, "action_class": int(aid),
                            "likelihood": p})
    return top5_verb, top5_noun, top5_action


def synth_clip(idx: int, res: int, n_frames: int) -> np.ndarray:
    """합성 [T,H,W,C] uint8 클립. 샘플마다 명도/텍스처를 달리해 forward 가 상수입력이 아니게 함."""
    rng = np.random.RandomState(1000 + idx)
    base = rng.randint(40, 200)
    clip = np.full((n_frames, res, res, 3), base, dtype=np.uint8)
    # 약한 시간적 그라디언트 + 노이즈 (softmax 가 완전 flat 이 되지 않도록)
    for t in range(n_frames):
        clip[t] = np.clip(clip[t].astype(np.int16) + (t - n_frames // 2) * 2
                          + rng.randint(-15, 15, (res, res, 3)), 0, 255).astype(np.uint8)
    return clip


def grid_jpeg(path: Path, idx: int, num_frames: int) -> None:
    """VLM 앵커 이미지(회색 grid). V-JEPA2 클립과 별개 — 정책이 보는 프레임."""
    from PIL import Image, ImageDraw
    side = 448
    img = Image.new("RGB", (side, side), (90, 90, 90))
    d = ImageDraw.Draw(img)
    if num_frames == 4:
        shades = [(70, 70, 70), (100, 100, 100), (130, 130, 130), (160, 160, 160)]
        for q, (x0, y0) in enumerate([(0, 0), (side // 2, 0), (0, side // 2), (side // 2, side // 2)]):
            d.rectangle([x0, y0, x0 + side // 2, y0 + side // 2], fill=shades[(q + idx) % 4])
            d.text((x0 + 8, y0 + 8), f"f{q + 1}", fill=(230, 230, 230))
    else:
        d.text((16, 16), f"frame {idx}", fill=(230, 230, 230))
    img.save(path, "JPEG", quality=85)


def assemble_record(i: int, img_path: Path, num_frames: int, t5a, t5n) -> dict:
    acts, acts_ws = [], []
    for a in t5a[:5]:
        acts.append({"verb": a["verb"], "noun": a["noun"], "action": a["action"], "score": a["likelihood"]})
        acts_ws.append({"verb": a["verb"], "noun": a["noun"], "likelihood": a["likelihood"], "rank": a["rank"]})
    nouns, nouns_ws = [], []
    for n in t5n[:5]:
        nouns.append({"noun": n["noun"], "score": n["likelihood"]})
        nouns_ws.append({"noun": n["noun"], "likelihood": n["likelihood"], "rank": n["rank"]})
    gt = acts[0]  # top-1 을 GT 로 (로깅 전용, 학습 미사용)
    if num_frames == 4:
        mem = ("Frame 1 (4.0s ago): take knife\nFrame 2 (2.67s ago): no completed action\n"
               "Frame 3 (1.33s ago): wash tomato\nFrame 4 (0.0s ago): no completed action")
        fmeta = {"n_frames": 4, "offsets_sec": FRAME_OFFSETS_4}
    else:
        mem = "Previously completed actions: take knife -> wash tomato."
        fmeta = {"n_frames": 1, "offsets_sec": [0.0]}
    return {
        "sample_id": f"vj{i:03d}", "frame_id": f"vj{i:03d}", "episode_id": f"V{i % 3:02d}",
        "image_path": str(img_path),
        "gt_verb": gt["verb"], "gt_noun": gt["noun"],
        "gt_label": {"verb": gt["verb"], "noun": gt["noun"]},
        "memory_context": mem, "frame_meta": fmeta,
        "topk_actions": acts, "topk_actions_with_score": acts_ws,
        "topk_nouns": nouns, "topk_nouns_with_score": nouns_ws,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V-JEPA2 EK100 AC → F0 jsonl (synthetic clips)")
    ap.add_argument("--vjepa_repo", required=True, help="cloned facebookresearch/vjepa2 경로")
    ap.add_argument("--backbone_ckpt", required=True, help="vitg-384.pt")
    ap.add_argument("--probe_ckpt", required=True, help="ek100-vitg-384.pt")
    ap.add_argument("--ann_dir", required=True, help="EK100 annotation CSV 디렉토리")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=4, choices=[1, 4])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    logging.getLogger().setLevel(logging.ERROR)  # filter_annotations 의 'file not found' 로그 억제
    sys.path.insert(0, str(Path(a.vjepa_repo).resolve()))

    from evals.action_anticipation_frozen.dataloader import make_transforms
    from evals.action_anticipation_frozen.epickitchens import filter_annotations
    from evals.action_anticipation_frozen.models import init_classifier, init_module

    ann = Path(a.ann_dir)
    train_csv = ann / "EPIC_100_train.csv"
    val_csv = ann / "EPIC_100_validation.csv"
    verb_csv = ann / "EPIC_100_verb_classes.csv"
    noun_csv = ann / "EPIC_100_noun_classes.csv"
    for p in (train_csv, val_csv, verb_csv, noun_csv):
        if not p.exists():
            print(f"✗ annotation 없음: {p}")
            sys.exit(2)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"[vjepa] device={device}")

    # 클래스 매핑 — 영상 없이도 full vocab (파일 존재검사 이전에 구성됨)
    anns = filter_annotations(base_path="__no_videos__", train_annotations_path=str(train_csv),
                              val_annotations_path=str(val_csv), file_format=0)
    verb_classes, noun_classes, action_classes = anns["verbs"], anns["nouns"], anns["actions"]
    inv_verb = {v: k for k, v in verb_classes.items()}
    inv_noun = {v: k for k, v in noun_classes.items()}
    inv_action = {v: k for k, v in action_classes.items()}
    verb_id2key = build_id_to_key(verb_csv)
    noun_id2key = build_id_to_key(noun_csv)
    print(f"[vjepa] classes: verbs={len(verb_classes)} nouns={len(noun_classes)} actions={len(action_classes)}")

    print(f"[vjepa] building backbone (ViT-g/384) — {a.backbone_ckpt}")
    model = init_module(module_name=MODULE_NAME, device=device,
                        frames_per_clip=FRAMES_PER_CLIP, frames_per_second=FRAMES_PER_SECOND,
                        resolution=RESOLUTION, checkpoint=a.backbone_ckpt,
                        model_kwargs=PRETRAIN_KWARGS, wrapper_kwargs=WRAPPER_KWARGS)
    clf = init_classifier(embed_dim=model.embed_dim, num_heads=CLF_NUM_HEADS,
                          num_blocks=CLF_NUM_BLOCKS, device=device, num_classifiers=1,
                          action_classes=action_classes, verb_classes=verb_classes,
                          noun_classes=noun_classes)[0]
    print(f"[vjepa] loading probe — {a.probe_ckpt}")
    ck = torch.load(a.probe_ckpt, map_location="cpu", weights_only=False)
    clf.load_state_dict(strip_module_prefix(ck["classifiers"][0]))
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    transform = make_transforms(training=False, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(3 / 4, 4 / 3),
                                random_resize_scale=(1.0, 1.0), reprob=0, auto_augment=False,
                                motion_shift=False, crop_size=RESOLUTION)

    frames_dir = Path(a.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with torch.no_grad():
        for i in range(a.n_samples):
            buf = synth_clip(i, RESOLUTION, FRAMES_PER_CLIP)   # [T,H,W,C] uint8
            clip = transform(buf).unsqueeze(0).to(device)      # [1,C,T,H,W]
            ant = torch.full((1,), ANTICIPATION_SEC, device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                feat = model(clip, ant)
                o = clf(feat)
            t5v, t5n, t5a = decode_topk(o, 0, inv_verb, inv_noun, inv_action, verb_id2key, noun_id2key)
            img = frames_dir / f"grid_{i:03d}.jpg"
            grid_jpeg(img, i, a.num_frames)
            records.append(assemble_record(i, img, a.num_frames, t5a, t5n))
            print(f"[vjepa] sample {i}: top1_action={t5a[0]['action']} "
                  f"lik={t5a[0]['likelihood']:.3f}")

    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[vjepa] wrote {len(records)} records → {out}")


if __name__ == "__main__":
    main()
