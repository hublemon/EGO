#!/usr/bin/env python3
"""eval_candidate_scored.py — 생성 없이 후보 스코어링으로 heldout 을 평가한다.

기존 평가(eval_battery / eval_harness_v2)는 **생성 전용**이라, 후보 CE 로 학습한 정책의
실력을 그대로 보여주지 못한다. 학습 목적함수와 평가 경로가 어긋나면 개선이 있어도
디코딩에서 사라진다 — 지금까지 정책들의 조건부 정확도(≈0.39)가 base 모델의 후보 스코어링
능력(0.388, teacher_headroom 실측)과 같은 자리에 머문 것이 그 징후다.

이 스크립트는 후보 5개를 teacher forcing 으로 스코어링해 argmax 를 예측으로 삼는다.
출력 records 는 `decompose_g1g2.py` 와 같은 스키마라 G1/G2 분해에 바로 넣을 수 있다.

★ 항상 `L0`(WM top-1 무학습)를 함께 낸다. 이 베이스라인을 표에서 빼면 "학습의 순효과가
   음수"라는 사실이 드러나지 않는다 — 실제로 이 프로젝트에서 3일간 그랬다.

    python scripts/step2/eval_candidate_scored.py --jsonl <heldout> --adapter <ckpt> --out x.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "step2"))
sys.path.insert(0, str(REPO / "src"))
from pro_gr_train import score_candidates  # noqa: E402
from ego.step2_vlm_alignment import train_grpo_action as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0=전량")
    ap.add_argument("--action_only", action="store_true",
                    help="학습(pro_gx)과 동일한 프롬프트 레짐. 학습·평가가 반드시 일치해야 한다")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max_pixels", type=int, default=602112)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.action_only:
        T.ACTION_ONLY = True

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_name, padding_side="left", use_fast=True,
        min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    rng = random.Random(42)          # 프롬프트 셔플 재현 — 다른 평가와 동일 규약
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rec_f = out_p.with_suffix(".records.jsonl").open("w", encoding="utf-8")

    n = g1 = g2 = out_n = 0
    g1_keep = g2_fix = g2_other = correct = wm_follow = l0_hit = 0
    t0 = time.time()
    total = len(rows)
    for i, ex in enumerate(rows):
        if not Path(ex["image_path"]).exists() or not ex.get("topk_actions"):
            continue
        conv = T.build_joint_conversation(ex, top_k=5, rng=rng)
        cands = [(str(d["verb"]), str(d["noun"]))
                 for d in json.loads(conv["topk_actions_display"])]
        gt = (ex["gt_verb"], ex["gt_noun"])
        top5 = [(a["verb"], a["noun"]) for a in ex["topk_actions"][:5]]
        wm1 = top5[0]
        img = Image.open(conv["image"]).convert("RGB")
        msgs = [{"role": "system", "content": [{"type": "text", "text": conv["prompt"][0]["content"]}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": conv["prompt"][1]["content"]}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            sc = score_candidates(model, processor, enc, cands)
        pred = cands[int(sc.argmax())]
        del enc

        n += 1
        ok = pred == gt
        correct += ok
        wm_follow += (pred == wm1)
        l0_hit += (wm1 == gt)                       # L0 = WM top-1 그냥 따르기
        if gt not in top5:
            out_n += 1
        elif wm1 == gt:
            g1 += 1; g1_keep += ok
        else:
            g2 += 1; g2_fix += ok
            if not ok and pred != wm1:
                g2_other += 1
        # WM likelihood 를 disp(셔플) 순서에 정렬 — 융합 분석은 이 두 벡터만 있으면
        # 재추론 없이 α 스윕이 산술로 끝난다
        lik_map = {(str(a.get("verb")), str(a.get("noun"))): a.get("likelihood")
                   for a in (ex.get("topk_actions_with_score") or [])[:5]}
        rec_f.write(json.dumps({"sample_id": conv["sample_id"],
                                "pred_verb": pred[0], "pred_noun": pred[1],
                                "gt_verb": gt[0], "gt_noun": gt[1], "correct": bool(ok),
                                "g2": bool(gt in top5 and wm1 != gt),
                                "wm_follow": bool(pred == wm1),
                                "margin": round(float(sc.max() - sc.sort().values[-2]), 3),
                                "cands": [[v, n] for v, n in cands],
                                "scores": [round(float(x), 4) for x in sc],
                                "wm_lik": [lik_map.get(c) for c in cands],
                                "wm1_idx": (cands.index(wm1) if wm1 in cands else None),
                                "gt_idx": (cands.index(gt) if gt in cands else None)},
                               ensure_ascii=False) + "\n")
        rec_f.flush()
        if n % 100 == 0:
            r = n / (time.time() - t0)
            print(f"[{n}/{total}] acc {correct/n:.4f} · L0 {l0_hit/n:.4f} · "
                  f"남은 {int((total-i-1)/max(r,1e-9)/60)}분", flush=True)
    rec_f.close()

    summ = {
        "jsonl": args.jsonl, "adapter": args.adapter, "action_only": bool(args.action_only),
        "readout": "candidate_scored(sum-logp, argmax)", "n": n,
        "acc": round(correct / n, 4),
        "L0_wm_top1": round(l0_hit / n, 4),           # ★ 항상 병기
        "beats_L0": bool(correct > l0_hit),
        "g1_n": g1, "g1_retention": round(g1_keep / g1, 4) if g1 else None,
        "g2_n": g2, "g2_correction": round(g2_fix / g2, 4) if g2 else None,
        "g2_non_gt_switch": round(g2_other / g2, 4) if g2 else None,
        "out_n": out_n, "R5": round((g1 + g2) / n, 4),
        "conditional_acc": round(correct / (g1 + g2), 4) if (g1 + g2) else None,
        "wm_follow": round(wm_follow / n, 4),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summ, indent=2, ensure_ascii=False), flush=True)
    out_p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {out_p}", flush=True)


if __name__ == "__main__":
    main()
