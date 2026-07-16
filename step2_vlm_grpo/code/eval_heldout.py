#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_heldout.py — P0: held-out(EPIC_100_validation) 평가 파이프라인.

체크포인트(LoRA adapter 또는 base)를 held-out JSONL 에서 평가:
  (a) GT 정확도            : verb / noun(exact·fuzzy) / action
  (b) G2 (WM-disagreement) : WM top-1 != GT & GT ∈ top-5 구간에서의 VLM 정답 선택률
                             (chance=0.20 [후보 5], 0.25 [WM top-1 제외 가정])
  (c) 후보 이탈률           : 선택 verb/noun 이 후보 목록 밖 (hallucination)
  (d) WM-follow rate       : WM top-1 을 그대로 고른 비율 (모방 vs 판단 진단)
  (e) WM top-1 참조선       : WM 자신의 top-1 GT 정확도 (G2 의 비교 대상)

프롬프트 조립·파싱은 train_qwen25vl_grpo_ek100.py 를 그대로 import — 학습과 평가의
입력 분포가 어긋나지 않게 보장. GT 는 이 스크립트(평가)에서만 사용된다.

사용:
  python eval_heldout.py --jsonl data/grpo_dataset/grpo_heldout.jsonl \
      --adapter runs/grpo_run1_wmonly/checkpoint-250 --limit 500 \
      --out runs/grpo_run1_wmonly/heldout_eval/step250.json
  (--adapter 생략 시 base 모델 = step-0 참조점)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
sys.path.insert(0, str(EGO_ROOT))

import train_qwen25vl_grpo_ek100 as T  # noqa: E402


def build_eval_rows(jsonl_path: str, limit: int | None, seed: int = 42):
    """train 과 동일한 build_dataset 경로로 프롬프트 생성 (min_wm_spread=0 — 평가는
    필터 없는 원분포에서). 반환: (converted rows list, raw rows list)."""
    rows = T.load_jsonl(jsonl_path)
    if limit:
        rows = rows[:limit]
    import random
    rng = random.Random(seed)
    converted, raw = [], []
    for ex in rows:
        if not Path(ex["image_path"]).exists():
            continue
        if not ex.get("topk_nouns") or not ex.get("topk_actions"):
            continue
        converted.append(T.make_conversation(ex, stage="gt", top_k=5, rng=rng,
                                             reward_mode="wm_likelihood"))
        raw.append(ex)
    return converted, raw


def to_multimodal_messages(prompt_msgs, image: Image.Image):
    """trl 학습 포맷(문자열 content)을 processor.apply_chat_template 용
    멀티모달 메시지로 변환 (user 턴 앞에 image 블록 주입 — trl 내부와 동일 위치)."""
    out = []
    for m in prompt_msgs:
        if m["role"] == "user":
            out.append({"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": m["content"]},
            ]})
        else:
            out.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    return out


@torch.no_grad()
def generate_batch(model, processor, batch_rows, max_new_tokens=256):
    texts, images = [], []
    for r in batch_rows:
        img = Image.open(r["image"]).convert("RGB")
        msgs = to_multimodal_messages(r["prompt"], img)
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        images.append(img)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=processor.tokenizer.pad_token_id)
    outs = []
    for i in range(len(batch_rows)):
        new_tokens = gen[i][inputs["input_ids"].shape[1]:]
        outs.append(processor.tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outs


def evaluate(preds, raws):
    """preds: 생성 텍스트 리스트, raws: 원 JSONL 행 리스트."""
    n = len(preds)
    m = {
        "n": n, "parsed": 0, "escape": 0,
        "gt_verb": 0, "gt_noun_exact": 0, "gt_noun_fuzzy": 0, "gt_action": 0, "gt_action_fuzzy": 0,
        "wm_follow": 0,
        "wm_top1_gt_action": 0,          # WM top-1 자체의 GT 정확도 (참조선)
        "gt_in_top5_action": 0,
        "g2_n": 0, "g2_correct": 0,      # WM top-1 오답 & GT ∈ top-5 구간
        "think_word_sum": 0, "think_nonempty": 0,
    }
    records = []
    for pred, raw in zip(preds, raws):
        verb, noun, think = T.parse_action_from_think_format(pred)
        gv, gn = raw.get("gt_verb", ""), raw.get("gt_noun", "")
        actions = (raw.get("topk_actions_with_score") or [])[:5]
        cand_verbs = {str(v) for v in (raw.get("topk_verbs") or [])[:5]}
        cand_nouns = {a["noun"] for a in (raw.get("topk_nouns_with_score") or [])[:5]}
        top1 = next((a for a in actions if a.get("rank") == 1), actions[0] if actions else {})

        wm1_gt = (top1.get("verb") == gv and (top1.get("noun") == gn or
                                              T._noun_fuzzy_match(top1.get("noun", ""), gn)))
        gt_in5 = any(a.get("verb") == gv and (a.get("noun") == gn or
                                              T._noun_fuzzy_match(a.get("noun", ""), gn))
                     for a in actions)
        m["wm_top1_gt_action"] += int(wm1_gt)
        m["gt_in_top5_action"] += int(gt_in5)
        is_g2 = (not wm1_gt) and gt_in5
        m["g2_n"] += int(is_g2)

        rec = {"sample_id": raw.get("frame_id", ""), "pred_verb": verb, "pred_noun": noun,
               "gt_verb": gv, "gt_noun": gn, "wm1_gt": wm1_gt, "gt_in_top5": gt_in5,
               "is_g2": is_g2, "think_words": len((think or "").split())}
        if think:
            m["think_nonempty"] += 1
            m["think_word_sum"] += rec["think_words"]
        if not verb or not noun:
            records.append(rec)
            continue
        m["parsed"] += 1
        if verb not in cand_verbs or noun not in cand_nouns:
            m["escape"] += 1
        vok = verb == gv
        nok_e = noun == gn
        nok_f = nok_e or T._noun_fuzzy_match(noun, gn)
        m["gt_verb"] += int(vok)
        m["gt_noun_exact"] += int(nok_e)
        m["gt_noun_fuzzy"] += int(nok_f)
        m["gt_action"] += int(vok and nok_e)
        m["gt_action_fuzzy"] += int(vok and nok_f)
        m["wm_follow"] += int(verb == top1.get("verb") and noun == top1.get("noun"))
        if is_g2:
            m["g2_correct"] += int(vok and nok_f)
        rec.update({"correct_action": vok and nok_f})
        records.append(rec)

    def rate(k, d):
        return round(m[k] / d, 4) if d else None

    summary = {
        "n": n,
        "parse_rate": rate("parsed", n),
        "candidate_escape_rate": rate("escape", m["parsed"]),
        "gt_verb_acc": rate("gt_verb", n),
        "gt_noun_acc_exact": rate("gt_noun_exact", n),
        "gt_noun_acc_fuzzy": rate("gt_noun_fuzzy", n),
        "gt_action_acc": rate("gt_action", n),
        "gt_action_acc_fuzzy": rate("gt_action_fuzzy", n),
        "wm_follow_rate": rate("wm_follow", m["parsed"]),
        "wm_top1_gt_action_acc": rate("wm_top1_gt_action", n),   # 참조선 (sample-level)
        "gt_in_top5_action_rate": rate("gt_in_top5_action", n),
        "g2_n": m["g2_n"],
        "g2_vlm_acc": rate("g2_correct", m["g2_n"]),
        "g2_chance": 0.20,
        "think_words_mean": round(m["think_word_sum"] / max(1, m["think_nonempty"]), 1),
        "think_rate": rate("think_nonempty", n),
    }
    return summary, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(EGO_ROOT / "data/grpo_dataset/grpo_heldout.jsonl"))
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir (미지정 = base 모델)")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default=None, help="요약 JSON 출력 경로 (.records.jsonl 도 함께 생성)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    T.PARSE_FORMAT = "think"

    rows, raws = build_eval_rows(args.jsonl, args.limit)
    print(f"[load] {len(rows)} held-out samples from {args.jsonl}")

    processor = AutoProcessor.from_pretrained(args.model_name, padding_side="left", use_fast=True,
                                              min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device})
    if args.adapter:
        from peft import PeftModel
        print(f"[load] LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    preds = []
    for i in tqdm(range(0, len(rows), args.batch_size), desc="generate"):
        preds.extend(generate_batch(model, processor, rows[i:i + args.batch_size],
                                    max_new_tokens=args.max_new_tokens))

    summary, records = evaluate(preds, raws)
    summary["adapter"] = args.adapter
    summary["jsonl"] = args.jsonl
    summary["time"] = datetime.now().isoformat(timespec="seconds")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        rec_path = out.with_suffix(".records.jsonl")
        with rec_path.open("w") as f:
            for pred, rec in zip(preds, records):
                rec["completion"] = pred[:1500]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[done] summary → {out}\n[done] records → {rec_path}")


if __name__ == "__main__":
    main()
