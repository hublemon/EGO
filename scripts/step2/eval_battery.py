#!/usr/bin/env python3
"""eval_battery.py — 사전 검증 배터리 (handoff §10.1 ①②⑤) 실행기.

v2 프롬프트 빌더(train_grpo_action.build_joint_conversation)를 **그대로** 사용해
학습과 평가의 입력 분포를 일치시킨다 (evaluate.py 는 v1 플랫 스크립트를 import 하여
4f grid 안내·L2-c 정렬을 모른다 — 배터리 ⑤는 이 러너로만 유효).

배터리 매핑:
  ⑤ 4f-base   : --jsonl <4f-strict heldout>                    (게이트: acc > 0.30)
  ⑤ 1f-base   : --jsonl <1f-strict heldout>                    (동일 샘플 대조)
  ① no-memory : --no_memory                                     (히스토리 기여도)
  ② history-only: --history_only  (L2-a 마스킹 경로 전샘플 적용 — 프레임 없이 5지선다)

지표: acc(정확 일치)·verb/noun acc·in_joint5·wm_follow·G2·parse·belief/reasoning 통계.
GT 는 평가에만 사용 (리워드 경로와 무관).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ego.step2_vlm_alignment import train_grpo_action as T  # noqa: E402


def to_multimodal_messages(prompt_msgs, n_images: int = 1):
    """trl 학습 포맷(문자열 content) → processor 용 멀티모달 메시지 (user 턴 앞 image 블록)."""
    out = []
    for m in prompt_msgs:
        if m["role"] == "user":
            out.append({"role": "user", "content": [
                *([{"type": "image"}] * n_images), {"type": "text", "text": m["content"]}]})
        else:
            out.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    return out


@torch.no_grad()
def generate_batch(model, processor, convs, max_new_tokens, multi_image_dir=None):
    texts, images = [], []
    for c in convs:
        if multi_image_dir:
            # 재게이트: 합성 grid 대신 개별 4프레임 — 각 이미지가 독립 픽셀 예산을 받는다
            imgs = [Image.open(Path(multi_image_dir) / f'{c["sample_id"]}_f{i}.jpg').convert("RGB")
                    for i in range(4)]
        else:
            imgs = [Image.open(c["image"]).convert("RGB")]
        msgs = to_multimodal_messages(c["prompt"], n_images=len(imgs))
        texts.append(processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
        images.append(imgs)
    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True).to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=processor.tokenizer.pad_token_id)
    outs = []
    for i in range(len(convs)):
        outs.append(processor.tokenizer.decode(gen[i][inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True))
    return outs


_ACT_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_V_RE = re.compile(r'"verb"\s*:\s*"([^"]*)"')
_N_RE = re.compile(r'"noun"\s*:\s*"([^"]*)"')


def parse_pred(text: str):
    """<action>{"verb","noun"}</action> 추출 (json 실패 시 regex 폴백)."""
    m = _ACT_RE.search(text or "")
    if not m:
        return None, None
    blob = m.group(1).strip()
    try:
        d = json.loads(blob)
        return (d.get("verb") or "").strip(), (d.get("noun") or "").strip()
    except Exception:
        mv, mn = _V_RE.search(blob), _N_RE.search(blob)
        return (mv.group(1).strip() if mv else None,
                mn.group(1).strip() if mn else None)


def load_convs(jsonl: str, limit: int | None = None, seed: int = 42):
    """heldout jsonl → (convs, raws). 프롬프트 빌더·필터·seed 규약의 단일 출처.

    ⚠ rng 는 행 순서대로 소비되므로 `limit` 을 바꾸면 앞쪽 샘플의 프롬프트는 그대로지만
      슬라이싱 지점이 달라진다 — subset 비교는 **전량 1회 생성 후 records 를 분할**해야
      동일 프롬프트 위에서 비교된다 (eval_harness_v2 가 그렇게 한다).
    """
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    rng = random.Random(seed)  # evaluate.py 와 동일 seed — 셔플 재현
    convs, raws = [], []
    for ex in rows:
        if not Path(ex["image_path"]).exists() or not ex.get("topk_actions"):
            continue
        convs.append(T.build_joint_conversation(ex, top_k=5, rng=rng))
        raws.append(ex)
    return convs, raws


def score_predictions(preds, convs, raws):
    """생성 텍스트 → (지표 카운터 m, per-sample records). **지표 정의의 단일 출처.**

    eval_harness_v2 가 subset·bootstrap 재집계에 그대로 재사용한다 — 여기서만 정의를 바꾼다.
    """
    n = len(preds)
    m = dict(n=n, parsed=0, acc=0, verb_acc=0, noun_acc=0, in_joint5=0, wm_follow=0,
             wm_top1_gt=0, gt_in_top5=0, g2_n=0, g2_correct=0, acc_in5=0,
             belief_present=0, belief_restate=0, reasoning_words=0)
    records = []
    for pred_text, conv, ex in zip(preds, convs, raws):
        v, nn_ = parse_pred(pred_text)
        gt_v, gt_n = ex["gt_verb"], ex["gt_noun"]
        top5 = [(a["verb"], a["noun"]) for a in ex["topk_actions"][:5]]
        wm1 = top5[0]
        disp = [(d["verb"], d["noun"]) for d in json.loads(conv["topk_actions_display"])]
        gt_in5 = (gt_v, gt_n) in top5
        m["wm_top1_gt"] += int(wm1 == (gt_v, gt_n))
        m["gt_in_top5"] += int(gt_in5)
        g2 = gt_in5 and wm1 != (gt_v, gt_n)
        m["g2_n"] += int(g2)
        ok = v is not None and nn_ is not None
        m["parsed"] += int(ok)
        correct = ok and (v, nn_) == (gt_v, gt_n)
        m["acc"] += int(correct)
        m["verb_acc"] += int(ok and v == gt_v)
        m["noun_acc"] += int(ok and nn_ == gt_n)
        m["in_joint5"] += int(ok and (v, nn_) in disp)
        m["wm_follow"] += int(ok and (v, nn_) == wm1)
        m["g2_correct"] += int(g2 and correct)
        m["acc_in5"] += int(gt_in5 and correct)      # conditional acc 분자 (GT∈top5)
        mb = re.search(r"<task_belief>(.*?)</task_belief>", pred_text, re.DOTALL)
        belief = (mb.group(1).strip() if mb else "")
        m["belief_present"] += int(bool(belief))
        m["belief_restate"] += int(bool(belief) and ok and v in belief and nn_ in belief)
        mr = re.search(r"<reasoning>(.*?)</reasoning>", pred_text, re.DOTALL)
        m["reasoning_words"] += len((mr.group(1) if mr else "").split())
        records.append({"sample_id": conv["sample_id"], "pred_verb": v, "pred_noun": nn_,
                        "gt_verb": gt_v, "gt_noun": gt_n, "correct": correct,
                        "g2": g2, "wm_follow": ok and (v, nn_) == wm1,
                        "n_frames": conv["n_frames"], "completion": pred_text})
    return m, records


def summarize_metrics(m):
    """카운터 → 비율 지표. 실행 메타(jsonl/adapter/time 등)는 호출자가 덧붙인다."""
    n = m["n"]

    def rate(k, d=None):
        den = m[d] if d else n
        return round(m[k] / den, 4) if den else None

    return {
        "n": n,
        "parse_rate": rate("parsed"), "acc": rate("acc"),
        "verb_acc": rate("verb_acc"), "noun_acc": rate("noun_acc"),
        "in_joint5": rate("in_joint5"), "wm_follow": rate("wm_follow"),
        "wm_top1_gt_acc": rate("wm_top1_gt"), "gt_in_top5_rate": rate("gt_in_top5"),
        "g2_n": m["g2_n"], "g2_acc": rate("g2_correct", "g2_n"), "g2_chance": 0.2,
        # oracle-subset 해석용 3분리 (통합 핸드오프 2.4): coverage / conditional / overall
        "acc_given_gt_in_top5": rate("acc_in5", "gt_in_top5") if m["gt_in_top5"] else None,
        "oracle_upper_bound_proxy": (round(m["gt_in_top5"] / n * (m["acc_in5"] / m["gt_in_top5"]), 4)
                                     if m["gt_in_top5"] else None),
        "rank1_given_in5": rate("wm_follow", "wm_top1_gt") if m["wm_top1_gt"] else None,
        "belief_present_rate": rate("belief_present"),
        "belief_restatement_rate": rate("belief_restate", "belief_present")
        if m["belief_present"] else None,
        "mean_reasoning_words": round(m["reasoning_words"] / n, 1) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=384)  # 학습 completion 예산과 동일
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_memory", action="store_true", help="배터리 ①: memory_context 공란화")
    ap.add_argument("--action_only", action="store_true",
                    help="F0-GA 진단: action-only 프롬프트 (T.ACTION_ONLY)")
    ap.add_argument("--history_only", action="store_true",
                    help="배터리 ②: 전 샘플 프레임 마스킹 (L2-a 경로) — 히스토리 단독 예측력")
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28,
                    help="재게이트: 픽셀 예산 상향 시 조정 (기본 = v1 평가·학습과 동일 602k)")
    ap.add_argument("--multi_image_dir", default=None,
                    help="재게이트: {sample_id}_f{0..3}.jpg 개별 4프레임 입력 (합성 grid 대체)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.multi_image_dir:
        # 프레임 설명을 grid → 개별 4이미지로 교체 (build 시점에 모듈 전역 참조)
        T.JOINT_FRAME_DESC_4 = (
            "1. Four first-person frames sampled over the last 4 seconds, given as four\n"
            "   separate images in order (4.0s ago, 2.7s ago, 1.3s ago, now).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    T.NO_MEMORY = args.no_memory
    if args.action_only:
        T.ACTION_ONLY = True
    if args.history_only:
        T.MASK_FRAME_PROB = 1.0
        T.BLANK_IMAGE_PATH = T._prepare_blank_image(str(out.parent))

    convs, raws = load_convs(args.jsonl, args.limit)
    print(f"[load] {len(convs)} samples  no_memory={args.no_memory} "
          f"history_only={args.history_only}")
    if convs:
        print(f"[check] n_frames={convs[0]['n_frames']} frame_masked={convs[0]['frame_masked']}")

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

    preds = []
    for i in tqdm(range(0, len(convs), args.batch_size), desc="generate"):
        preds.extend(generate_batch(model, processor, convs[i:i + args.batch_size],
                                    args.max_new_tokens,
                                    multi_image_dir=args.multi_image_dir))

    m, records = score_predictions(preds, convs, raws)
    summary = {
        "n": m["n"], "jsonl": args.jsonl, "model": args.model_name, "adapter": args.adapter,
        "no_memory": args.no_memory, "history_only": args.history_only,
        "n_frames": convs[0]["n_frames"] if convs else None,
        **summarize_metrics(m),
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "multi_image": bool(args.multi_image_dir),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with out.with_suffix(".records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()
