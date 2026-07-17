# NOTE: F0 (WM-only GRPO) 트랙의 **실제로 검증된 코드**를 그대로 옮긴 것이다.
# 2026-07-17 F0 final 결과(docs/experiments/2026-07-17_f0_final.md)를 만든 코드가 이 파일이다.
# 패키지 구조에 맞춘 추측성 리팩터는 하지 않았다 — 재검증 없이 쪼개면 결과 재현이 깨진다.
# 실행은 configs/step2/f0_final_wm_only.yaml 과 scripts/step2/train_f0_final.sh 참조.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_reasoning_trace.py — G3 (reasoning 인과성) 사후 검증 모듈. 학습 루프 밖, 보상 미사용.

입력: eval_heldout.py 가 만든 *.records.jsonl (sample_id + completion)
      + held-out JSONL (이미지·후보 재구성용)
판정 모델: 동결 base VLM (학습 전 Qwen2.5-VL) — 학습된 정책이 아니라 고정 참조점.

계층 1 — coherence lift (정량):
    lift = log p(선택 | 이미지, 후보, think) − log p(선택 | 이미지, 후보, think 없음)
    think 가 장식이면 lift ≈ 0, 판단을 실제로 좁혔으면 lift > 0.
    체크포인트별 records 에 반복 적용 → 학습 step 에 따른 G3 곡선.

계층 2 — 반사실 테스트 (규칙 기반, 판정 모델 불필요):
    --mode shuffle : 다른 샘플의 think 와 짝지어 lift 재측정. grounded reasoning 이면 붕괴해야 정상.
                     붕괴하지 않으면 think 가 범용 템플릿이라는 증거.
    --mode mask    : think 말미의 명시적 결론 선언 문장을 제거하고 lift 재측정.
                     마스킹 후 lift 가 0 으로 붕괴하면 '답안 예고편'(P3 hacking 검출기).

계층 3 — 외부 judge (정성):
    --mode judge   : 다른 모델 계열(Gemini, letsur 게이트웨이)로 루브릭 5항목 채점 (각 0-2).
                     같은 Qwen 계열 판정은 self-preference bias + P3 독립성 훼손으로 금지.
                     LETSUR_API_KEY 환경변수 필요. 학습에는 절대 미사용.

사용:
  python eval_reasoning_trace.py --records runs/X/heldout_eval/step250.records.jsonl \
      --jsonl data/grpo_dataset/grpo_heldout.jsonl --mode lift --out runs/X/trace_eval/step250_lift.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
sys.path.insert(0, str(EGO_ROOT))

import train_qwen25vl_grpo_ek100 as T  # noqa: E402
from eval_heldout import build_eval_rows, to_multimodal_messages  # noqa: E402

# 결론 선언 문장 패턴 (계층 2 mask): think 말미의 "따라서 X를 선택" 류.
CONCLUSION_PAT = re.compile(
    r"(?:therefore|so,|thus|hence|i (?:will|should|choose|select|pick)|"
    r"the (?:best|most likely|next) action is|my (?:choice|answer|selection))",
    re.IGNORECASE)


# ---------------- 공통: assistant 텍스트의 action 부분 조건부 로그확률 ----------------

@torch.no_grad()
def action_logprob(model, processor, prompt_msgs, image, think: str | None, action_json: str) -> float:
    """log p(<action>action_json</action> | 이미지, 후보 프롬프트, [think]).
    think=None 이면 think 블록 없이 측정 (lift 의 베이스라인)."""
    prefix = f"<think>\n{think}\n</think>\n" if think is not None else ""
    target = f"<action>{action_json}</action>"
    msgs = to_multimodal_messages(prompt_msgs, image)
    chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    full = chat + prefix + target
    ids_full = processor(text=[full], images=[image], return_tensors="pt")
    ids_ctx = processor(text=[chat + prefix], images=[image], return_tensors="pt")
    n_ctx = ids_ctx["input_ids"].shape[1]

    ids_full = {k: v.to(model.device) for k, v in ids_full.items()}
    out = model(**ids_full)
    logits = out.logits[0, :-1]                       # t 시점 logits → t+1 토큰 예측
    labels = ids_full["input_ids"][0, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    # target 토큰 구간만 합산 (컨텍스트 마지막 토큰이 target 첫 토큰을 예측하는 지점부터)
    return float(tok_lp[n_ctx - 1:].sum().item())


def split_completion(completion: str):
    """completion → (think 텍스트 | None, action JSON 문자열 | None)."""
    think = T.extract_think_block(completion)
    m = re.search(r"<action>(.*?)</action>", completion, re.DOTALL)
    action = m.group(1).strip() if m else None
    return think, action


def mask_conclusion(think: str) -> str:
    """think 말미의 결론 선언 문장 제거. 문장 단위로 뒤에서부터 검사해
    결론 패턴이 있는 마지막 연속 구간을 잘라낸다."""
    sents = re.split(r"(?<=[.!?])\s+", think.strip())
    while sents and CONCLUSION_PAT.search(sents[-1]):
        sents.pop()
    return " ".join(sents)


# ---------------- 계층 3: 외부 judge ----------------

JUDGE_RUBRIC_PROMPT = """You are evaluating the reasoning trace of an embodied action-selection model.
The model saw an egocentric kitchen frame and had to pick ONE next action by combining
a verb from {verbs} and a noun from {nouns}. Task goal: "{goal}".

Model reasoning:
---
{think}
---
Model's final choice: {action}

Score each criterion 0 (absent), 1 (partial), or 2 (clear), then output JSON only:
1. candidate_review  — explicitly evaluates named candidates from the lists (not generic talk)
2. visual_grounding  — cites observable evidence from the frame/context for its judgments
3. no_hallucination  — mentions no verbs/nouns/objects outside the candidate lists (2 = none)
4. conclusion_follows — the final choice follows logically from the reasoning content
5. goal_awareness    — the task goal meaningfully informs the judgment

Output: {{"candidate_review": N, "visual_grounding": N, "no_hallucination": N,
"conclusion_follows": N, "goal_awareness": N, "note": "<one short sentence>"}}"""


def judge_one(client, model_name, rec, raw):
    prompt = JUDGE_RUBRIC_PROMPT.format(
        verbs=json.dumps((raw.get("topk_verbs") or [])[:5]),
        nouns=json.dumps([n["noun"] for n in (raw.get("topk_nouns_with_score") or [])[:5]]),
        goal=raw.get("task_goal", ""),
        think=(T.extract_think_block(rec["completion"]) or "")[:2000],
        action=f'{rec.get("pred_verb")} {rec.get("pred_noun")}')
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        # Gemini 계열은 응답 전에 reasoning 토큰을 소모(실측: 한 단어 응답에 133토큰)
        # → 예산이 작으면 content가 None으로 옴. 루브릭 JSON 여유분 포함 1500.
        max_completion_tokens=1500, temperature=0)
    txt = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group(0)) if m else {"error": txt[:200]}


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="eval_heldout.py 의 *.records.jsonl")
    ap.add_argument("--jsonl", default=str(EGO_ROOT / "data/grpo_dataset/grpo_heldout.jsonl"))
    ap.add_argument("--mode", choices=["lift", "shuffle", "mask", "judge"], default="lift")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="계층 1/2 판정용 동결 base 모델 (학습된 adapter 를 지정하지 말 것)")
    ap.add_argument("--judge_model", default="gemini-2.5-flash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    T.PARSE_FORMAT = "think"
    recs = [json.loads(l) for l in open(args.records) if l.strip()]
    rows, raws = build_eval_rows(args.jsonl, None)
    by_id = {r.get("frame_id", ""): (c, r) for c, r in zip(rows, raws)}

    # completion 에 think+action 이 모두 있는 레코드만
    usable = []
    for rec in recs:
        think, action = split_completion(rec.get("completion", ""))
        if think and action and rec.get("sample_id") in by_id:
            usable.append((rec, think, action))
    random.Random(args.seed).shuffle(usable)
    usable = usable[: args.limit]
    print(f"[load] usable records: {len(usable)} (mode={args.mode})")

    results = []

    if args.mode == "judge":
        import openai
        client = openai.OpenAI(base_url="https://gw.letsur.ai/v1",
                               api_key=os.environ["LETSUR_API_KEY"])
        for rec, think, action in tqdm(usable, desc="judge"):
            _, raw = by_id[rec["sample_id"]]
            try:
                scores = judge_one(client, args.judge_model, rec, raw)
            except Exception as e:
                scores = {"error": f"{type(e).__name__}: {e}"}
            results.append({"sample_id": rec["sample_id"], **scores})
        keys = ["candidate_review", "visual_grounding", "no_hallucination",
                "conclusion_follows", "goal_awareness"]
        ok = [r for r in results if "error" not in r]
        summary = {f"judge_{k}_mean": round(sum(r[k] for r in ok) / max(1, len(ok)), 3) for k in keys}
        summary.update({"n": len(ok), "n_error": len(results) - len(ok), "judge_model": args.judge_model})
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": args.device})
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True,
                                                  min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
        # shuffle 모드: think 를 한 칸 밀어 다른 샘플과 짝지음
        shifted = usable[1:] + usable[:1]
        for i, (rec, think, action) in enumerate(tqdm(usable, desc=args.mode)):
            conv, raw = by_id[rec["sample_id"]]
            img = Image.open(conv["image"]).convert("RGB")
            if args.mode == "shuffle":
                think_used = shifted[i][1]          # 엉뚱한 맥락의 think
            elif args.mode == "mask":
                think_used = mask_conclusion(think)
                if not think_used.strip():          # 전부 결론 문장이면 lift 정의상 0 처리
                    results.append({"sample_id": rec["sample_id"], "lift": None,
                                    "note": "think 가 결론 선언만으로 구성"})
                    continue
            else:
                think_used = think
            try:
                lp_with = action_logprob(model, processor, conv["prompt"], img, think_used, action)
                lp_wo = action_logprob(model, processor, conv["prompt"], img, None, action)
                results.append({"sample_id": rec["sample_id"],
                                "lift": round(lp_with - lp_wo, 4),
                                "logp_with_think": round(lp_with, 4),
                                "logp_no_think": round(lp_wo, 4)})
            except Exception as e:
                results.append({"sample_id": rec["sample_id"], "lift": None,
                                "note": f"{type(e).__name__}: {e}"})
        lifts = [r["lift"] for r in results if r.get("lift") is not None]
        summary = {"mode": args.mode, "n": len(lifts),
                   "lift_mean": round(sum(lifts) / max(1, len(lifts)), 4),
                   "lift_pos_rate": round(sum(1 for x in lifts if x > 0) / max(1, len(lifts)), 4)}

    summary["records"] = args.records
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "results": results},
                                  indent=2, ensure_ascii=False))
        print(f"[done] → {out}")


if __name__ == "__main__":
    main()
