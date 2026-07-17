"""evaluate_b0.py — FAA vs B0 held-out 비교 (핸드오프 §16~21, GPU).

세 검증을 동일 past-only 프롬프트·generation config 로 수행:
  A. held-out preference margin  : m = logπ(chosen) - logπ(rejected). m_B0 > m_FAA 기대.
                                   relation 별(DIFFERENT/SAME/SAME-SAME) 분해 보고.
  B. GT action accuracy          : candidate_recall / conditional / end-to-end 분리(§18).
                                   recovery/regression, 62% 목표 근접.
  C. full-trace coherence(§20)   : 순수 프록시(future leak / belief 재진술 / 반복) — API 0.

이 파일은 GPU 에서 실행. 순수 계산 함수(compute_margin_stats, accuracy_split,
coherence_proxies)는 dependency-free 라 smoke 로 검사한다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

from .trace_utils import (canonical_action, has_future_leak_language, parse_full_trace)


# ─────────────────────────── 순수 계산 (smoke 대상) ───────────────────────────

def compute_margin_stats(margins: list[float]) -> dict:
    if not margins:
        return {"n": 0, "mean_margin": None, "preference_accuracy": None}
    n = len(margins)
    return {
        "n": n,
        "mean_margin": round(sum(margins) / n, 4),
        "preference_accuracy": round(sum(1 for m in margins if m > 0) / n, 4),
    }


def accuracy_split(preds: list[dict]) -> dict:
    """preds: [{pred_verb,pred_noun, gt_verb,gt_noun, gt_in_cand(bool)}].
    §18 분해: candidate_recall / conditional / end-to-end 를 분리 보고."""
    n = len(preds) or 1
    recall = sum(1 for p in preds if p["gt_in_cand"]) / n
    e2e = sum(1 for p in preds if _correct(p)) / n
    in_cand = [p for p in preds if p["gt_in_cand"]]
    cond = (sum(1 for p in in_cand if _correct(p)) / len(in_cand)) if in_cand else None
    return {
        "candidate_recall": round(recall, 4),
        "conditional_accuracy": round(cond, 4) if cond is not None else None,
        "end_to_end_accuracy": round(e2e, 4),
    }


def _correct(p: dict) -> bool:
    return canonical_action(p["pred_verb"], p["pred_noun"]) == \
        canonical_action(p["gt_verb"], p["gt_noun"])


def recovery_regression(faa_preds: list[dict], b0_preds: list[dict]) -> dict:
    """§18 recovery/regression. 두 리스트는 같은 순서·같은 sample."""
    rec = reg = ret = uns = 0
    for f, b in zip(faa_preds, b0_preds):
        fc, bc = _correct(f), _correct(b)
        if not fc and bc:
            rec += 1
        elif fc and not bc:
            reg += 1
        elif fc and bc:
            ret += 1
        else:
            uns += 1
    n = len(faa_preds) or 1
    return {
        "recovery": rec, "regression": reg, "retained": ret, "unresolved": uns,
        "net_recovery": round((rec - reg) / n, 4),
    }


def coherence_proxies(completion: str) -> dict:
    """§20 coherence 프록시 (API 0). 판정 아님 — 스크리닝 지표."""
    tr = parse_full_trace(completion)
    belief_restate = False
    if tr.belief and tr.verb and tr.noun:
        b = tr.belief.lower()
        belief_restate = tr.verb.lower() in b and tr.noun.lower().split(":")[0] in b
    words = tr.reasoning.split()
    # 반복 template: 같은 5-gram 이 2회 이상
    grams = [" ".join(words[i:i+5]) for i in range(max(0, len(words) - 4))]
    repeated = len(grams) != len(set(grams)) if grams else False
    return {
        "parsed": tr.is_complete(),
        "future_leak": has_future_leak_language(tr.reasoning),
        "belief_restatement": belief_restate,
        "repeated_template": repeated,
        "reasoning_words": len(words),
    }


def aggregate_coherence(completions: list[str]) -> dict:
    ps = [coherence_proxies(c) for c in completions]
    n = len(ps) or 1
    return {
        "parse_rate": round(sum(p["parsed"] for p in ps) / n, 4),
        "future_leak_rate": round(sum(p["future_leak"] for p in ps) / n, 4),
        "belief_restatement_rate": round(sum(p["belief_restatement"] for p in ps) / n, 4),
        "repeated_template_rate": round(sum(p["repeated_template"] for p in ps) / n, 4),
        "mean_reasoning_words": round(sum(p["reasoning_words"] for p in ps) / n, 1),
    }


# ─────────────────────────── GPU 실행부 ───────────────────────────

def _seq_logprob(model, processor, prompt_msgs, image, completion: str) -> float:
    """log π(completion | prompt) — teacher forcing 합. GPU."""
    import torch
    from PIL import Image as _I  # noqa
    conv = [{"role": m["role"], "content": (
        [{"type": "image", "image": image}, {"type": "text", "text": m["content"]}]
        if m["role"] == "user" else m["content"])} for m in prompt_msgs]
    ptext = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    full = ptext + completion
    pin = processor(text=[ptext], images=[image], return_tensors="pt").to(model.device)
    fin = processor(text=[full], images=[image], return_tensors="pt").to(model.device)
    plen = pin["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(**fin).logits
    ids = fin["input_ids"][0]
    logp = torch.log_softmax(logits[0], dim=-1)
    tot = 0.0
    for t in range(plen, ids.shape[0]):
        tot += float(logp[t - 1, ids[t]])
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_jsonl", required=True, help="held-out DPO pairs (preference margin용)")
    ap.add_argument("--faa_adapter", required=True)
    ap.add_argument("--b0_adapter", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True, padding_side="left")

    def load(adapter):
        base = AutoModelForImageTextToText.from_pretrained(
            args.model_name, dtype=torch.bfloat16, device_map="auto")
        m = PeftModel.from_pretrained(base, adapter)
        m.eval()
        return m

    faa, b0 = load(args.faa_adapter), load(args.b0_adapter)
    rows = [json.loads(l) for l in Path(args.dpo_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    faa_m, b0_m, by_rel = [], [], {}
    for r in rows:
        img = Image.open(r["image_path"]).convert("RGB")
        fm = (_seq_logprob(faa, processor, r["prompt"], img, r["chosen"])
              - _seq_logprob(faa, processor, r["prompt"], img, r["rejected"]))
        bm = (_seq_logprob(b0, processor, r["prompt"], img, r["chosen"])
              - _seq_logprob(b0, processor, r["prompt"], img, r["rejected"]))
        faa_m.append(fm)
        b0_m.append(bm)
        rel = f'{r["metadata"].get("belief_relation")}/{r["metadata"].get("action_relation")}'
        by_rel.setdefault(rel, {"faa": [], "b0": []})
        by_rel[rel]["faa"].append(fm)
        by_rel[rel]["b0"].append(bm)

    result = {
        "A_preference": {
            "faa": compute_margin_stats(faa_m),
            "b0": compute_margin_stats(b0_m),
            "margin_improvement": round((sum(b0_m) - sum(faa_m)) / (len(b0_m) or 1), 4),
            "by_relation": {k: {"faa": compute_margin_stats(v["faa"]),
                                "b0": compute_margin_stats(v["b0"])} for k, v in by_rel.items()},
        },
        "note": "B: GT action accuracy / C: coherence 는 generation eval 로 별도 산출 — "
                "accuracy_split / recovery_regression / aggregate_coherence 사용.",
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["A_preference"], ensure_ascii=False, indent=2))
    print(f"[done] → {args.out}")


if __name__ == "__main__":
    main()
