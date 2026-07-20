#!/usr/bin/env python3
"""remeasure_retro_margin.py — 코드리뷰 B0-4/B0-5/B0-6 반영 B0 margin 재측정.

기존 evaluate_b0 는 sum log-prob margin → 짧은 chosen 이 구조적으로 유리(길이 편향).
여기서는 동일 heldout DPO 쌍에 대해 정책(FAA/B0/A1)별로:
  1. sum       margin  : logπ(chosen) − logπ(rejected)            (기존 재현, sanity)
  2. mean-token margin : 토큰당 평균으로 정규화 (길이 편향 제거)
  3. span별   margin  : reasoning / task_belief / action 각 span 의 토큰당 평균 margin
     → 선호 이동이 'action 정렬'에서 왔는지 'teacher 문체(reasoning/belief)'에서 왔는지 분리.

margin_improvement(정책) = mean_pairs( margin_정책 − margin_FAA ).
GT 는 어떤 경로에도 쓰이지 않는다(로그확률만). 단일 tokenization 으로 logprob·offset 정렬.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = Path(__file__).resolve().parents[2]
TAGS = {"reasoning": r"<reasoning>(.*?)</reasoning>",
        "task_belief": r"<task_belief>(.*?)</task_belief>",
        "action": r"<action>(.*?)</action>"}
TAG_RE = {k: re.compile(v, re.DOTALL) for k, v in TAGS.items()}


def load_policy(model_name, adapter, device):
    m = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device})
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter); m = m.merge_and_unload()
    m.eval(); return m


@torch.no_grad()
def token_logps(model, processor, prompt_msgs, image, completion):
    """완성부 각 토큰의 logp 배열 + 완성 문자열 내 char-offset (span 귀속용)."""
    conv = [{"role": mm["role"], "content": (
        [{"type": "image", "image": image}, {"type": "text", "text": mm["content"]}]
        if mm["role"] == "user" else mm["content"])} for mm in prompt_msgs]
    ptext = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    full = ptext + completion
    pin = processor(text=[ptext], images=[image], return_tensors="pt").to(model.device)
    fin = processor(text=[full], images=[image], return_tensors="pt").to(model.device)
    plen = pin["input_ids"].shape[1]
    ids = fin["input_ids"][0]
    logits = model(**fin).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    comp_ids = ids[plen:]
    lps = [float(logp[t - 1, ids[t]]) for t in range(plen, ids.shape[0])]
    # 완성부는 순수 텍스트(이미지 토큰은 prompt 쪽) → 단독 재토큰화 offset 으로 span 귀속
    tok = processor.tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    offs = tok["offset_mapping"]
    n = min(len(lps), len(offs))          # 말미 EOS 등은 span 밖
    return lps[:n], offs[:n]


def span_char_ranges(completion):
    r = {}
    for k, rgx in TAG_RE.items():
        m = rgx.search(completion)
        r[k] = (m.start(1), m.end(1)) if m else None
    return r


def seq_stats(lps, offs, completion):
    """완성 1개 → {sum, mean, span별 (sum,count)}."""
    ranges = span_char_ranges(completion)
    out = {"sum": sum(lps), "ntok": len(lps),
           "span": {k: [0.0, 0] for k in TAGS}}
    for lp, (a, b) in zip(lps, offs):
        for k, rng in ranges.items():
            if rng and a >= rng[0] and b <= rng[1] + 1:
                out["span"][k][0] += lp; out["span"][k][1] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_jsonl", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--policies", required=True,
                    help="name:adapter,name:adapter,... (adapter 빈칸=base)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dpo_jsonl, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    dev = int(args.device.split(":")[-1]) if ":" in args.device else 0
    processor = AutoProcessor.from_pretrained(
        args.model_name, use_fast=True, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)

    specs = []
    for tok in args.policies.split(","):
        name, _, ad = tok.partition(":")
        specs.append((name, ad or None))

    ref_name = specs[0][0]
    per_policy = {}   # name -> list of per-pair dict {sum_m, mean_m, span_m:{k:val or None}, rel}
    for name, adapter in specs:
        print(f"[load] {name} ({adapter or 'base'})", flush=True)
        model = load_policy(args.model_name, adapter, dev)
        recs = []
        for i, r in enumerate(rows):
            img = Image.open(r["image_path"]).convert("RGB") if r.get("image_path") and Path(r["image_path"]).exists() else Image.new("RGB", (448, 448))
            cs = seq_stats(*token_logps(model, processor, r["prompt"], img, r["chosen"]), r["chosen"])
            rs = seq_stats(*token_logps(model, processor, r["prompt"], img, r["rejected"]), r["rejected"])
            sum_m = cs["sum"] - rs["sum"]
            mean_m = (cs["sum"] / max(cs["ntok"], 1)) - (rs["sum"] / max(rs["ntok"], 1))
            span_m = {}
            for k in TAGS:
                cse, cnt_c = cs["span"][k]; rse, cnt_r = rs["span"][k]
                span_m[k] = ((cse / cnt_c) - (rse / cnt_r)) if (cnt_c and cnt_r) else None
            meta = r.get("metadata", {})
            if isinstance(meta, str):
                try: meta = json.loads(meta.replace("'", '"'))
                except Exception: meta = {}
            rel = f"{meta.get('belief_relation','?')}/{meta.get('action_relation','?')}"
            recs.append({"sum_m": sum_m, "mean_m": mean_m, "span_m": span_m, "rel": rel})
            if (i + 1) % 100 == 0:
                print(f"  {name} {i+1}/{len(rows)}", flush=True)
        per_policy[name] = recs
        del model; torch.cuda.empty_cache()

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    ref = per_policy[ref_name]
    summary = {"n": len(rows), "ref": ref_name, "policies": {}}
    for name, recs in per_policy.items():
        block = {
            "sum_margin": avg([r["sum_m"] for r in recs]),
            "mean_token_margin": avg([r["mean_m"] for r in recs]),
            "pref_acc_sum": round(sum(1 for r in recs if r["sum_m"] > 0) / len(recs), 4),
            "pref_acc_mean": round(sum(1 for r in recs if r["mean_m"] > 0) / len(recs), 4),
            "span_margin": {k: avg([r["span_m"][k] for r in recs]) for k in TAGS},
        }
        if name != ref_name:
            block["improvement_vs_ref"] = {
                "sum": avg([r["sum_m"] - rr["sum_m"] for r, rr in zip(recs, ref)]),
                "mean_token": avg([r["mean_m"] - rr["mean_m"] for r, rr in zip(recs, ref)]),
                "span": {k: avg([(r["span_m"][k] - rr["span_m"][k])
                                 if (r["span_m"][k] is not None and rr["span_m"][k] is not None) else None
                                 for r, rr in zip(recs, ref)]) for k in TAGS},
            }
            # DIFFERENT/DIFFERENT 부분집합 (가장 깨끗한 신호)
            dd = [(r, rr) for r, rr in zip(recs, ref) if r["rel"] == "DIFFERENT/DIFFERENT"]
            if dd:
                block["improvement_DIFF_DIFF"] = {
                    "n": len(dd),
                    "mean_token": avg([r["mean_m"] - rr["mean_m"] for r, rr in dd]),
                    "action_span": avg([(r["span_m"]["action"] - rr["span_m"]["action"])
                                        if (r["span_m"]["action"] is not None and rr["span_m"]["action"] is not None) else None
                                        for r, rr in dd]),
                }
        summary["policies"][name] = block

    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
