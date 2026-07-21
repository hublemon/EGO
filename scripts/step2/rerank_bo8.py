#!/usr/bin/env python3
"""rerank_bo8.py — best-of-8 자기 재순위화 실험.

**왜 이 실험인가**
  F0(FAA) 자기 롤아웃 8회의 pass@1 = 0.389, pass@8 = 0.614 (학습 1500샘플).
  그 +0.225 가 B0-P1 이 노리는 전부다. 그런데 P1(=P12) 은 유의한 효과를 못 냈다.
  두 가지 가능성이 있고 처방이 완전히 다르다:
    (a) 구현 문제 — 정답/오답 롤아웃이 모델 눈에는 구별되는데, DPO 쌍 설계가 그걸 못 썼다.
    (b) 근본 문제 — 구별 자체가 불가능하다(차이가 순전한 샘플링 잡음).
  판별법: **학습 없이** 모델 자신의 시퀀스 logprob 으로 8개를 재순위화한다.
    best-of-8 ≈ pass@8  → (a). 정보는 이미 모델 안에 있다. 선택만 고치면 된다.
    best-of-8 ≈ pass@1  → (b). 없는 순서를 DPO 로 만들어내야 한다.

GT 는 채점에만 쓰이고 점수 계산 경로에는 들어가지 않는다.
토큰 logprob·offset 정렬 규약은 remeasure_retro_margin.py 를 그대로 따른다.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

TAGS = {"reasoning": r"<reasoning>(.*?)</reasoning>",
        "task_belief": r"<task_belief>(.*?)</task_belief>",
        "action": r"<action>(.*?)</action>"}
TAG_RE = {k: re.compile(v, re.DOTALL) for k, v in TAGS.items()}
ACT_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)


def canon(v, n):
    return (str(v).strip().lower(), str(n).strip().lower())


def trace_action(s):
    m = ACT_RE.search(s or "")
    if not m:
        return None
    try:
        j = json.loads(m.group(1))
        return canon(j.get("verb"), j.get("noun"))
    except Exception:
        return None


def load_policy(model_name, adapter, device):
    m = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device})
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
        m = m.merge_and_unload()
    m.eval()
    return m


@torch.no_grad()
def token_logps(model, processor, prompt_msgs, image, completion):
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
    lps = [float(logp[t - 1, ids[t]]) for t in range(plen, ids.shape[0])]
    tok = processor.tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    offs = tok["offset_mapping"]
    n = min(len(lps), len(offs))
    return lps[:n], offs[:n]


def seq_stats(lps, offs, completion):
    ranges = {}
    for k, rgx in TAG_RE.items():
        m = rgx.search(completion)
        ranges[k] = (m.start(1), m.end(1)) if m else None
    out = {"sum": sum(lps), "ntok": len(lps), "span": {k: [0.0, 0] for k in TAGS}}
    for lp, (a, b) in zip(lps, offs):
        for k, rng in ranges.items():
            if rng and a >= rng[0] and b <= rng[1] + 1:
                out["span"][k][0] += lp
                out["span"][k][1] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="b0_samples_8gen.jsonl")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True, help="per-trace 점수 jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    dev = int(args.device.split(":")[-1]) if ":" in args.device else 0
    processor = AutoProcessor.from_pretrained(
        args.model_name, use_fast=True, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    print(f"[load] adapter={args.adapter or 'base'}", flush=True)
    model = load_policy(args.model_name, args.adapter, dev)

    t0 = time.time()
    done = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for i, r in enumerate(rows):
            ga = r["gt_action"]
            gt = canon(*(ga if isinstance(ga, (list, tuple)) else (ga["verb"], ga["noun"])))
            cands = [canon(a["verb"], a["noun"]) for a in r["candidates"]]
            rank = cands.index(gt) + 1 if gt in cands else 0
            p = r["image_path"]
            img = Image.open(p).convert("RGB") if p and Path(p).exists() else Image.new("RGB", (448, 448))
            recs = []
            for t in r["faa_traces"]:
                comp = t if isinstance(t, str) else (t.get("completion") or t.get("text") or "")
                st = seq_stats(*token_logps(model, processor, r["prompt"], img, comp), comp)
                a = trace_action(comp)
                spans = {}
                for k in TAGS:
                    s, c = st["span"][k]
                    spans[k] = (s / c) if c else None
                recs.append({
                    "action": list(a) if a else None,
                    "correct": bool(a == gt),
                    "sum": st["sum"], "ntok": st["ntok"],
                    "mean": st["sum"] / max(st["ntok"], 1),
                    "span_mean": spans,
                })
            fo.write(json.dumps({"sample_id": r["sample_id"], "gt": list(gt),
                                 "gt_rank": rank, "traces": recs}, ensure_ascii=False) + "\n")
            fo.flush()
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                eta = el / done * (len(rows) - done)
                print(f"[{done}/{len(rows)}] {el/60:.1f}m 경과 · ETA {eta/60:.1f}m", flush=True)
    print(f"[done] {done} samples · {(time.time()-t0)/60:.1f}m → {args.out}", flush=True)


if __name__ == "__main__":
    main()
