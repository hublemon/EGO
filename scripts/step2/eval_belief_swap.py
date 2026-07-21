#!/usr/bin/env python3
"""eval_belief_swap.py — 배터리 ③: belief-swap 개입 테스트 (handoff §10.1-③).

질문: <task_belief>가 <action>을 실제로 조향하는가?
방법: 기존 평가 records의 완성 trace에서 reasoning은 고정한 채
  <task_belief>만 다른 샘플의 belief(derangement)로 교체하고, 그 지점부터
  <action> 만 이어서 재생성(greedy). 대조군은 원본 belief 로 동일 재생성.

지표:
  control_action_change : 대조군에서 원 action 과 달라진 비율 (디코딩 노이즈 플로어)
  swap_action_change    : swap 군에서 원 action 과 달라진 비율
  causal_sensitivity    : swap − control (B0 성공 판정의 baseline — 0이면 belief 는 인과적으로 무력)

B0 스펙 §1/§22 의 사전값. GT 는 어떤 경로에도 사용되지 않는다.
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

TAG_RE = {
    "reasoning": re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL),
    "belief": re.compile(r"<task_belief>(.*?)</task_belief>", re.DOTALL),
    "action": re.compile(r"<action>(.*?)</action>", re.DOTALL),
}
_V_RE = re.compile(r'"verb"\s*:\s*"([^"]*)"')
_N_RE = re.compile(r'"noun"\s*:\s*"([^"]*)"')


def parse_action(blob: str):
    try:
        d = json.loads(blob.strip())
        return (d.get("verb") or "").strip(), (d.get("noun") or "").strip()
    except Exception:
        mv, mn = _V_RE.search(blob), _N_RE.search(blob)
        return (mv.group(1).strip() if mv else None, mn.group(1).strip() if mn else None)


@torch.no_grad()
def continue_actions(model, processor, items, batch_size, max_new_tokens=48):
    """items: [{conv, prefix}] — prompt + assistant prefix 뒤에서 <action> 만 이어 생성."""
    outs = []
    for i in tqdm(range(0, len(items), batch_size), desc="continue"):
        chunk = items[i:i + batch_size]
        texts, images = [], []
        for it in chunk:
            c = it["conv"]
            img = Image.open(c["image"]).convert("RGB")
            msgs = []
            for m in c["prompt"]:
                if m["role"] == "user":
                    msgs.append({"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": m["content"]}]})
                else:
                    msgs.append({"role": m["role"],
                                 "content": [{"type": "text", "text": m["content"]}]})
            base = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(base + it["prefix"])   # assistant 응답을 prefix 로 강제
            images.append([img])
        inputs = processor(text=texts, images=images, return_tensors="pt",
                           padding=True).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
        for j in range(len(chunk)):
            outs.append(processor.tokenizer.decode(gen[j][inputs["input_ids"].shape[1]:],
                                                   skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="평가 heldout jsonl (records 와 동일 순서)")
    ap.add_argument("--records", required=True, help="trace 원천: eval_battery *.records.jsonl")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=500,
                    help="0=heldout 전량 (개선 1: ③ 도 전량으로 재고. 기본값 500 은 기존 호환)")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--swap_offset", type=int, default=250, help="derangement 간격")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # `--limit 0` 은 '전량'(eval_battery 규약). 이전 코드는 [:0] 이라 빈 실행이 됐다.
    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    recs = {r["sample_id"]: r for r in map(json.loads, open(args.records, encoding="utf-8"))}
    rng = random.Random(42)  # eval_battery 와 동일 — 동일 프롬프트(셔플 포함) 재현
    convs = []
    for ex in rows:
        if not Path(ex["image_path"]).exists() or not ex.get("topk_actions"):
            continue
        convs.append(T.build_joint_conversation(ex, top_k=5, rng=rng))

    # trace 파싱 + belief derangement 구성
    usable = []
    for c in convs:
        r = recs.get(c["sample_id"])
        if not r:
            continue
        comp = r["completion"]
        mt, mb, ma = (TAG_RE["reasoning"].search(comp), TAG_RE["belief"].search(comp),
                      TAG_RE["action"].search(comp))
        if not (mt and mb and ma):
            continue
        usable.append({"conv": c, "reasoning": mt.group(1).strip(),
                       "belief": mb.group(1).strip(),
                       "orig_action": parse_action(ma.group(1))})
    n = len(usable)
    for i, u in enumerate(usable):
        cand = usable[(i + args.swap_offset) % n]["belief"]
        if cand.strip().lower() == u["belief"].strip().lower():
            cand = usable[(i + args.swap_offset + 1) % n]["belief"]
        u["swap_belief"] = cand
    print(f"[load] usable traces: {n}")

    processor = AutoProcessor.from_pretrained(
        args.model_name, padding_side="left", use_fast=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    def prefix_of(u, belief):
        return (f"<reasoning>{u['reasoning']}</reasoning>\n"
                f"<task_belief>{belief}</task_belief>\n<action>")

    results = {}
    for cond in ("control", "swap"):
        items = [{"conv": u["conv"],
                  "prefix": prefix_of(u, u["belief"] if cond == "control" else u["swap_belief"])}
                 for u in usable]
        gens = continue_actions(model, processor, items, args.batch_size)
        changed = valid = 0
        recs_out = []
        for u, g in zip(usable, gens):
            v, nn_ = parse_action(g.split("</action>")[0])
            ok = v is not None and nn_ is not None
            valid += int(ok)
            ch = ok and (v, nn_) != u["orig_action"]
            changed += int(ch)
            recs_out.append({"sample_id": u["conv"]["sample_id"], "cond": cond,
                             "orig": list(u["orig_action"]), "new": [v, nn_], "changed": bool(ch)})
        results[cond] = {"n": n, "valid": valid, "action_change": round(changed / n, 4),
                         "records": recs_out}
        print(f"[{cond}] action_change = {changed}/{n} = {changed/n:.4f} (valid {valid})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": n, "adapter": args.adapter, "records_src": args.records,
        "control_action_change": results["control"]["action_change"],
        "swap_action_change": results["swap"]["action_change"],
        "causal_sensitivity": round(results["swap"]["action_change"]
                                    - results["control"]["action_change"], 4),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with out.with_suffix(".records.jsonl").open("w", encoding="utf-8") as f:
        for cond in ("control", "swap"):
            for r in results[cond]["records"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()
