#!/usr/bin/env python3
"""f0_gr_train.py — F0-GR 진단: action-only 생성 기반 REINFORCE + EMA 기준선.

설계 결정(2026-07-20): 후보 문자열 teacher-forcing 스코어링(exact-CE)은 "생성으로 행동을
선택하는 에이전트" 전제와 train-test 불일치라 기각. 행동 선택은 **생성**으로 유지하되,
GRPO 의 그룹-내 기준선(형제 롤아웃 평균 — 짧은 출력에선 전원 동일해 신호 소멸)을
**보상 이동평균(EMA) 기준선**으로 교체한다:

  프롬프트당 롤아웃 1개 생성(T 샘플링) → r = 1[action==GT] (무효 0)
  advantage = r − EMA(r)  →  loss = −advantage · mean_logp(생성 토큰)

그룹 다양성이 없어도 r ≠ EMA 면 gradient 가 흐른다. 결정론적 정책이어도 학습 가능.
진단 지위: GT 사용 → 방법 아님 (F0-G 와 동일). oracle-subset 학습.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ego.step2_vlm_alignment import train_grpo_action as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=7000)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--full_trace", action="store_true",
                    help="3태그 full-trace 출력 유지 (F0-WE 확정 run 용)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_gen", type=int, default=1,
                    help="생성 배치 크기 (autoregressive 생성이 지배 비용)")
    ap.add_argument("--ema", type=float, default=0.99, help="기준선 EMA 모멘텀")
    ap.add_argument("--reward", choices=["gt", "wm"], default="gt",
                    help="gt=바이너리(진단/WE) · wm=후보 정규화 likelihood (W-EMA, GT-free)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    T.ACTION_ONLY = not args.full_trace   # F0-WE: full-trace 유지

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_name, padding_side="left", use_fast=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": args.device})
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    rows = [json.loads(l) for l in open(args.train_jsonl, encoding="utf-8") if l.strip()]
    kept = []
    for ex in rows:
        if not Path(ex["image_path"]).exists():
            continue
        if args.reward == "wm":   # GT-free: GT 필터 금지 — 전 샘플 유지
            kept.append(ex); continue
        # gt 계열: oracle-subset (gt_only/gt_action_only 와 동일 정책)
        c5 = [(str(a.get("verb", "")), str(a.get("noun", ""))) for a in (ex.get("topk_actions") or [])[:5]]
        if (ex.get("gt_verb"), ex.get("gt_noun")) in c5:
            kept.append(ex)
    print(f"[data] oracle-subset {len(kept)}/{len(rows)} (coverage {len(kept)/len(rows):.3f})")
    (out / "oracle_manifest.json").write_text(json.dumps(
        {"num_total": len(rows), "num_kept": len(kept),
         "coverage": round(len(kept) / len(rows), 4), "policy": "drop"}))

    rng = random.Random(42)
    order = kept * ((args.max_samples // max(1, len(kept))) + 1)
    rng.shuffle(order)
    order = order[: args.max_samples]

    prompt_rng = random.Random(42)
    log = open(out / "gr_log.jsonl", "a", encoding="utf-8")
    baseline = 0.3   # 초기값 ≈ base 예상 acc — 수십 샘플 내 EMA 로 수렴
    run_loss = run_r = run_adv = 0.0
    seen = 0
    opt.zero_grad(set_to_none=True)
    def encode_one(ex):
        conv = T.build_joint_conversation(ex, top_k=5, rng=prompt_rng)
        img = Image.open(conv["image"]).convert("RGB")
        msgs = [{"role": "system", "content": [{"type": "text", "text": conv["prompt"][0]["content"]}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": conv["prompt"][1]["content"]}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text, img

    B = max(1, args.batch_gen)
    for bstart in range(0, len(order), B):
        batch = order[bstart:bstart + B]
        pairs = [encode_one(ex) for ex in batch]
        benc = processor(text=[t_ for t_, _ in pairs], images=[[im] for _, im in pairs],
                         return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            bgen = model.generate(**benc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                  temperature=args.temperature, top_p=0.95,
                                  pad_token_id=processor.tokenizer.pad_token_id)
        plen_pad = benc["input_ids"].shape[1]
        items = []
        for bi, ex in enumerate(batch):
            cids = bgen[bi][plen_pad:]
            items.append((ex, pairs[bi], cids))
        del benc, bgen
        i = bstart - 1
        for ex, (text, img), comp_ids in items:
         i += 1
         comp_text = processor.tokenizer.decode(comp_ids, skip_special_tokens=True)
         enc = processor(text=[text], images=[[img]], return_tensors="pt").to(model.device)
         v, n, _ = T.parse_action_from_think_format(comp_text)
         if args.reward == "gt":
             r = 1.0 if (v and n and v == ex["gt_verb"] and n == ex["gt_noun"]) else 0.0
         else:   # wm: 선택 후보의 재정규화 likelihood (wm_clean 과 동일 정의, GT 불사용)
             acts = (ex.get("topk_actions_with_score") or [])[:5]
             liks = [a.get("likelihood") for a in acts]
             mi = next((k for k, a in enumerate(acts)
                        if a.get("verb") == v and a.get("noun") == n), None)
             s = sum(float(x) for x in liks if x is not None)
             r = (float(liks[mi]) / s) if (mi is not None and liks[mi] is not None and s > 0) else 0.0
         adv = r - baseline
         baseline = args.ema * baseline + (1 - args.ema) * r

         keep = comp_ids != processor.tokenizer.pad_token_id
         comp_ids = comp_ids[keep]
         if comp_ids.numel() == 0:
             continue
         full = torch.cat([enc["input_ids"][0], comp_ids]).unsqueeze(0)
         attn = torch.ones_like(full)
         extra = {}
         if "mm_token_type_ids" in enc:   # Qwen3-VL M-RoPE: completion 토큰은 텍스트(0)
             mm = enc["mm_token_type_ids"][0]
             pad = torch.zeros(comp_ids.numel(), dtype=mm.dtype, device=mm.device)
             extra["mm_token_type_ids"] = torch.cat([mm, pad]).unsqueeze(0)
         out_logits = model(input_ids=full, attention_mask=attn,
                            pixel_values=enc.get("pixel_values"),
                            image_grid_thw=enc.get("image_grid_thw"), **extra).logits
         plen = enc["input_ids"].shape[1]
         lp = torch.log_softmax(out_logits[0, plen - 1:-1].float(), dim=-1)
         tok_lp = lp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
         loss = -(adv * tok_lp.mean()) / args.accum
         loss.backward()
         del out_logits, lp, enc

         run_loss += float(loss) * args.accum; run_r += r; run_adv += abs(adv); seen += 1
         if (i + 1) % args.accum == 0:
             torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
             opt.step(); opt.zero_grad(set_to_none=True)
         if seen % args.log_every == 0:
             rec = {"seen": seen, "loss": round(run_loss / args.log_every, 4),
                    "reward_ma": round(run_r / args.log_every, 4),
                    "baseline": round(baseline, 4),
                    "mean_abs_adv": round(run_adv / args.log_every, 4)}
             print(f"[gr] {rec}", flush=True)
             log.write(json.dumps(rec) + "\n"); log.flush()
             run_loss = run_r = run_adv = 0.0
         if seen % args.save_every == 0:
             model.save_pretrained(out / f"checkpoint-{seen}")
    model.save_pretrained(out / "checkpoint-final")
    processor.save_pretrained(out / "checkpoint-final")
    print(f"[DONE] F0-GR adapter → {out}/checkpoint-final")


if __name__ == "__main__":
    main()
