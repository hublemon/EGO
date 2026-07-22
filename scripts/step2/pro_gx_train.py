#!/usr/bin/env python3
"""pro_gx_train.py — F0-GX 진단: action-only 5지선다 exact-gradient (listwise CE).

배경: action-only GRPO(F0-GA)는 completion 이 짧아 T1.3 에서도 8롤아웃이 전부 동일
(그룹 분산 0 → advantage 0 → 무학습). 행동 공간이 유한(후보 5)하고 reward 가 선택의
결정론적 함수이므로 샘플링 추정 대신 **정확한 기대보상 gradient** 를 쓴다:
  각 후보 a 의 completion "<action>{verb,noun}</action>" 를 teacher-forcing 스코어링
  → 후보 5개 sum-logp 의 softmax = π(a|x) → CE toward GT.
탐색·credit 배분·trace 생성이 전부 제거된 "가장 순수한 학습 가능성" 프로브.

진단 해석: F0-GX ↑ → 병목은 RL 최적화/생성 경로. F0-GX 평탄 → 모델/데이터 상한.
GT 를 쓰므로 방법이 아니라 진단 (F0-G 와 같은 지위). oracle-subset(GT∈top5) 학습.
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
from ego.common.run_provenance import write_run_config  # noqa: E402


def cand_completion(v: str, n: str) -> str:
    return f'<action>{{"verb": "{v}", "noun": "{n}"}}</action>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=7000, help="셔플 후 학습 샘플 수(에폭 환산)")
    ap.add_argument("--accum", type=int, default=16, help="샘플 단위 grad accumulation")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)
    # ── Exp-C: G1 selective-trust 앵커 ──
    ap.add_argument("--keep_weight", type=float, default=0.0,
                    help="G1(WM top-1==GT)에서 모델 후보분포를 WM 재정규화 분포로 당기는 KL 가중치. "
                         "0=순수 candidate CE(Exp-A). 참조는 F0가 아니라 WM prior 자신이다 — "
                         "F0의 G1 보존이 0.497이라 F0에 앵커하면 오히려 끌어내린다. WM은 G1에서 "
                         "정의상 top-1=GT라 완벽한 참조이고 점수가 데이터에 이미 있다.")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_run_config(out, vars(args), data_paths=[args.train_jsonl])
    T.ACTION_ONLY = True   # 프롬프트를 F0-GA/eval --action_only 와 동일하게

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_name, padding_side="right", use_fast=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    tok = processor.tokenizer
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
    # oracle-subset: GT ∈ joint top-5 (gt_only/gt_action_only 와 동일 정책)
    kept = []
    for ex in rows:
        c5 = [(str(a.get("verb", "")), str(a.get("noun", ""))) for a in (ex.get("topk_actions") or [])[:5]]
        if (ex.get("gt_verb"), ex.get("gt_noun")) in c5 and Path(ex["image_path"]).exists():
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
    log = open(out / "gx_log.jsonl", "a", encoding="utf-8")
    running, correct, seen = 0.0, 0, 0
    run_keep, keep_n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for i, ex in enumerate(order):
        conv = T.build_joint_conversation(ex, top_k=5, rng=prompt_rng)
        disp = json.loads(conv["topk_actions_display"])
        cands = [(str(d["verb"]), str(d["noun"])) for d in disp]
        try:
            gt_idx = cands.index((ex["gt_verb"], ex["gt_noun"]))
        except ValueError:
            continue
        img = Image.open(conv["image"]).convert("RGB")
        msgs = [{"role": "system", "content": [{"type": "text", "text": conv["prompt"][0]["content"]}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": conv["prompt"][1]["content"]}]}]
        base = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts = [base + cand_completion(v, n) for v, n in cands]
        enc = processor(text=texts, images=[[img]] * len(texts), return_tensors="pt",
                        padding=True).to(model.device)
        # 프롬프트 토큰 수: 같은 배치 내에서 base 부분은 공통 — base 단독 인코딩으로 길이 산출
        base_len = processor(text=[base], images=[[img]], return_tensors="pt")["input_ids"].shape[1]
        logits = model(**enc).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, : base_len - 1] = 0   # completion 토큰만
        tokl = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cand_lp = (tokl * mask).sum(dim=1)                       # (5,) sum-logp
        logq = torch.log_softmax(cand_lp, dim=0)                 # 모델 후보 log-분포
        ce = -logq[gt_idx]                                       # candidate CE (Exp-A 와 동일)
        total = ce                                               # keep 항은 아래서 더한다
        # ── Exp-C: G1 에서만 WM prior 로 당긴다 (selective trust) ──
        keep_val = 0.0
        if args.keep_weight > 0:
            acts = (ex.get("topk_actions_with_score") or [])[:5]
            wm1 = None
            liks = {}
            for a in acts:
                v, n = str(a.get("verb", "")), str(a.get("noun", ""))
                if a.get("rank") == 1:
                    wm1 = (v, n)
                if a.get("likelihood") is not None:
                    liks[(v, n)] = float(a["likelihood"])
            is_g1 = wm1 is not None and wm1 == (ex["gt_verb"], ex["gt_noun"])
            if is_g1 and len(liks) == len(cands):
                # disp(셔플됨) 순서에 맞춰 WM likelihood 재정규화 → 참조 분포 p_wm
                pw = torch.tensor([liks.get(c, 0.0) for c in cands],
                                  device=cand_lp.device, dtype=torch.float32)
                s = pw.sum()
                if s > 0:
                    pw = pw / s
                    # KL(p_wm ‖ p_θ) = Σ p_wm (log p_wm − log q) — 상수항 빼고 −Σ p_wm·log q.
                    # ★ loss 와 같은 그래프(logq)를 공유하므로 backward 는 합산 후 한 번만 호출한다.
                    keep = -(pw * logq).sum()
                    total = ce + args.keep_weight * keep
                    keep_val = float(keep)
        (total / args.accum).backward()
        running += float(ce)          # 로그의 loss 는 CE 만 — Exp-A 와 비교 가능하게
        run_keep += keep_val; keep_n += int(keep_val != 0.0)
        correct += int(int(cand_lp.argmax()) == gt_idx)
        seen += 1
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        if seen % args.log_every == 0:
            rec = {"seen": seen, "loss": round(running / args.log_every, 4),
                   "train_acc": round(correct / args.log_every, 4)}
            if args.keep_weight > 0:
                rec["keep_loss"] = round(run_keep / keep_n, 4) if keep_n else None
                rec["keep_frac"] = round(keep_n / args.log_every, 4)
            print(f"[gx] {rec}", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            run_keep, keep_n = 0.0, 0
            running, correct = 0.0, 0
        if seen % args.save_every == 0:
            model.save_pretrained(out / f"checkpoint-{seen}")
    model.save_pretrained(out / "checkpoint-final")
    processor.save_pretrained(out / "checkpoint-final")
    (out / "TRAINING_DONE").touch()
    print(f"[DONE] F0-GX adapter → {out}/checkpoint-final")


if __name__ == "__main__":
    main()
