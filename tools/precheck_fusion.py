"""Precheck (F): 학습 0 score-fusion baseline — S2/DPO 학습이 필요한지 판정.

F(a|x) = ℓ_LM(a | x, candidate-free) + α·log q_WM(a|x),  argmax over WM top-K → pred.
  - ℓ_LM: candidate-free 프롬프트(후보 목록 없음) 아래 action a의 length-norm logp
  - q_WM: WM softmax score (support의 wm_scores)
  - α: dev split에서 acc|cov 최대로 보정 → heldout에 적용 (eval로 α 선택 금지, review §5.3)

참조점:
  α=0   → 순수 candidate-free LM (융합 없음)
  α=∞   → 순수 WM top-1 (= L0)
  F-base → base LM + 융합.  (--adapter로 F-C 측정 가능)

판정: F-base acc|cov 가 학습 모델(retro3 base 0.223 / r1_sft 0.234)에 필적하면
  → 학습-free 융합으로 충분, S2/DPO 트랙 재고 (Case E).

사용: RETRO3_RUNS=runs/retro4 PYTHONPATH=src python3 tools/precheck_fusion.py \
        --config configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml [--adapter PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import read_jsonl, runs_root

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]  # 1e9 ≈ 순수 WM(top-1)

CANDFREE_USER = (
    "Completed actions so far (oldest to newest):\n{hist}\n\n"
    "Predict the single next action the person does. Respond in the required "
    "format; the <action> line must be a 'verb noun' pair."
)


def messages_text(rec: dict) -> list[dict]:
    """candidate-free 텍스트 전용 메시지 (후보 목록 없음, 이미지 없이 스코어링)."""
    user = CANDFREE_USER.format(hist=vlm.fmt_history(rec))
    return [{"role": "system", "content": [{"type": "text", "text": vlm.SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]


@torch.no_grad()
def action_logps(model, processor, rec, actions, device):
    """candidate-free 프롬프트 아래 action별 length-norm logp (raw, softmax 아님)."""
    tok = processor.tokenizer
    prefix = "<|im_start|>assistant\n<action>\n"
    text = processor.apply_chat_template(messages_text(rec),
                                         tokenize=False, add_generation_prompt=True) + prefix
    base = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    aid = [tok(a, add_special_tokens=False)["input_ids"] for a in actions]
    maxc = max(len(c) for c in aid)
    B, L = len(aid), len(base) + maxc
    input_ids = base.new_full((B, L), tok.eos_token_id)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, cids in enumerate(aid):
        seq = torch.cat([base, torch.tensor(cids, device=device)])
        input_ids[i, :len(seq)] = seq
        attn[i, :len(seq)] = 1
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return [float(tok_lp[i, len(base) - 1: len(base) - 1 + len(c)].mean()) for i, c in enumerate(aid)]


def acc_at_alpha(recs, alpha):
    """covered(GT∈K) 샘플에서 F argmax == GT 비율."""
    n = c = 0
    for r in recs:
        gt = f"{r['gt_verb']} {r['gt_noun']}"
        K = r["candidates"]
        if gt not in K:
            continue
        n += 1
        lm = r["_lm"]                       # length-norm logp per K action
        qw = [math.log(max(1e-9, s)) for s in r["wm_scores"]]
        if alpha >= 1e8:                    # 순수 WM
            score = qw
        else:
            score = [lm[i] + alpha * qw[i] for i in range(len(K))]
        pred = K[max(range(len(K)), key=lambda i: score[i])]
        c += int(pred == gt)
    return c / max(1, n), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda"
    rows = read_jsonl(runs_root() / "data" / "support_val.jsonl")
    dev = [r for r in rows if r["split"] == "dev"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    rng = random.Random(args.seed)
    rng.shuffle(dev); rng.shuffle(heldout)
    dev, heldout = dev[: args.n_dev], heldout[: args.n_eval]

    model, processor = vlm.load_model()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    for split_name, recs in [("dev", dev), ("heldout", heldout)]:
        for i, r in enumerate(recs):
            r["_lm"] = action_logps(model, processor, r, r["candidates"], device)
            if (i + 1) % 100 == 0:
                print(f"  [{split_name}] scored {i+1}/{len(recs)}")

    # dev에서 α 보정
    dev_curve = {a: acc_at_alpha(dev, a)[0] for a in ALPHAS}
    finite = [a for a in ALPHAS if a < 1e8]
    best_alpha = max(finite, key=lambda a: dev_curve[a])
    ho = {a: acc_at_alpha(heldout, a) for a in ALPHAS}

    report = {
        "adapter": args.adapter or "base", "n_dev_covered": acc_at_alpha(dev, 0.0)[1],
        "n_heldout_covered": ho[0.0][1],
        "dev_acc_by_alpha": {str(a): round(dev_curve[a], 4) for a in ALPHAS},
        "best_alpha_on_dev": best_alpha,
        "heldout": {
            "LM_only_alpha0": round(ho[0.0][0], 4),
            "WM_only_top1": round(ho[1e9][0], 4),
            "F_fusion_bestalpha": round(ho[best_alpha][0], 4),
        },
        "reference_trained_acc_cov": {"retro3_base_presented": 0.223, "retro3_r1_sft_presented": 0.234},
    }
    f = report["heldout"]["F_fusion_bestalpha"]
    lm = report["heldout"]["LM_only_alpha0"]
    wm = report["heldout"]["WM_only_top1"]
    gain = f - max(lm, wm)
    if f >= 0.234 and gain > 0.01:
        report["verdict_hint"] = (f"F-fusion acc|cov {f} ≥ 학습모델(0.234) 이며 LM/WM 단독({lm}/{wm}) 초과 "
                                  f"→ 학습-free 융합으로 충분, S2/DPO 재고 (Case E)")
    elif gain <= 0.005:
        report["verdict_hint"] = (f"F-fusion {f} ≈ max(LM {lm}, WM {wm}) — 융합 이득 미미, "
                                  f"WM/LM 한쪽이 지배. S2 여지 판단은 학습 실측 필요")
    else:
        report["verdict_hint"] = (f"F-fusion {f} > 단독 최대 {max(lm,wm)} (이득 +{gain:.3f}) 이나 "
                                  f"학습모델 0.234 미달 — 학습이 추가 이득 줄 여지 有")
    out = runs_root() / "eval" / f"precheck_fusion_{report['adapter'].replace('/', '_')[:40]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n[precheck] written: {out}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
