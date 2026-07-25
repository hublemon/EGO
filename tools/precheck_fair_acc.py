"""공정 재검 — vision-grounded + candidates-in-prompt + reason-then-score (+ WM fusion).

precheck_fusion.py는 vision-blind·no-reason이라 VLM에 불공정했다. 여기선:
  1. 프레임(8) + 후보 제시 프롬프트로 모델이 reasoning·belief를 **생성** (reason-then-answer)
  2. 그 프레임+생성 prefix 아래 Top-K 후보를 **채점**(length-norm logp) → VLM argmax
  3. WM score와 융합: s = ℓ_VLM(a) + α·log q_WM(a), α는 dev 보정 → heldout

배치-RoPE 버그 우회: 수동 input_ids 대신 processor에 (K개 full text + K개 이미지리스트)를
넘기고 **left padding** — vlm.generate_batch가 이미지와 함께 쓰는 검증된 경로.

비교: WM top-1(0.246 ref) vs VLM-alone vs VLM+WM fusion vs 배터리 free-gen.

사용: RETRO3_RUNS=runs/retro4 PYTHONPATH=src python3 tools/precheck_fair_acc.py \
        --config configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml [--adapter PATH] [--n_eval 500]
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

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]


@torch.no_grad()
def score_candidates_vision(model, processor, rec, imgs, reasoning, belief, device):
    """프레임 + (reasoning·belief) prefix 아래 Top-K 후보 length-norm logp.
    processor 배칭 + left-pad (수동 tiling 금지 — RoPE index 정합)."""
    tok = processor.tokenizer
    cands = rec["candidates"]
    prefix = ("<|im_start|>assistant\n<reasoning>\n" + reasoning.strip() + "\n</reasoning>\n\n"
              "<task_belief>\n" + belief.strip() + "\n</task_belief>\n\n<action>\n")
    base_text = processor.apply_chat_template(
        vlm.build_messages(rec, imgs), tokenize=False, add_generation_prompt=True) + prefix
    texts = [base_text + c for c in cands]
    clens = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in cands]

    old = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = processor(text=texts, images=[imgs] * len(cands), return_tensors="pt",
                        padding=True).to(device)
    finally:
        tok.padding_side = old
    logits = model(**enc).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    ids = enc["input_ids"][:, 1:]
    tok_lp = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # (K, L-1)
    scores = []
    for i, cl in enumerate(clens):
        scores.append(float(tok_lp[i, -cl:].mean()))       # 후보는 left-pad라 항상 맨 끝 cl토큰
    return scores


def acc_at_alpha(recs, alpha):
    n = c = 0
    for r in recs:
        gt = f"{r['gt_verb']} {r['gt_noun']}"
        K = r["candidates"]
        if gt not in K:
            continue
        n += 1
        lm = r["_lm"]
        qw = [math.log(max(1e-9, s)) for s in r["wm_scores"]]
        score = qw if alpha >= 1e8 else [lm[i] + alpha * qw[i] for i in range(len(K))]
        pred = K[max(range(len(K)), key=lambda i: score[i])]
        c += int(pred == gt)
    return c / max(1, n), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n_dev", type=int, default=300)
    ap.add_argument("--n_eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    device = "cuda"
    rows = read_jsonl(runs_root() / "data" / "context_val.jsonl")
    dev = [r for r in rows if r["split"] == "dev"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    rng = random.Random(args.seed)
    rng.shuffle(dev); rng.shuffle(heldout)
    dev, heldout = dev[: args.n_dev], heldout[: args.n_eval]

    model, processor = vlm.load_model()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter); model.eval()

    for split_name, recs in [("dev", dev), ("heldout", heldout)]:
        keep = []
        for i, r in enumerate(recs):
            try:
                imgs = vlm.extract_frames(video_root, r["video_uid"], r["obs_start_sec"], r["obs_end_sec"])
            except Exception:
                continue
            # 1) reason-then-answer: 프레임+후보로 reasoning·belief 생성
            gen = vlm.generate(model, processor, vlm.build_messages(r, imgs), max_new_tokens=320)
            parsed = vlm.parse_trace(gen)
            reasoning = parsed["reasoning"] if parsed else "Consider the visible scene and recent actions."
            belief = parsed["task_belief"] if parsed else "Continue the current procedure."
            # 2) 그 prefix 아래 후보 채점 (vision-grounded)
            r["_lm"] = score_candidates_vision(model, processor, r, imgs, reasoning, belief, device)
            keep.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{split_name}] {i+1}/{len(recs)}")
        recs[:] = keep
        vlm.close_readers()

    dev_curve = {a: acc_at_alpha(dev, a)[0] for a in ALPHAS}
    best_alpha = max([a for a in ALPHAS if a < 1e8], key=lambda a: dev_curve[a])
    ho = {a: acc_at_alpha(heldout, a) for a in ALPHAS}

    report = {
        "adapter": args.adapter or "base", "setup": "vision+candidates+reason-then-score",
        "n_heldout_covered": ho[0.0][1],
        "dev_acc_by_alpha": {str(a): round(dev_curve[a], 4) for a in ALPHAS},
        "best_alpha_on_dev": best_alpha,
        "heldout": {
            "VLM_alone": round(ho[0.0][0], 4),
            "WM_only_top1": round(ho[1e9][0], 4),
            "VLM_plus_WM_fusion": round(ho[best_alpha][0], 4),
        },
    }
    v, w, f = (report["heldout"][k] for k in ("VLM_alone", "WM_only_top1", "VLM_plus_WM_fusion"))
    if f > w + 0.01:
        report["verdict"] = f"fusion {f} > WM {w} — VLM(vision+reason)이 WM에 정보 추가, acc 트랙 부활 근거"
    elif v >= w - 0.01:
        report["verdict"] = f"VLM-alone {v} ≈ WM {w} — 공정하면 대등(불공정 0.123 아님). 융합 이득은 미미({f})"
    else:
        report["verdict"] = f"VLM {v} < WM {w}, fusion {f} — 공정해도 전문가 WM 못 넘음 (근거 있는 음성)"
    out = runs_root() / "eval" / f"precheck_fair_{report['adapter'].replace('/', '_')[:40]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n[precheck] written: {out}")
    vlm.close_readers()


if __name__ == "__main__":
    main()
