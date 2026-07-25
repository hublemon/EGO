"""공정 재검 (빠른판) — vision-grounded 후보 채점 + WM fusion.

느린 부분(reasoning 재생성)을 제거: 배터리 records에 **이미 저장된** VLM 자신의
reasoning·task_belief를 재사용한다(= battery가 평가한 그 trace 그대로, 일관됨).
각 샘플에서 프레임 + (저장된 reasoning·belief) prefix 아래 Top-K 후보를 채점(vision-grounded).

  s(a) = ℓ_VLM(a | frames, reasoning, belief) + α·log q_WM(a)   argmax over Dt
  α는 dev(covered) 보정 → heldout. 참조: WM top-1(α=∞), VLM-alone(α=0).

핵심 질문: G1 유지율이 낮아 VLM 단독이 WM에 지는데, WM score를 섞으면(확신할 때 WM
믿기) WM top-1을 넘나? → 넘으면 논문 acc 스토리 부활.

사용: RETRO3_RUNS=runs/retro4 PYTHONPATH=src python3 tools/precheck_fair_fusion.py \
        --records runs/retro4/eval/base.records.jsonl --config configs/.../hist_k8.yaml [--adapter PATH]
"""
from __future__ import annotations

import argparse
import zlib
import json
import math
import random
import sys
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.runtime import read_jsonl, runs_root

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]


@torch.no_grad()
def score_candidates_vision(model, processor, rec, imgs, reasoning, belief, device):
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
        enc = processor(text=texts, images=[imgs] * len(cands), return_tensors="pt", padding=True).to(device)
    finally:
        tok.padding_side = old
    logits = model(**enc).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    ids = enc["input_ids"][:, 1:]
    tok_lp = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return [float(tok_lp[i, -cl:].mean()) for i, cl in enumerate(clens)]


def acc_at_alpha(rows, alpha):
    n = c = 0
    for r in rows:
        n += 1
        lm, qw, K, gt = r["_lm"], r["_qw"], r["_cands"], r["_gt"]
        score = qw if alpha >= 1e8 else [lm[i] + alpha * qw[i] for i in range(len(K))]
        pred = K[max(range(len(K)), key=lambda i: score[i])]
        c += int(pred == gt)
    return c / max(1, n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2_retrospection/goalstep_end_m1_hist_k8.yaml")
    ap.add_argument("--records", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video_root = Path(cfg["shared_assets"]["video_root"])
    device = "cuda"
    # support: candidates + wm_scores (per-candidate), context: split
    support = {r["sample_id"]: r for r in read_jsonl(runs_root() / "data" / "support_val.jsonl")}
    ctx = {r["sample_id"]: r for r in read_jsonl(runs_root() / "data" / "context_val.jsonl")}
    recs = [r for r in read_jsonl(Path(args.records))
            if not r.get("malformed") and r.get("reasoning") and r.get("task_belief")
            and r["sample_id"] in support]
    rng = random.Random(args.seed); rng.shuffle(recs)
    recs = recs[: args.n]

    model, processor = vlm.load_model()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter); model.eval()

    dev, heldout = [], []
    for i, r in enumerate(recs):
        sid = r["sample_id"]
        s = support[sid]
        gt = f"{s['gt_verb']} {s['gt_noun']}"
        if gt not in s["candidates"]:
            continue
        try:
            imgs = vlm.extract_frames(video_root, s["video_uid"], s["obs_start_sec"], s["obs_end_sec"])
        except Exception:
            continue
        lm = score_candidates_vision(model, processor, s, imgs, r["reasoning"], r["task_belief"], device)
        row = {"_lm": lm, "_qw": [math.log(max(1e-9, x)) for x in s["wm_scores"]],
               "_cands": s["candidates"], "_gt": gt}
        # records가 전부 heldout이므로 α 보정용 calib을 sample_id 해시로 30% 분리 (결정적).
        is_calib = (zlib.crc32(sid.encode()) % 10) < 3
        (dev if is_calib else heldout).append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}  (dev {len(dev)}, heldout {len(heldout)})", flush=True)
        if (i + 1) % 50 == 0:
            vlm.close_readers()

    dev_curve = {a: acc_at_alpha(dev, a)[0] for a in ALPHAS} if dev else {a: 0 for a in ALPHAS}
    best_alpha = max([a for a in ALPHAS if a < 1e8], key=lambda a: dev_curve[a]) if dev else 1.0
    ho = {a: acc_at_alpha(heldout, a) for a in ALPHAS}
    v, w, f = ho[0.0][0], ho[1e9][0], ho[best_alpha][0]
    report = {
        "adapter": args.adapter or "base", "records": args.records,
        "setup": "vision + candidates + stored-reasoning + WM fusion",
        "n_heldout": ho[0.0][1], "n_dev": len(dev),
        "dev_acc_by_alpha": {str(a): round(dev_curve[a], 4) for a in ALPHAS},
        "best_alpha_on_dev": best_alpha,
        "heldout": {"VLM_alone": round(v, 4), "WM_top1": round(w, 4),
                    "VLM_plus_WM_fusion": round(f, 4),
                    "fusion_gain_over_WM": round(f - w, 4)},
    }
    report["verdict"] = (
        f"fusion {f} > WM {w} (+{f-w:.3f}) — VLM이 WM에 정보 추가, acc 스토리 부활" if f > w + 0.01
        else f"fusion {f} ≈ WM {w} — 융합해도 WM 못 넘음, acc 기여 없음 (GADR·faithfulness로)")
    out = runs_root() / "eval" / f"fair_fusion_{report['adapter'].replace('/', '_')[:36]}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    print(f"[written] {out}", flush=True)
    vlm.close_readers()


if __name__ == "__main__":
    sys.exit(main())
