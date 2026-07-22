#!/usr/bin/env python3
"""decompose_g1g2.py — 정확도를 G1 보존 / G2 교정으로 분해한다 (GPU 불필요).

배경: 40개 arm 전수 조사에서 이 프로젝트의 정확도 차이는 사실상 **G1 보존율**로 설명된다
(corr +0.930, 변동폭 0.262). G2 교정률은 corr +0.825 에 변동폭 0.154 로 기여가 작다.
전체 acc 하나만 보면 이 구분이 보이지 않아, 어느 쪽이 움직였는지 모른 채 몇 일을 쓴다.

정의 (heldout 실측: P(G1)=0.399 · P(G2)=0.227 · P(OUT)=0.374 · 상한 0.626):
  G1  : WM top-1 == GT            → 정책이 지켜야 하는 구간
  G2  : GT ∈ top-5 이고 WM1 != GT → 정책이 고쳐야 하는 구간
  OUT : GT ∉ top-5                → candidate-constrained 로는 구조적으로 불가

  Acc ≈ P(G1)·(G1 보존율) + P(G2)·(G2 교정률)

`L0`(무학습 WM top-1 추종 = 0.399) 행을 항상 함께 낸다. 이 베이스라인을 표에 넣지 않아서
"학습된 어떤 모델도 WM top-1 을 넘지 못했다"는 사실이 오래 드러나지 않았다.

    python scripts/step2/decompose_g1g2.py --jsonl <heldout.jsonl> --records a.jsonl b.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reference(jsonl: str) -> dict:
    ref = {}
    for line in open(jsonl, encoding="utf-8"):
        if not line.strip():
            continue
        e = json.loads(line)
        top5 = [(a["verb"], a["noun"]) for a in (e.get("topk_actions") or [])[:5]]
        if not top5:
            continue
        ref[e["frame_id"]] = {"gt": (e["gt_verb"], e["gt_noun"]),
                              "wm1": top5[0], "top5": top5}
    return ref


def decompose(records_path: str, ref: dict) -> dict | None:
    recs = [json.loads(l) for l in open(records_path, encoding="utf-8") if l.strip()]
    if not recs or "pred_verb" not in recs[0]:
        return None
    n = g1 = g2 = out = 0
    g1_keep = g2_fix = g2_other = wm_follow = correct = 0
    for r in recs:
        m = ref.get(r["sample_id"])
        if not m:
            continue
        n += 1
        gt, pred = m["gt"], (r.get("pred_verb"), r.get("pred_noun"))
        ok = pred == gt
        correct += ok
        wm_follow += (pred == m["wm1"])
        if gt not in m["top5"]:
            out += 1
            continue
        if m["wm1"] == gt:
            g1 += 1; g1_keep += ok
        else:
            g2 += 1; g2_fix += ok
            # G2 실패의 형태: WM top-1 을 베낀 것인가, 제3의 오답으로 흩어진 것인가.
            # 후자가 많으면 pairwise margin 보다 listwise CE 가 맞다.
            if not ok and pred != m["wm1"]:
                g2_other += 1
    if not (n and g1 and g2):
        return None
    keep, fix = g1_keep / g1, g2_fix / g2
    return {
        "n": n, "acc": round(correct / n, 4),
        "P_G1": round(g1 / n, 4), "P_G2": round(g2 / n, 4), "P_OUT": round(out / n, 4),
        "g1_retention": round(keep, 4), "g1_regression": round(1 - keep, 4),
        "g2_correction": round(fix, 4), "g2_non_gt_switch": round(g2_other / g2, 4),
        "wm_follow": round(wm_follow / n, 4),
        # 분해식이 실측 acc 를 재현하는지 — 크게 어긋나면 정의나 records 가 어긋난 것이다
        "acc_reconstructed": round((g1 / n) * keep + (g2 / n) * fix, 4),
        "ceiling_R5": round((g1 + g2) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="heldout jsonl (topk_actions 포함)")
    ap.add_argument("--records", nargs="+", required=True, help="*.records.jsonl 하나 이상")
    ap.add_argument("--out", default=None, help="JSON 저장 경로 (선택)")
    args = ap.parse_args()

    ref = load_reference(args.jsonl)
    # L0: 무학습 베이스라인 — 항상 WM top-1 을 따르는 정책
    l0 = sum(1 for m in ref.values() if m["wm1"] == m["gt"]) / len(ref)

    hdr = (f"{'arm':<34}{'n':>5}{'acc':>8}{'G1보존':>8}{'G1퇴행':>8}"
           f"{'G2교정':>8}{'G2→타오답':>10}{'wm추종':>8}")
    print(f"heldout n={len(ref)}  ·  L0(WM top-1 무학습) = {l0:.4f}  ← 모든 arm 이 넘어야 하는 선")
    print(hdr); print("-" * len(hdr))
    print(f"{'L0  WM top-1 (무학습)':<34}{len(ref):>5}{l0:>8.4f}{1.0:>8.4f}{0.0:>8.4f}"
          f"{0.0:>8.4f}{0.0:>10.4f}{1.0:>8.4f}")
    out_all = {"heldout_n": len(ref), "L0_wm_top1": round(l0, 4), "arms": {}}
    for p in args.records:
        d = decompose(p, ref)
        name = Path(p).name.replace(".records.jsonl", "")
        if not d:
            print(f"{name:<34}  (records 형식 불일치 — 건너뜀)"); continue
        out_all["arms"][name] = d
        flag = "" if d["acc"] > l0 else "  ← L0 미달"
        print(f"{name:<34}{d['n']:>5}{d['acc']:>8.4f}{d['g1_retention']:>8.4f}"
              f"{d['g1_regression']:>8.4f}{d['g2_correction']:>8.4f}"
              f"{d['g2_non_gt_switch']:>10.4f}{d['wm_follow']:>8.4f}{flag}")
    print(f"\n구조적 상한 = P(G1)+P(G2) = {out_all['arms'] and list(out_all['arms'].values())[0]['ceiling_R5'] if out_all['arms'] else '—'}"
          f"   (OUT 구간은 candidate-constrained 로 불가)")
    if args.out:
        Path(args.out).write_text(json.dumps(out_all, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[done] → {args.out}")


if __name__ == "__main__":
    main()
