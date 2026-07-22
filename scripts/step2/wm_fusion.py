#!/usr/bin/env python3
"""wm_fusion.py — 추론 시 WM prior 융합: s_final = s_θ + α·log p_WM  (GPU 불필요)

배경: Exp-A(candidate CE)와 L0(WM top-1 무학습)는 상보적이다 —
  모델  G1 0.671 / G2 0.520          L0  G1 1.000 / G2 0.000
학습으로 selective trust 를 가르치는 것(Exp-C)은 실패했다: G1/G2 구분이 입력에서 추론
불가능해 앵커가 전역 prior 이동으로 번졌다 (kw↑ → G1 +0.009 / G2 −0.019, 순손실).
융합은 그 구분 자체를 요구하지 않는다 — WM 확률이라는 연속 신호를 점수에 직접 더한다.

── α 선정 규약 (사전 등록) ──────────────────────────────────────────────────
Exp-A 가 train oracle subset 전체를 학습했으므로 train 에서 뗀 dev 는 오염된다
(본 샘플이라 s_θ 가 과신됨 → α 편향). 따라서:
  dev  = heldout 전반부 708   ← α 는 여기서만 argmax (동률이면 작은 α)
  test = heldout 후반부 709   ← 성공 판정은 여기서 1회
  전량 1,417 은 참고로만 보고 (α 가 그 절반에서 선정됐음을 명기)
성공 = test acc(α*) > test L0.  α 그리드 = 0.00~4.00 step 0.05.

    python scripts/step2/wm_fusion.py --records eval_scored_dump.records.jsonl --out fusion.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(records: str) -> list[dict]:
    rows = []
    for line in open(records, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "scores" not in r or "wm_lik" not in r:
            raise SystemExit("records 에 scores/wm_lik 가 없다 — --dump 버전 eval 로 다시 만들 것")
        rows.append(r)
    return rows


def fused_metrics(rows: list[dict], alpha: float) -> dict:
    n = correct = g1 = g2 = g1k = g2f = l0 = 0
    for r in rows:
        liks = [x if (x is not None and x > 0) else 1e-9 for x in r["wm_lik"]]
        s = sum(liks)
        pw = [x / s for x in liks]                      # top-5 재정규화 (pro_gr 규약)
        f = [sc + alpha * math.log(p) for sc, p in zip(r["scores"], pw)]
        pred = f.index(max(f))
        gt_idx, wm1 = r.get("gt_idx"), r.get("wm1_idx")
        ok = gt_idx is not None and pred == gt_idx
        n += 1; correct += ok
        l0 += (gt_idx is not None and wm1 == gt_idx)
        if gt_idx is None:
            continue                                    # OUT — 구조적으로 불가
        if wm1 == gt_idx:
            g1 += 1; g1k += ok
        else:
            g2 += 1; g2f += ok
    return {"n": n, "acc": correct / n, "L0": l0 / n,
            "g1_retention": g1k / g1 if g1 else None,
            "g2_correction": g2f / g2 if g2 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="scores/wm_lik 포함 records.jsonl")
    ap.add_argument("--alpha_max", type=float, default=4.0)
    ap.add_argument("--alpha_step", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.records)
    half = len(rows) // 2
    dev, test = rows[:half], rows[half:]                # 하니스의 contiguous half 관례
    print(f"[data] 전체 {len(rows)} = dev {len(dev)} + test {len(test)}")

    grid = [round(i * args.alpha_step, 2)
            for i in range(int(args.alpha_max / args.alpha_step) + 1)]
    curve = []
    best = None
    for a in grid:
        m = fused_metrics(dev, a)
        curve.append({"alpha": a, "dev_acc": round(m["acc"], 4)})
        if best is None or m["acc"] > best[1] + 1e-12:  # 동률이면 작은 α 유지
            best = (a, m["acc"])
    a_star = best[0]

    res_dev = fused_metrics(dev, a_star)
    res_test = fused_metrics(test, a_star)              # ★ 성공 판정은 여기 1회
    res_full = fused_metrics(rows, a_star)
    res_test0 = fused_metrics(test, 0.0)                # α=0 = 순수 Exp-A (대조)

    out = {
        "records": args.records, "n_dev": len(dev), "n_test": len(test),
        "alpha_grid": {"max": args.alpha_max, "step": args.alpha_step},
        "alpha_star": a_star,
        "dev": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res_dev.items()},
        "test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res_test.items()},
        "test_alpha0": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res_test0.items()},
        "full_reference_only": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in res_full.items()},
        "verdict_preregistered": ("성공 — test 에서 L0 초과"
                                  if res_test["acc"] > res_test["L0"] else
                                  "실패 — test 에서 L0 미달"),
        "curve": curve,
    }
    print(f"\n  α* = {a_star}  (dev acc {res_dev['acc']:.4f}, dev L0 {res_dev['L0']:.4f})")
    print(f"  test:  융합 {res_test['acc']:.4f}  vs  α=0 {res_test0['acc']:.4f}  vs  L0 {res_test['L0']:.4f}")
    print(f"         G1 {res_test['g1_retention']:.4f}  G2 {res_test['g2_correction']:.4f}")
    print(f"  전량(참고): {res_full['acc']:.4f}  (L0 {res_full['L0']:.4f})")
    print(f"  판정: {out['verdict_preregistered']}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {args.out}")


if __name__ == "__main__":
    main()
