"""논문 핵심 지표 — SelAcc@K / GADR / G1-retention 단계별 측정 (records 기반, GPU 불필요).

논문 eq 12-14:
  Coverage@K = Pr(GT ∈ Dt)                               [battery summary의 pool_coverage]
  SelAcc@K   = Pr(a=GT | GT∈Dt)                          [records covered-only]
  GADR       = Pr(a=GT | GT∈Dt, WM-top1 ≠ GT)            [모방으로 개선 불가한 부분]
  G1-retention = Pr(a=GT | GT∈Dt, WM-top1 = GT)          [WM이 맞은 걸 지키나]

핵심 방어 판정:
  · SelAcc(LM) > SelAcc(WM-top1 모방)?  → LM이 순효과로 더 낫나
  · GADR(LM) > 0 이고 stage별로 오르나? → "모방을 넘어선다"는 논문 주장
  · G1-retention이 낮으면 → LM이 WM의 정답을 버린다는 진단

사용: PYTHONPATH=src python3 tools/measure_gadr.py \
        --records runs/retro3/eval/base.records.jsonl:base \
                  runs/retro3/eval/r1_sft.records.jsonl:r1_sft [...]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def boot_ci(flags, n_boot=2000, seed=0):
    """이진 배열의 mean + 95% bootstrap CI."""
    if not flags:
        return (0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(flags)
    point = sum(flags) / n
    bs = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        bs.append(s / n)
    bs.sort()
    return (round(point, 4), round(bs[int(.025 * n_boot)], 4), round(bs[int(.975 * n_boot)], 4), n)


def analyze(path: Path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    # 모두 covered(gt_in_support) 가정 (battery covered_only). 방어적으로 필터.
    cov = [r for r in recs if r.get("gt_in_support")]
    correct = lambda r: int(bool(r.get("correct")))  # noqa: E731
    g1_bucket = [r for r in cov if r.get("wm_top1_correct")]        # WM top-1 = GT
    gadr_bucket = [r for r in cov if not r.get("wm_top1_correct")]  # WM top-1 ≠ GT

    sel = boot_ci([correct(r) for r in cov])
    gadr = boot_ci([correct(r) for r in gadr_bucket])
    g1 = boot_ci([correct(r) for r in g1_bucket])
    mal = sum(1 for r in cov if r.get("malformed")) / max(1, len(cov))
    # WM-top1 모방 참조: covered에서 wm_top1_correct 비율 = SelAcc if LM==WM-top1
    sel_wm = sum(1 for r in cov if r.get("wm_top1_correct")) / max(1, len(cov))
    return {
        "n_covered": len(cov),
        "SelAcc": sel, "GADR": gadr, "G1_retention": g1,
        "SelAcc_WMtop1_ref": round(sel_wm, 4),
        "gadr_bucket_frac": round(len(gadr_bucket) / max(1, len(cov)), 3),
        "malformed_rate": round(mal, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="path:label ...")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for spec in args.records:
        path, _, label = spec.partition(":")
        label = label or Path(path).stem
        if not Path(path).is_file():
            print(f"[skip] {path} 없음"); continue
        results[label] = analyze(Path(path))

    # 출력 표
    print(f"\n{'arm':16s} {'n':>5s} {'SelAcc':>18s} {'GADR':>18s} {'G1-ret':>8s} {'WMtop1':>7s} {'mal':>5s}")
    for lab, r in results.items():
        s, g, g1 = r["SelAcc"], r["GADR"], r["G1_retention"]
        beat = "↑WM" if s[1] > r["SelAcc_WMtop1_ref"] else ("↓WM" if s[2] < r["SelAcc_WMtop1_ref"] else "≈WM")
        print(f"{lab:16s} {r['n_covered']:5d} "
              f"{s[0]:.3f}[{s[1]:.3f},{s[2]:.3f}] {g[0]:.3f}[{g[1]:.3f},{g[2]:.3f}] "
              f"{g1[0]:>6.3f}  {r['SelAcc_WMtop1_ref']:>6.3f} {beat} {r['malformed_rate']:.2f}")

    # 핵심 판정
    print("\n[판정]")
    for lab, r in results.items():
        s = r["SelAcc"]; g = r["GADR"]
        v1 = "PASS" if s[1] > r["SelAcc_WMtop1_ref"] else "FAIL"
        v2 = "PASS" if g[1] > 0 else "FAIL"
        print(f"  {lab:14s} SelAcc>WMtop1: {v1} (LM {s[0]} vs WM {r['SelAcc_WMtop1_ref']}) · "
              f"GADR>0: {v2} ({g[0]}) · G1유지 {r['G1_retention'][0]}")
    # base→sft GADR 델타 (같은 계보끼리)
    labs = list(results)
    if "base" in labs:
        b = results["base"]["GADR"][0]
        for lab in labs:
            if lab != "base" and "sft" in lab.lower():
                d = results[lab]["GADR"][0] - b
                print(f"  Δ GADR ({lab} − base) = {d:+.4f}  ({'개선' if d > 0.01 else '정체/하락'})")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
