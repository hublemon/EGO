#!/usr/bin/env python3
"""unified_estimand — 모든 게이트를 **하나의 추정대상**으로 재계산한다.

추정대상 확정 (2026-07-27 사용자 결정):
    모집단 = 공통 covered set n=1,000 전량
    malformed = 오답 (표본에서 제외하지 않는다)

이유: 기존 `paired_boot.py` 는 arm 별 non-malformed 교집합을 쓰는데, 이는
  ① 점추정(n=1,000)과 CI(교집합)가 서로 다른 추정대상을 가리키게 만들고
  ② 제외 기준이 **모델 출력**이라 arm-dependent post-treatment selection 이 생긴다.
실측: 교집합으로 옮기면 malformed 가 많은 arm 이 더 많이 구제된다
  (θ_CE +1.77pp vs EGO +0.62pp) → EGO−θ_CE 격차가 −0.30pp 에서 −1.45pp 로 벌어진다.

부트스트랩 단위는 `paired_boot.py` 와 동일하게 **video_uid 클러스터**다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cesft_v2_fp_c"
N_BOOT = 2000
SEED = 0

ARMS = {
    "base":       RUN / "eval" / "base.records.jsonl",
    "cand_free":  RUN / "eval" / "cand_free.records.jsonl",
    "theta_ce":   RUN / "eval" / "theta_ce.records.jsonl",
    "sft_r15":    RUN / "eval" / "sft_r15.records.jsonl",
    "sft_r15_c":  RUN / "eval" / "sft_r15_c.records.jsonl",
}


def load(p: Path) -> dict:
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def vids() -> dict:
    out = {}
    for line in open(RUN / "data" / "context_val.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[r["sample_id"]] = r["video_uid"]
    return out


def ok(rec: dict) -> int:
    """malformed = 오답. 이 한 줄이 추정대상의 전부다."""
    return int(bool(rec.get("correct")) and not rec.get("malformed", False))


def boot(ids, byvid, fn_a, fn_b=None, n_boot=N_BOOT, seed=SEED):
    """video-cluster paired bootstrap. fn_* 는 sample_id -> (numer, denom) 을 준다."""
    clusters = sorted({byvid[s] for s in ids})
    grp = {c: [] for c in clusters}
    for s in ids:
        grp[byvid[s]].append(s)

    def val(drawn):
        na = da = nb = db = 0
        for c in drawn:
            for s in grp[c]:
                x, y = fn_a(s)
                na += x; da += y
                if fn_b:
                    x2, y2 = fn_b(s)
                    nb += x2; db += y2
        a = na / da if da else 0.0
        b = (nb / db if db else 0.0) if fn_b else 0.0
        return a - b, a, b

    point, pa, pb = val(clusters)
    rng = random.Random(seed)
    n = len(clusters)
    bs = sorted(val([clusters[rng.randrange(n)] for _ in range(n)])[0] for _ in range(n_boot))
    return {"point": point, "lo": bs[int(.025 * n_boot)], "hi": bs[int(.975 * n_boot)],
            "a": pa, "b": pb, "n": len(ids), "clusters": n}


def main():
    A = {k: load(p) for k, p in ARMS.items()}
    V = vids()
    ids = sorted(set.intersection(*[set(d) for d in A.values()]))
    byvid = {s: V[s] for s in ids}
    print(f"모집단 n={len(ids)}  클러스터={len({byvid[s] for s in ids})}  · malformed=오답\n")

    # ── 조건별 단일 지표 ──
    print("조건별 (n=1,000, malformed=오답)")
    print(f"{'조건':12s} {'SelAcc':>8s} {'malf':>6s} {'Retain(G1)':>11s} {'Correct(G2)':>12s}")
    for k, d in A.items():
        acc = sum(ok(d[s]) for s in ids) / len(ids)
        mal = sum(1 for s in ids if d[s].get("malformed")) / len(ids)
        g1 = [s for s in ids if d[s].get("wm_top1_correct")]
        g2 = [s for s in ids if not d[s].get("wm_top1_correct")]
        print(f"{k:12s} {100*acc:8.2f} {100*mal:6.2f} "
              f"{100*sum(ok(d[s]) for s in g1)/len(g1):11.2f} "
              f"{100*sum(ok(d[s]) for s in g2)/len(g2):12.2f}")
    wm = sum(1 for s in ids if A["theta_ce"][s].get("wm_top1_correct")) / len(ids)
    print(f"{'WM Top-1':12s} {100*wm:8.2f}")
    ng1 = sum(1 for s in ids if A["theta_ce"][s].get("wm_top1_correct"))
    print(f"   (G1 모집단 {ng1}건 / G2 모집단 {len(ids)-ng1}건)\n")

    # ── 짝지은 차이 ──
    def sel(arm):
        return lambda s: (ok(A[arm][s]), 1)

    def wmtop(s):
        return (int(A["theta_ce"][s].get("wm_top1_correct", False)), 1)

    def gadr(arm):
        return lambda s: ((0, 0) if A["theta_ce"][s].get("wm_top1_correct")
                          else (ok(A[arm][s]), 1))

    tests = [
        ("Prospection − Answer-Only  (G-DELTA)", sel("theta_ce"), sel("cand_free")),
        ("Prospection − WM Top-1     (G-ACC1)",  sel("theta_ce"), wmtop),
        ("EGO − WM Top-1             (G-ACC1)",  sel("sft_r15_c"), wmtop),
        ("EGO − Prospection  SelAcc  (G-NH)",    sel("sft_r15_c"), sel("theta_ce")),
        ("EGO − Prospection  GADR    (G-NH)",    gadr("sft_r15_c"), gadr("theta_ce")),
        ("EGO − Answer-Only",                    sel("sft_r15_c"), sel("cand_free")),
        ("fp SFT − Prospection SelAcc",          sel("sft_r15"),  sel("theta_ce")),
    ]
    print("짝지은 차이 (video-cluster bootstrap, 동일 모집단)")
    for name, fa, fb in tests:
        r = boot(ids, byvid, fa, fb)
        star = "  CI가 0 배제" if (r["lo"] > 0 or r["hi"] < 0) else ""
        print(f"  {name:38s} {100*r['point']:+6.2f}pp  "
              f"[{100*r['lo']:+6.2f}, {100*r['hi']:+6.2f}]   "
              f"({100*r['a']:.2f} vs {100*r['b']:.2f}){star}")


if __name__ == "__main__":
    main()
