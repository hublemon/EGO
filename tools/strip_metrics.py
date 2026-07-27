#!/usr/bin/env python3
"""strip_metrics.py — history_strip 지표 (대시보드 규약: covered · video-cluster paired bootstrap).

`tools/oom_opt/strip_eval.py` 가 만든 `{arm}_nohist.records.jsonl` 을 같은 arm 의
`{arm}.records.jsonl` 과 per-sample_id paired 로 붙여, history 개입 효과를
G-ACC1/G-NH 와 **같은 통계 규약**으로 산출한다:

  * 모집단 = covered (GT∈WM Top-10) — 대시보드 §1 과 동일. uncovered 는 전 arm 강제 0점.
  * 재표집 단위 = video_uid 클러스터 (tools/paired_boot.paired_cluster_delta 재사용).
  * 지표 = SelAcc / GADR / G1 · Δ = hist − strip (양수면 history 를 인과적으로 사용).
  * history_length 층화(H0/H1-3/H4-7/H8+) — 긴 history 에서 Δ 가 커야 "이력을 읽는다".

`strip_verdict.json`(기존)과의 차이: 그쪽은 full-set(uncovered 포함) · sample bootstrap
이라 Δ가 구조적 0점에 희석된다. 본 스크립트는 covered · cluster bootstrap.

채점 규약 — malformed 는 **오답 처리**(pairing 유지). 두 조건 모두 non-malformed 인
부분집합만 쓰는 보수적 값은 `selacc_strict` 로 병기.

게이트
  G-HIST   : Δ SelAcc(covered) CI 하한 > 0 → 해당 체크포인트가 history 를 인과적으로 사용.
  G-HIST8  : H8+ 층에서 동일 조건 (긴 이력 특이성).

CLI:
    python3 tools/strip_metrics.py --run runs/cesft_v2 \
        --arms base theta_ce sft_r15 --out runs/cesft_v2/eval/strip_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_boot import (cluster_key, load_context_video_uids,  # noqa: E402
                         paired_cluster_delta, read_jsonl)

METRICS = ("SelAcc", "GADR", "G1")
HBINS = ("H0", "H1-3", "H4-7", "H8+")


def hbin(h: int | None) -> str:
    if h is None:
        return "?"
    if h == 0:
        return "H0"
    if h <= 3:
        return "H1-3"
    if h <= 7:
        return "H4-7"
    return "H8+"


def load_condition(run: Path, arm: str, suffix: str = "") -> dict[str, dict]:
    """{sample_id: row} — malformed/error 행도 유지(오답 처리용). covered 필터는 나중에."""
    rows = read_jsonl(run / "eval" / f"{arm}{suffix}.records.jsonl")
    return {r["sample_id"]: r for r in rows if r.get("sample_id")}


def scored(r: dict) -> bool:
    """malformed-as-wrong 채점."""
    return bool(r.get("correct")) and not r.get("malformed") and r.get("error") is None


def context_meta(run: Path) -> tuple[dict[str, int], dict[str, str]]:
    ctx = read_jsonl(run / "data" / "context_val.jsonl")
    hlen = {r["sample_id"]: len(r.get("history", [])) for r in ctx if r.get("sample_id")}
    return hlen, load_context_video_uids(run)


def arm_strip(run: Path, arm: str, n_boot: int, seed: int,
              hist_sfx: str = "", strip_sfx: str = "_nohist") -> dict:
    """한 arm 의 hist vs strip paired 분석. 데이터 없으면 error 를 담아 반환.

    hist_sfx/strip_sfx 로 어떤 조건 쌍을 볼지 고른다. 2026-07-27 이후 권장 조합은
    같은 세션에서 함께 돌린 쌍(예: hist_sfx="_hist_v3", strip_sfx="_nohist_v3") —
    아카이브 배터리(hist_sfx="")와 신규 strip 을 섞으면 배치 잡음이 Δ 에 실린다.
    """
    out: dict = {"arm": arm, "n_paired": 0, "n_clusters": 0,
                 "hist_file": f"{arm}{hist_sfx}.records.jsonl",
                 "strip_file": f"{arm}{strip_sfx}.records.jsonl"}
    hist = load_condition(run, arm, hist_sfx)
    strip = load_condition(run, arm, strip_sfx)
    if not hist:
        out["error"] = f"missing {out['hist_file']}"
        return out
    if not strip:
        out["error"] = f"missing {out['strip_file']} — strip 추론 미실행"
        return out

    hlen, ctx_map = context_meta(run)
    # covered 판정은 sample 속성 → hist 행의 gt_in_support 사용(두 조건 동일 후보셋).
    sids = sorted(s for s in hist if s in strip and hist[s].get("gt_in_support"))
    if not sids:
        out["error"] = "empty covered inner-join"
        return out

    by_cluster: dict[str, list] = {}
    by_bin: dict[str, dict[str, list]] = {b: {} for b in HBINS}
    for sid in sids:
        h, s = hist[sid], strip[sid]
        wm = bool(h.get("wm_top1_correct"))
        tup = (scored(h), wm, scored(s), wm)  # A=hist, B=strip, 같은 WM 기준선
        key = cluster_key(sid, ctx_map)
        by_cluster.setdefault(key, []).append(tup)
        b = hbin(hlen.get(sid))
        if b in by_bin:
            by_bin[b].setdefault(key, []).append(tup)

    clusters = sorted(by_cluster)
    out["n_paired"] = len(sids)
    out["n_clusters"] = len(clusters)

    out["delta"] = {}
    for m in METRICS:
        d = paired_cluster_delta(clusters, by_cluster, m, m, n_boot=n_boot, seed=seed)
        out["delta"][m] = {
            "hist": round(100 * d["point_a"], 2),
            "strip": round(100 * d["point_b"], 2),
            "delta_pp": round(100 * d["point"], 2),
            "ci": [round(100 * d["lo"], 2), round(100 * d["hi"], 2)],
        }

    out["delta_selacc_by_hlen"] = {}
    for b in HBINS:
        cl = sorted(by_bin[b])
        if not cl:
            continue
        d = paired_cluster_delta(cl, by_bin[b], "SelAcc", "SelAcc", n_boot=n_boot, seed=seed)
        out["delta_selacc_by_hlen"][b] = {
            "delta_pp": round(100 * d["point"], 2),
            "ci": [round(100 * d["lo"], 2), round(100 * d["hi"], 2)],
            "n": sum(len(v) for v in by_bin[b].values()),
            "n_clusters": len(cl),
        }

    out["malformed_pct"] = {
        "hist": round(100 * sum(bool(hist[s].get("malformed")) for s in sids) / len(sids), 2),
        "strip": round(100 * sum(bool(strip[s].get("malformed")) for s in sids) / len(sids), 2),
    }
    strict = [s for s in sids
              if not hist[s].get("malformed") and not strip[s].get("malformed")]
    if strict:
        out["selacc_strict"] = {
            "n": len(strict),
            "hist": round(100 * sum(scored(hist[s]) for s in strict) / len(strict), 2),
            "strip": round(100 * sum(scored(strip[s]) for s in strict) / len(strict), 2),
        }
        out["selacc_strict"]["delta_pp"] = round(
            out["selacc_strict"]["hist"] - out["selacc_strict"]["strip"], 2)

    sel = out["delta"]["SelAcc"]
    h8 = out["delta_selacc_by_hlen"].get("H8+")
    out["gate_G-HIST"] = bool(sel["ci"][0] > 0)
    out["gate_G-HIST8"] = bool(h8 and h8["ci"][0] > 0)
    return out


def cross_arm_did(run: Path, a: str, b: str, n_boot: int, seed: int,
                  hist_sfx: str = "", strip_sfx: str = "_nohist") -> dict:
    """difference-in-differences: Δ_strip(a) − Δ_strip(b) — 학습이 history 의존을 키웠나.

    두 arm 의 hist·strip 4조건 모두에 존재하는 sample 만(공통 covered) 사용하고,
    클러스터 재표집을 공유해 arm 간 상관을 유지한다.
    """
    out = {"arm_a": a, "arm_b": b, "n_paired": 0}
    cond = {(x, k): load_condition(run, x, strip_sfx if k == "strip" else hist_sfx)
            for x in (a, b) for k in ("hist", "strip")}
    for key, d in cond.items():
        if not d:
            out["error"] = f"missing records for {key}"
            return out
    _, ctx_map = context_meta(run)
    sids = sorted(s for s in cond[(a, "hist")]
                  if all(s in cond[k] for k in cond) and cond[(a, "hist")][s].get("gt_in_support")
                  and cond[(b, "hist")][s].get("gt_in_support"))
    if not sids:
        out["error"] = "empty 4-way inner-join"
        return out

    # Δ_a − Δ_b 를 sample 단위 스칼라로 접고 SelAcc 규약으로 클러스터 부트스트랩.
    by_cluster: dict[str, list] = {}
    for sid in sids:
        da = int(scored(cond[(a, "hist")][sid])) - int(scored(cond[(a, "strip")][sid]))
        db = int(scored(cond[(b, "hist")][sid])) - int(scored(cond[(b, "strip")][sid]))
        by_cluster.setdefault(cluster_key(sid, ctx_map), []).append((da, db))

    clusters = sorted(by_cluster)
    import random
    def ev(cls: list[str]) -> tuple[float, float]:
        xs = [t for c in cls for t in by_cluster[c]]
        n = max(1, len(xs))
        return sum(t[0] for t in xs) / n, sum(t[1] for t in xs) / n

    pa, pb = ev(clusters)
    rng = random.Random(seed)
    ds = []
    for _ in range(n_boot):
        drawn = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        qa, qb = ev(drawn)
        ds.append(qa - qb)
    ds.sort()
    lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]
    out.update({
        "n_paired": len(sids), "n_clusters": len(clusters),
        "delta_strip_a_pp": round(100 * pa, 2), "delta_strip_b_pp": round(100 * pb, 2),
        "did_pp": round(100 * (pa - pb), 2), "ci": [round(100 * lo, 2), round(100 * hi, 2)],
        "significant": bool(lo > 0 or hi < 0),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2")
    ap.add_argument("--arms", nargs="+", default=["base", "theta_ce", "sft_r15"])
    ap.add_argument("--did", nargs="*", default=[],
                    help="difference-in-differences 쌍 'a:b' (예: sft_r15:theta_ce)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist_suffix", default="",
                    help="hist 조건 파일 접미사 ('' = 아카이브 배터리, '_hist_v3' = 동일 세션 재실행)")
    ap.add_argument("--strip_suffix", default="_nohist")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    res = {
        "run": str(run),
        "convention": "covered(GT∈WM Top-10) · video-cluster paired bootstrap · "
                      "malformed=오답 · Δ = hist − strip (pp)",
        "hist_suffix": args.hist_suffix,
        "strip_suffix": args.strip_suffix,
        "arms": {a: arm_strip(run, a, args.n_boot, args.seed,
                              args.hist_suffix, args.strip_suffix) for a in args.arms},
    }
    if args.did:
        res["did"] = {}
        for spec in args.did:
            a, _, b = spec.partition(":")
            res["did"][spec] = cross_arm_did(run, a, b, args.n_boot, args.seed,
                                             args.hist_suffix, args.strip_suffix)

    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
