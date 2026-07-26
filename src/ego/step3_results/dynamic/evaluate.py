"""Closed-loop 채점 — 선택 정확도 · WM 대비 분해 · 진행 곡선 · 회복 · paired 비교.

VPA 지표(SR/mAcc/mIoU)는 "한 번에 T개"용이라 여기서는 쓰지 않는다. 닫힌 루프에서 볼 것은
**스텝별 선택이 맞았는가**, 그리고 **틀린 뒤 궤적이 어떻게 되는가**이다.

  SelAcc          스텝 정확도 (전체)
  SelAcc|covered  GT 가 WM top-10 안에 있는 스텝만 — 구조적으로 정답 가능한 부분집합
  WM top-1        같은 스텝에서 월드 모델 단독 (모든 arm 공통 바닥선)
  G1              WM top-1 이 맞은 지점을 유지한 비율   (모방으로도 오를 수 있음)
  GADR            WM top-1 이 틀린 지점을 교정한 비율   (모방으로는 오를 수 없음)
  progress curve  에피소드 진행(정규화 위치 5분위)별 정확도 — 논문 신호 (1)
  recovery        오답 이후 다시 맞히기까지 걸린 스텝 수 — 논문 신호 (2)
  hist_purity     그 스텝에서 프롬프트에 들어간 자기 히스토리 중 GT 와 일치한 비율(오염도)

CI 는 **영상 클러스터 부트스트랩**(영상당 10~40 스텝이 상관돼 있음), arm 비교는 **paired**.

사용:
  PYTHONPATH=src python -m ego.step3_results.dynamic.evaluate --arms ego_closed ego_nobelief oracle_gt_hist
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ego.step3_results.dynamic import common as C

N_BINS = 5


def load_records(pred_dir: Path, arm: str) -> list[dict]:
    recs = C.read_jsonl(pred_dir / f"{arm}.records.jsonl")
    # 재개 시 같은 (video, step) 이 두 번 기록될 수 있다 — 마지막 것을 채택
    latest: dict[tuple, dict] = {}
    for r in recs:
        latest[(r["video_uid"], r["step_idx"])] = r
    return [latest[k] for k in sorted(latest)]


def episode_lengths(recs: list[dict]) -> dict[str, int]:
    n: dict[str, int] = defaultdict(int)
    for r in recs:
        n[r["video_uid"]] = max(n[r["video_uid"]], r["step_idx"] + 1)
    return n


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    n = len(recs)
    corr = [r["correct"] for r in recs]
    wm = [r["wm_top1"] == r["gt_action"] for r in recs]
    cov = [r["gt_in_candidates"] for r in recs]
    covered = [c for c, k in zip(corr, cov) if k]
    g1 = [c for c, w in zip(corr, wm) if w]
    gadr = [c for c, w in zip(corr, wm) if not w]
    agree = [r["pred_action"] == r["wm_top1"] for r in recs]

    lens = episode_lengths(recs)
    bins: list[list[bool]] = [[] for _ in range(N_BINS)]
    for r in recs:
        L = max(1, lens[r["video_uid"]])
        bins[min(N_BINS - 1, int(r["step_idx"] / L * N_BINS))].append(r["correct"])

    # 회복: 오답 스텝 이후 같은 에피소드에서 다시 맞히기까지의 스텝 수 (끝까지 못 맞히면 미집계)
    per_ep: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        per_ep[r["video_uid"]].append(r)
    recov, never = [], 0
    for v, rs in per_ep.items():
        rs.sort(key=lambda r: r["step_idx"])
        flags = [r["correct"] for r in rs]
        for i, ok in enumerate(flags):
            if ok:
                continue
            nxt = next((j for j in range(i + 1, len(flags)) if flags[j]), None)
            if nxt is None:
                never += 1
            else:
                recov.append(nxt - i)

    # 히스토리 오염도: 스텝 k 프롬프트에 실린 자기 히스토리 중 GT 와 일치한 비율
    gt_by = {(r["video_uid"], r["step_idx"]): r["gt_action"] for r in recs}
    pur = []
    for r in recs:
        h = r.get("history_used") or []
        if not h:
            continue
        gts = [gt_by.get((r["video_uid"], i)) for i in range(len(h))]
        pur.append(sum(1 for a, g in zip(h, gts) if g is not None and a == g) / len(h))

    m = lambda x: round(100 * float(np.mean(x)), 2) if len(x) else None  # noqa: E731
    return {
        "n": n, "n_episodes": len(per_ep),
        "SelAcc": m(corr), "SelAcc_covered": m(covered), "n_covered": len(covered),
        "coverage": m(cov), "WM_top1": m(wm), "WM_top1_covered": m([w for w, k in zip(wm, cov) if k]),
        "G1": m(g1), "n_G1": len(g1), "GADR": m(gadr), "n_GADR": len(gadr),
        "agree_with_WM_top1": m(agree),
        "malformed": m([r["malformed"] for r in recs]), "forced": m([r["forced"] for r in recs]),
        "progress_curve": [m(b) for b in bins],
        "progress_curve_n": [len(b) for b in bins],
        "recovery_steps_mean": round(float(np.mean(recov)), 2) if recov else None,
        "recovery_steps_median": float(np.median(recov)) if recov else None,
        "n_never_recovered": never,
        "hist_purity": m(pur),
    }


def cluster_ci(recs: list[dict], key, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """영상 클러스터 부트스트랩 95% CI (percent 단위)."""
    by: dict[str, list[bool]] = defaultdict(list)
    for r in recs:
        v = key(r)
        if v is not None:
            by[r["video_uid"]].append(bool(v))
    keys = [k for k in by if by[k]]
    if not keys:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        vals = [x for i in picked for x in by[keys[i]]]
        draws.append(100 * float(np.mean(vals)))
    return [round(float(np.percentile(draws, 2.5)), 2), round(float(np.percentile(draws, 97.5)), 2)]


def paired_delta(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """A − B (SelAcc). 공통 스텝만, 영상 클러스터를 복원추출해 **차이의 분포**를 직접 구한다."""
    ai = {(r["video_uid"], r["step_idx"]): r["correct"] for r in a}
    bi = {(r["video_uid"], r["step_idx"]): r["correct"] for r in b}
    common = sorted(set(ai) & set(bi))
    by: dict[str, list[tuple]] = defaultdict(list)
    for k in common:
        by[k[0]].append(k)
    keys = list(by)
    if not keys:
        return {"n_paired": 0}
    pa = 100 * float(np.mean([ai[k] for k in common]))
    pb = 100 * float(np.mean([bi[k] for k in common]))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        ks = [k for i in picked for k in by[keys[i]]]
        draws.append(100 * (float(np.mean([ai[k] for k in ks])) - float(np.mean([bi[k] for k in ks]))))
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"n_paired": len(common), "n_clusters": len(keys),
            "A_SelAcc": round(pa, 2), "B_SelAcc": round(pb, 2), "delta": round(pa - pb, 2),
            "ci95": [round(lo, 2), round(hi, 2)], "significant": bool(lo > 0 or hi < 0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", default="runs/dynamic_v1/preds")
    p.add_argument("--out-dir", default="runs/dynamic_v1/metrics")
    p.add_argument("--arms", nargs="+", default=list(C.ARMS))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    loaded = {a: load_records(pred_dir, a) for a in args.arms}
    loaded = {a: r for a, r in loaded.items() if r}
    if not loaded:
        raise SystemExit(f"no records in {pred_dir}")

    # 공통 스텝으로 맞춘 뒤 채점 — arm 간 진행도가 다르면 비교가 깨진다
    common = set.intersection(*[{(r["video_uid"], r["step_idx"]) for r in rs} for rs in loaded.values()])
    report = {"arms": {}, "common_steps": len(common), "n_arms": len(loaded)}
    for a, rs in loaded.items():
        full = summarize(rs)
        cm = [r for r in rs if (r["video_uid"], r["step_idx"]) in common]
        s = summarize(cm)
        s["ci95_SelAcc"] = cluster_ci(cm, lambda r: r["correct"], args.n_boot)
        s["ci95_SelAcc_covered"] = cluster_ci([r for r in cm if r["gt_in_candidates"]],
                                              lambda r: r["correct"], args.n_boot)
        s["n_all_records"] = full["n"]
        report["arms"][a] = s

    order = [a for a in ("ego_closed", "ego_nobelief", "oracle_gt_hist") if a in loaded]
    pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    report["paired"] = {f"{x}__vs__{y}": paired_delta(
        [r for r in loaded[x] if (r["video_uid"], r["step_idx"]) in common],
        [r for r in loaded[y] if (r["video_uid"], r["step_idx"]) in common], args.n_boot)
        for x, y in pairs}

    # 에피소드별 표 (정성 리뷰 페이지와 대조용)
    report["per_episode"] = {}
    for a, rs in loaded.items():
        per: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            per[r["video_uid"]].append(r)
        report["per_episode"][a] = {
            v: {"n": len(x), "SelAcc": round(100 * float(np.mean([q["correct"] for q in x])), 1),
                "coverage": round(100 * float(np.mean([q["gt_in_candidates"] for q in x])), 1)}
            for v, x in sorted(per.items())}

    C.dump_json(Path(args.out_dir) / "metrics.json", report)

    w = 16
    print(f"공통 스텝 {len(common)}개 기준\n")
    hdr = ["metric"] + list(report["arms"])
    print("".join(h.rjust(w) for h in hdr))
    for k in ("n", "SelAcc", "ci95_SelAcc", "SelAcc_covered", "WM_top1", "WM_top1_covered",
              "G1", "GADR", "agree_with_WM_top1", "hist_purity", "malformed", "forced",
              "recovery_steps_mean", "n_never_recovered", "progress_curve"):
        row = [k] + [str(report["arms"][a].get(k)) for a in report["arms"]]
        print("".join(str(c).rjust(w) for c in row))
    print("\npaired ΔSelAcc (video-cluster bootstrap):")
    for name, d in report["paired"].items():
        if d.get("n_paired"):
            print(f"  {name:38s} Δ={d['delta']:+6.2f}  CI95[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]  "
                  f"{'SIG' if d['significant'] else 'ns'}")
    print(f"\nwrote {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
