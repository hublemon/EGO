"""3인 정성 평가 병합 — 다수결 · 일치도(Fleiss κ) · 정량 지표와의 교차표.

입력은 리뷰 사이트에서 각자 내보낸 `ratings_*.json` 들이다. 합의 없이 평균만 보고하면
"판정이 갈렸다"는 사실이 지워지므로, 다수결 결과와 함께 **일치도**를 반드시 같이 낸다.

산출 `runs/dynamic_v1/metrics/qualitative.json`:
  step_majority      스텝별 다수결(타당/애매/부적절)과 만장일치 여부
  fleiss_kappa       스텝 3범주 평정자간 일치도
  episode_scores     에피소드별 1~5점 평균·범위
  cross              다수결 × 정량 정오 교차표 — "GT와 다르지만 사람이 보기엔 타당" 칸이 핵심.
                     닫힌 루프에서 모델이 goal 을 향해 합리적으로 진행하되 GT 라벨과는 다른
                     경로를 택한 경우가 여기 잡힌다.

사용:
  PYTHONPATH=src python -m ego.step3_results.dynamic.merge_ratings runs/dynamic_v1/ratings/*.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ego.step3_results.dynamic import common as C

CATS = ("ok", "mid", "no")
KOR = {"ok": "타당", "mid": "애매", "no": "부적절"}


def fleiss_kappa(counts: list[list[int]]) -> float | None:
    """counts[i] = 항목 i 의 범주별 평정 수. 평정자 수가 항목마다 같아야 한다."""
    rows = [c for c in counts if sum(c) > 1]
    if not rows:
        return None
    n = sum(rows[0])
    if any(sum(c) != n for c in rows):
        rows = [c for c in rows if sum(c) == n]
        if not rows:
            return None
    N, k = len(rows), len(rows[0])
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(k)]
    P_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return None if abs(1 - P_e) < 1e-12 else round((P_bar - P_e) / (1 - P_e), 4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="ratings_*.json (평가자당 1개)")
    p.add_argument("--arm", default="ego_closed")
    p.add_argument("--pred-dir", default="runs/dynamic_v1/preds")
    p.add_argument("--out", default="runs/dynamic_v1/metrics/qualitative.json")
    args = p.parse_args()

    raters = {}
    for f in args.files:
        d = json.loads(Path(f).read_text())
        raters[d.get("rater") or Path(f).stem] = d
    print(f"평가자 {len(raters)}명: {', '.join(raters)}")

    # 정량 기록 (정오 교차표용)
    recs = {}
    for r in C.read_jsonl(Path(args.pred_dir) / f"{args.arm}.records.jsonl"):
        recs[r["sample_id"]] = r

    step_votes: dict[str, list[str]] = defaultdict(list)
    revealed: dict[str, int] = Counter()
    for name, d in raters.items():
        for sid, v in (d.get("steps") or {}).items():
            if v.get("v"):
                step_votes[sid].append(v["v"])
            if v.get("gt_revealed"):
                revealed[sid] += 1

    majority, counts_matrix, cross = {}, [], Counter()
    for sid, votes in step_votes.items():
        c = Counter(votes)
        top, n_top = c.most_common(1)[0]
        majority[sid] = {"majority": top, "unanimous": n_top == len(votes), "votes": dict(c),
                         "n_raters": len(votes), "gt_revealed_by": revealed.get(sid, 0)}
        if len(votes) >= 2:
            counts_matrix.append([c.get(x, 0) for x in CATS])
        r = recs.get(sid)
        if r:
            cross[(top, "GT일치" if r["correct"] else "GT불일치")] += 1

    ep_scores: dict[str, list[int]] = defaultdict(list)
    ep_memos: dict[str, list[str]] = defaultdict(list)
    for name, d in raters.items():
        for ep, v in (d.get("episodes") or {}).items():
            if v.get("score"):
                ep_scores[ep].append(int(v["score"]))
            if v.get("memo"):
                ep_memos[ep].append(f"[{name}] {v['memo']}")

    n_rated = len(majority)
    dist = Counter(m["majority"] for m in majority.values())
    out = {
        "arm": args.arm, "n_raters": len(raters), "raters": sorted(raters),
        "n_steps_rated": n_rated,
        "step_majority_dist": {KOR[k]: dist.get(k, 0) for k in CATS},
        "step_majority_pct": {KOR[k]: round(100 * dist.get(k, 0) / max(1, n_rated), 1) for k in CATS},
        "unanimous_rate": round(100 * sum(m["unanimous"] for m in majority.values()) / max(1, n_rated), 1),
        "fleiss_kappa": fleiss_kappa(counts_matrix),
        "cross_majority_x_gt": {f"{KOR[a]}×{b}": n for (a, b), n in sorted(cross.items())},
        "episode_scores": {ep: {"mean": round(sum(v) / len(v), 2), "scores": sorted(v),
                                "range": max(v) - min(v), "memos": ep_memos.get(ep, [])}
                           for ep, v in sorted(ep_scores.items())},
        "episode_score_mean_overall": (round(sum(s for v in ep_scores.values() for s in v)
                                             / max(1, sum(len(v) for v in ep_scores.values())), 2)
                                       if ep_scores else None),
        "per_step": majority,
    }
    C.dump_json(args.out, out)

    print(f"\n스텝 판정 {n_rated}개 · 만장일치 {out['unanimous_rate']}% · Fleiss κ={out['fleiss_kappa']}")
    for k in CATS:
        print(f"  {KOR[k]:4s} {dist.get(k, 0):5d}  ({out['step_majority_pct'][KOR[k]]}%)")
    print("\n다수결 × 정량 정오:")
    for k, v in out["cross_majority_x_gt"].items():
        print(f"  {k:14s} {v}")
    print(f"\n에피소드 종합 평균 {out['episode_score_mean_overall']} / 5  ({len(ep_scores)}개 에피소드)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
