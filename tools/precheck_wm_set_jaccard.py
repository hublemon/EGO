#!/usr/bin/env python3
"""Precheck (d): WM top-K 집합의 예시 간 구별성 — L3(instance-specificity) 검정력 사전 진단.

핵심 질문: P_true vs P_shuffle을 가를 수 있으려면, 한 예시의 진짜 WM 집합이 무작위
다른 예시의 집합과 실제로 달라야 한다. shuffle이 곧 cross-example 페어링이므로,
**cross-example Jaccard 평균 = 진짜 집합과 그 shuffled 대체본의 기대 겹침**이다.
  높으면(예 0.6) → shuffle이 60% 그대로라 P_true≈P_shuffle → L3 검정력 없음 (9h 재고)
  낮으면(예 0.15) → shuffle이 진짜 다름 → L3 검정력 있음

대조군: 전역 action 빈도에서 뽑은 랜덤 집합의 Jaccard(null). 관측 겹침이 null과
비슷하면 "집합 = 빈도 draw"라는 뜻이라 더더욱 나쁘다.

사용: PYTHONPATH=src python3 tools/precheck_wm_set_jaccard.py [--runs runs/retro4] [--pairs 20000]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def summ(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
    return {"mean": round(sum(xs) / n, 4), "p10": round(q(.1), 4), "median": round(q(.5), 4),
            "p90": round(q(.9), 4), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/retro4")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--level", choices=["action", "verb", "noun"], default="action",
                    help="집합 원소 단위 — action(verb noun) / verb / noun")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(args.runs) / "data" / f"support_{args.split}.jsonl")]

    def elems(cands: list[str]) -> set:
        if args.level == "action":
            return set(cands)
        i = 0 if args.level == "verb" else 1
        return {c.split(" ")[i] if " " in c else c for c in cands}

    sets = [elems(r["candidates"]) for r in rows]
    vids = [r["video_uid"] for r in rows]
    K = len(rows[0]["candidates"])
    rng = random.Random(args.seed)

    # 전역 빈도 (null 모델용) — 원소별 등장 횟수
    freq = Counter()
    for s in sets:
        freq.update(s)
    pool = list(freq)
    weights = [freq[e] for e in pool]
    uniq = len(pool)

    # 관측: cross-example / within-video / across-video Jaccard
    cross, within_v, across_v = [], [], []
    by_vid: dict[str, list[int]] = {}
    for i, v in enumerate(vids):
        by_vid.setdefault(v, []).append(i)
    vid_list = list(by_vid)

    for _ in range(args.pairs):
        i, j = rng.randrange(len(sets)), rng.randrange(len(sets))
        if i == j:
            continue
        jc = jaccard(sets[i], sets[j])
        cross.append(jc)
        (within_v if vids[i] == vids[j] else across_v).append(jc)

    # null: 전역 빈도에서 K개 비복원 추출한 랜덤 집합쌍의 Jaccard
    def rand_set() -> set:
        s = set()
        while len(s) < K:
            s.add(rng.choices(pool, weights=weights, k=1)[0])
        return s
    null = [jaccard(rand_set(), rand_set()) for _ in range(min(args.pairs, 5000))]

    # 원소 집중도: 상위 20개 원소가 전체 집합 멤버십의 몇 %를 차지하나
    total_mem = sum(freq.values())
    top20 = sum(c for _, c in freq.most_common(20))

    report = {
        "runs": args.runs, "split": args.split, "level": args.level,
        "n_samples": len(rows), "K": K, "unique_elements": uniq, "videos": len(by_vid),
        "jaccard_cross_example": summ(cross),
        "jaccard_within_video": summ(within_v) if within_v else None,
        "jaccard_across_video": summ(across_v) if across_v else None,
        "jaccard_null_frequency": summ(null),
        "top20_membership_share": round(top20 / total_mem, 4),
        "verdict_hint": None,
    }
    m = report["jaccard_cross_example"]["mean"]
    nullm = report["jaccard_null_frequency"]["mean"]
    # 판정 힌트: cross가 낮고(<0.35) null보다 유의하게 낮으면 L3 검정력 有
    if m >= 0.5:
        report["verdict_hint"] = f"HIGH overlap ({m}) — shuffle≈true, L3 검정력 희박 (9h 재고)"
    elif m <= 0.3 and m < nullm - 0.05:
        report["verdict_hint"] = f"LOW overlap ({m}) < null ({nullm}) — 집합이 instance-distinctive, L3 검정력 有"
    else:
        report["verdict_hint"] = f"MID overlap ({m}) vs null ({nullm}) — 경계선, 검정력 제한적"

    out = Path(args.runs) / "eval" / f"precheck_jaccard_{args.level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n[precheck] written: {out}")


if __name__ == "__main__":
    main()
