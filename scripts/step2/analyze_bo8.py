#!/usr/bin/env python3
"""analyze_bo8.py — rerank_bo8.py 산출물 집계.

판정 규칙 (핸드오프 논의 §4):
  best-of-8 ≈ pass@8  → (a) 구현 문제. 정보는 모델 안에 있고 선택만 고치면 된다.
  best-of-8 ≈ pass@1  → (b) 근본 문제. 정답/오답이 모델 눈에 구별되지 않는다.

scorer 별로 pass@1 / random / best-of-8 / oracle(pass@8) 을 GT rank 층별로 낸다.
부트스트랩 95% CI 와 pass@1 대비 paired McNemar 를 함께 낸다.
"""
from __future__ import annotations
import argparse, json, random
from math import comb


def boot(v, B=10000, seed=7):
    rnd = random.Random(seed)
    m = len(v)
    if not m:
        return (0.0, 0.0)
    s = sorted(sum(v[rnd.randrange(m)] for _ in range(m)) / m for _ in range(B))
    return s[int(0.025 * B)], s[int(0.975 * B)]


def mcnemar(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


SCORERS = {
    "sum_logp": lambda t: t["sum"],
    "mean_logp": lambda t: t["mean"],
    "reasoning_span": lambda t: (t["span_mean"].get("reasoning") if t["span_mean"].get("reasoning") is not None else -9e9),
    "belief_span": lambda t: (t["span_mean"].get("task_belief") if t["span_mean"].get("task_belief") is not None else -9e9),
    "action_span": lambda t: (t["span_mean"].get("action") if t["span_mean"].get("action") is not None else -9e9),
    "len_short": lambda t: -t["ntok"],
    "majority": None,   # 8개 중 최빈 action (logprob 미사용 기준선)
}


def pick(traces, name):
    if name == "majority":
        from collections import Counter
        c = Counter(tuple(t["action"]) for t in traces if t["action"])
        if not c:
            return traces[0]
        top = c.most_common(1)[0][0]
        for t in traces:
            if t["action"] and tuple(t["action"]) == top:
                return t
        return traces[0]
    f = SCORERS[name]
    return max(traces, key=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.scores, encoding="utf-8")]
    n = len(rows)

    base = [bool(r["traces"][0]["correct"]) for r in rows]
    oracle = [any(t["correct"] for t in r["traces"]) for r in rows]
    rnd_exp = sum(sum(t["correct"] for t in r["traces"]) / len(r["traces"]) for r in rows) / n

    print(f"n = {n} 샘플 · 롤아웃 {sum(len(r['traces']) for r in rows)}개\n")
    lo, hi = boot(base)
    print(f'{"정책":18s} {"acc":>7s} {"95% CI":>17s}  {"vs pass@1 p":>11s}  {"pass@1→@8 회수율":>16s}')
    print(f'{"pass@1 (현재)":18s} {sum(base)/n:7.4f}  [{lo:.4f},{hi:.4f}]  {"—":>11s}  {"—":>16s}')
    print(f'{"random-of-8":18s} {rnd_exp:7.4f}  {"(기댓값)":>17s}  {"—":>11s}  {"—":>16s}')

    span = (sum(oracle) - sum(base)) / n
    results = {}
    for name in SCORERS:
        sel = [pick(r["traces"], name) for r in rows]
        v = [bool(t["correct"]) for t in sel]
        lo, hi = boot(v)
        _, _, p = mcnemar(base, v)
        rec = (sum(v) - sum(base)) / n / span if span > 0 else float("nan")
        results[name] = sum(v) / n
        print(f'{"best-of-8 · "+name:18s} {sum(v)/n:7.4f}  [{lo:.4f},{hi:.4f}]  {p:11.4g}  {rec:15.1%}')
    lo, hi = boot(oracle)
    print(f'{"oracle (pass@8)":18s} {sum(oracle)/n:7.4f}  [{lo:.4f},{hi:.4f}]  {"—":>11s}  {"100.0%":>16s}')

    print("\n--- GT rank 층별 (best-of-8 · mean_logp) ---")
    print(f'{"rank":>6s} {"n":>5s} {"pass@1":>8s} {"BoF8":>8s} {"pass@8":>8s} {"회수율":>8s}')
    for rk in [1, 2, 3, 4, 5, 0]:
        idx = [i for i, r in enumerate(rows) if r["gt_rank"] == rk]
        if not idx:
            continue
        b = [base[i] for i in idx]
        o = [oracle[i] for i in idx]
        s = [bool(pick(rows[i]["traces"], "mean_logp")["correct"]) for i in idx]
        sp = (sum(o) - sum(b)) / len(idx)
        rec = ((sum(s) - sum(b)) / len(idx) / sp) if sp > 0 else float("nan")
        print(f'{rk:6d} {len(idx):5d} {sum(b)/len(idx):8.3f} {sum(s)/len(idx):8.3f} {sum(o)/len(idx):8.3f} {rec:7.1%}')

    print("\n--- 진단: mixed 샘플에서 정답/오답 롤아웃의 점수 분리도 ---")
    sep = {"sum": [], "mean": [], "reasoning": [], "task_belief": [], "action": []}
    nmix = 0
    for r in rows:
        cs = [t for t in r["traces"] if t["correct"]]
        ws = [t for t in r["traces"] if not t["correct"]]
        if not cs or not ws:
            continue
        nmix += 1
        sep["sum"].append(sum(t["sum"] for t in cs) / len(cs) - sum(t["sum"] for t in ws) / len(ws))
        sep["mean"].append(sum(t["mean"] for t in cs) / len(cs) - sum(t["mean"] for t in ws) / len(ws))
        for k in ("reasoning", "task_belief", "action"):
            c = [t["span_mean"][k] for t in cs if t["span_mean"][k] is not None]
            w = [t["span_mean"][k] for t in ws if t["span_mean"][k] is not None]
            if c and w:
                sep[k].append(sum(c) / len(c) - sum(w) / len(w))
    print(f"mixed 샘플 {nmix}개")
    for k, v in sep.items():
        if not v:
            continue
        lo, hi = boot(v) if False else (None, None)
        mu = sum(v) / len(v)
        pos = sum(1 for x in v if x > 0) / len(v)
        print(f"  {k:12s} 평균 logp 차(정답−오답) = {mu:+.4f}   정답이 더 높은 비율 = {pos:.3f}  (우연=0.500)")

    if args.out:
        json.dump({"n": n, "pass@1": sum(base) / n, "pass@8": sum(oracle) / n,
                   "random_of_8": rnd_exp, "best_of_8": results},
                  open(args.out, "w"), indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
