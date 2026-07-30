#!/usr/bin/env python3
"""did_history.py — history 사용 이중 해리(DiD)를 두 arm 의 strip 산출물에서 계산.

입력: runs/cesft_v2/eval/{arm}.records.jsonl (with-history) 와
      runs/cesft_v2/eval/{arm}_nohist.records.jsonl (strip) — strip_eval.py 산출.
계산: Delta(arm) = acc(hist) - acc(strip), DiD = Delta(A) - Delta(B).
      같은 sample_id 4조건 paired, sample bootstrap 5,000 (seed 42).
출력: runs/cesft_v2/eval/DiD_history_{A}_vs_{B}.json
"""
import argparse
import json
import pathlib
import random


def load(ev, name):
    p = ev / name
    if not p.is_file():
        return None
    return {r["sample_id"]: bool(r.get("correct"))
            for r in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2")
    ap.add_argument("--arm_a", default="theta_ce")
    ap.add_argument("--arm_b", default="cand_free")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ev = pathlib.Path(args.run) / "eval"
    arms = {}
    for arm in (args.arm_a, args.arm_b):
        h, n = load(ev, arm + ".records.jsonl"), load(ev, arm + "_nohist.records.jsonl")
        if h and n:
            arms[arm] = {s: (h[s], n[s]) for s in h if s in n}

    out_path = ev / f"DiD_history_{args.arm_a}_vs_{args.arm_b}.json"
    if len(arms) < 2:
        out = {"error": "strip records 부족", "have": sorted(arms)}
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(json.dumps(out, ensure_ascii=False))
        return

    common = sorted(set(arms[args.arm_a]) & set(arms[args.arm_b]))

    def delta(ids, arm):
        d = [arms[arm][s] for s in ids]
        return (sum(x for x, _ in d) - sum(y for _, y in d)) / len(d) * 100

    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.n_boot):
        pick = [common[rng.randrange(len(common))] for _ in common]
        boots.append(delta(pick, args.arm_a) - delta(pick, args.arm_b))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    out = {
        "arm_a": args.arm_a, "arm_b": args.arm_b, "n_paired": len(common),
        "delta_a_pp": round(delta(common, args.arm_a), 2),
        "delta_b_pp": round(delta(common, args.arm_b), 2),
        "DiD_pp": round(delta(common, args.arm_a) - delta(common, args.arm_b), 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "pass": lo > 0,
        "note": f"sample bootstrap {args.n_boot} seed{args.seed}, 같은 sample_id 4조건 paired",
    }
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
