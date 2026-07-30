"""스모크 평가 결과 요약 — arm별 지표 + trace 품질 지표 + 샘플 trace."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS = Path("/home/hogun/Project/EGO/runs/cesft_v2")


def load(arm: str):
    j = RUNS / "eval" / f"{arm}.json"
    r = RUNS / "eval" / f"{arm}.records.jsonl"
    summary = json.loads(j.read_text()) if j.is_file() else None
    recs = [json.loads(l) for l in open(r, encoding="utf-8")] if r.is_file() else []
    return summary, recs


def trace_stats(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("reasoning") and r.get("task_belief")]
    if not ok:
        return {}
    sent = lambda t: len([s for s in re.split(r"[.!?]+", t) if s.strip()])  # noqa: E731
    rs = [sent(r["reasoning"]) for r in ok]
    # task_belief이 정답 행동을 그대로 말하면 계약 위반(leak)
    leak = sum(1 for r in ok if r.get("gt") and r["gt"].lower() in r["task_belief"].lower())
    return {
        "n_with_trace": len(ok),
        "reasoning_sentences_mean": round(sum(rs) / len(rs), 2),
        "reasoning_in_3_6_sentences": round(sum(3 <= x <= 6 for x in rs) / len(rs), 3),
        "task_belief_1sent": round(sum(sent(r["task_belief"]) == 1 for r in ok) / len(ok), 3),
        "task_belief_verbatim_leak": round(leak / len(ok), 3),
        "reasoning_chars_mean": round(sum(len(r["reasoning"]) for r in ok) / len(ok)),
    }


def main() -> None:
    arms = sys.argv[1:] or ["sft_r15_local100"]
    for arm in arms:
        s, recs = load(arm)
        print("=" * 78)
        print(f"ARM: {arm}   (records={len(recs)})")
        if s:
            print(f"  n={s['n']}  SelAcc={s['acc']*100:.1f}%  malformed={s['malformed_rate']*100:.1f}%"
                  f"  cov@10={s['coverage_at_k']*100:.1f}%")
            print(f"  L0(합성 wm_scores, 실제 WM 아님)={s['L0_wm_top1']*100:.1f}%"
                  f"  G1_n={s['G1_n']} G1={s['G1_retention']*100:.1f}%"
                  f"  G2_n={s['G2_n']} G2={s['G2_correction']*100:.1f}%")
        ts = trace_stats(recs)
        if ts:
            print("  trace: " + "  ".join(f"{k}={v}" for k, v in ts.items()))
        err = [r for r in recs if r.get("error")]
        nomatch = [r for r in recs if r.get("malformed") and not r.get("error")]
        print(f"  frame/decode errors={len(err)}  parse-or-match 실패={len(nomatch)}")
        if nomatch[:2]:
            for r in nomatch[:2]:
                print(f"    - {r['sample_id']}: gt={r.get('gt')} action={r.get('action')}")

    s, recs = load(arms[0])
    print("=" * 78)
    print("샘플 trace 3건")
    for r in recs[:3]:
        if not r.get("reasoning"):
            continue
        print("-" * 78)
        print(f"[{r['sample_id']}] GT={r['gt']} | 선택={r['action']} | correct={r.get('correct')}")
        print(f"  reasoning: {r['reasoning'][:400]}")
        print(f"  task_belief: {r['task_belief']}")


if __name__ == "__main__":
    main()
