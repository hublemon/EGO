"""
⑥ analyze_train.py — grpo_dataset.jsonl 통계 출력.

GT in Top-5 (verb/noun/action) hit rate, rank-1 likelihood 분포,
rank-1 == GT 비율, memory context 충실도 등.

출력:
  - stdout 요약 표
  - data/grpo_dataset/stats/hit_rate.json
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
GRPO_DIR = EGO_ROOT / "data/grpo_dataset"
DATASET = GRPO_DIR / "grpo_dataset.jsonl"
STATS_DIR = GRPO_DIR / "stats"
HIT_RATE_JSON = STATS_DIR / "hit_rate.json"


def pct(n: int, d: int) -> str:
    return f"{n:>5d} / {d} ({100.0 * n / d:5.1f}%)" if d else "n/a"


def main():
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    n = len(rows)
    if n == 0:
        print("[error] empty dataset")
        return

    gt_verb = sum(r["wm_output"]["gt_in_top5_verb"] for r in rows)
    gt_noun = sum(r["wm_output"]["gt_in_top5_noun"] for r in rows)
    gt_action = sum(r["wm_output"]["gt_in_top5_action"] for r in rows)

    # rank-1 == GT
    r1_verb = sum(
        r["wm_output"]["top5_verb"][0]["verb_class"] == r["gt_label"]["verb_class"]
        for r in rows
    )
    r1_noun = sum(
        r["wm_output"]["top5_noun"][0]["noun_class"] == r["gt_label"]["noun_class"]
        for r in rows
    )
    r1_action = sum(
        r["wm_output"]["top5_action"][0]["verb_class"] == r["gt_label"]["verb_class"]
        and r["wm_output"]["top5_action"][0]["noun_class"] == r["gt_label"]["noun_class"]
        for r in rows
    )

    r1_verb_lh = [r["wm_output"]["top5_verb"][0]["likelihood"] for r in rows]
    r1_noun_lh = [r["wm_output"]["top5_noun"][0]["likelihood"] for r in rows]
    r1_action_lh = [r["wm_output"]["top5_action"][0]["likelihood"] for r in rows]

    hist_lens = [len(r["memory_context"]["task_history"]) for r in rows]
    tp_nonnull = [
        sum(1 for v in r["memory_context"]["temporal_proximity"].values() if v)
        for r in rows
    ]

    print("=== GRPO Dataset Stats ===")
    print(f"Total samples:         {n}")
    print(f"GT in Top-5 verb:      {pct(gt_verb, n)}")
    print(f"GT in Top-5 noun:      {pct(gt_noun, n)}")
    print(f"GT in Top-5 action:    {pct(gt_action, n)}")
    print(f"rank-1 == GT verb:     {pct(r1_verb, n)}")
    print(f"rank-1 == GT noun:     {pct(r1_noun, n)}")
    print(f"rank-1 == GT action:   {pct(r1_action, n)}")
    print(f"Mean rank-1 likelihood verb:   {statistics.mean(r1_verb_lh):.3f}  "
          f"(median {statistics.median(r1_verb_lh):.3f})")
    print(f"Mean rank-1 likelihood noun:   {statistics.mean(r1_noun_lh):.3f}  "
          f"(median {statistics.median(r1_noun_lh):.3f})")
    print(f"Mean rank-1 likelihood action: {statistics.mean(r1_action_lh):.3f}  "
          f"(median {statistics.median(r1_action_lh):.3f})")
    print(f"task_history len:      min={min(hist_lens)} mean={statistics.mean(hist_lens):.1f} max={max(hist_lens)}")
    print(f"temporal non-null:     min={min(tp_nonnull)} mean={statistics.mean(tp_nonnull):.1f} max={max(tp_nonnull)}")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_samples": n,
        "gt_in_top5": {
            "verb": gt_verb, "noun": gt_noun, "action": gt_action,
            "verb_pct": round(100 * gt_verb / n, 2),
            "noun_pct": round(100 * gt_noun / n, 2),
            "action_pct": round(100 * gt_action / n, 2),
        },
        "rank1_eq_gt": {
            "verb": r1_verb, "noun": r1_noun, "action": r1_action,
            "verb_pct": round(100 * r1_verb / n, 2),
            "noun_pct": round(100 * r1_noun / n, 2),
            "action_pct": round(100 * r1_action / n, 2),
        },
        "rank1_likelihood_mean": {
            "verb": round(statistics.mean(r1_verb_lh), 4),
            "noun": round(statistics.mean(r1_noun_lh), 4),
            "action": round(statistics.mean(r1_action_lh), 4),
        },
        "memory": {
            "task_history_mean_len": round(statistics.mean(hist_lens), 2),
            "temporal_nonnull_mean": round(statistics.mean(tp_nonnull), 2),
        },
    }
    HIT_RATE_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[done] → {HIT_RATE_JSON}")


if __name__ == "__main__":
    main()
