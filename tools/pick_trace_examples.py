#!/usr/bin/env python3
"""pick_trace_examples.py — 논문 Table trace 용 anchor 자동 추출 (cesft_v2_fp 설계 §5 #6).

조건 (GADR 실물 + 대조군 대비):
  GT ∈ 후보 (covered) ∧ WM top-1 오답 ∧ base 오답 ∧ cand_free 오답 ∧ θ_CE 정답 ∧ sft_r15 정답
→ "WM 도 base 도 GT-CE 대조군도 틀리는데, 후보 대조 학습만 맞히는" 사례.
조건 완화 순서: cand_free 오답 조건 제거 → base 오답 조건 제거 (n 부족 시).

사용: PY tools/pick_trace_examples.py [--run runs/cesft_v2_fp] [--k 4]
출력: <run>/eval/trace_examples.md (+ .json)
"""
from __future__ import annotations

import argparse
import json
import pathlib

ARMS = ["base", "theta_ce", "sft_r15", "cand_free"]


def load_records(ev: pathlib.Path, arm: str) -> dict[str, dict]:
    p = ev / f"{arm}.records.jsonl"
    if not p.is_file():
        return {}
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2_fp")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    ev = run / "eval"
    recs = {a: load_records(ev, a) for a in ARMS}
    missing = [a for a in ARMS if not recs[a]]
    if missing:
        print(f"[warn] records 없음: {missing} — 있는 arm 만으로 진행")

    ctx = {r["sample_id"]: r for r in (json.loads(l) for l in open(run / "data" / "context_val.jsonl", encoding="utf-8") if l.strip())}

    common = set.intersection(*[set(v) for v in recs.values() if v])

    def cond(sid, need_cand_free_wrong=True, need_base_wrong=True):
        t, s = recs["theta_ce"].get(sid), recs["sft_r15"].get(sid)
        b, c = recs["base"].get(sid), recs["cand_free"].get(sid)
        if not (t and s and t.get("gt_in_support") and not t.get("wm_top1_correct")):
            return False
        if not (t.get("correct") and s.get("correct")):
            return False
        if need_base_wrong and b and b.get("correct"):
            return False
        if need_cand_free_wrong and c and c.get("correct"):
            return False
        return True

    picks = [sid for sid in sorted(common) if cond(sid)]
    relax = "strict(GADR ∧ base✗ ∧ cand_free✗)"
    if len(picks) < args.k:
        picks += [sid for sid in sorted(common) if sid not in picks and cond(sid, need_cand_free_wrong=False)]
        relax = "relaxed(cand_free 조건 제거)"
    if len(picks) < args.k:
        picks += [sid for sid in sorted(common) if sid not in picks
                  and cond(sid, need_cand_free_wrong=False, need_base_wrong=False)]
        relax = "relaxed(GADR 만)"
    picks = picks[:args.k]

    lines = [f"# Trace anchors — cesft_v2_fp (조건: {relax}, k={len(picks)})", ""]
    out_j = []
    for sid in picks:
        c = ctx.get(sid, {})
        gt = f"{c.get('gt_verb')} {c.get('gt_noun')}"
        lines += [f"## Anchor `{sid}`", f"- GT: **{gt}** · WM top-1: {recs['theta_ce'][sid].get('wm_top1')}", ""]
        entry = {"sample_id": sid, "gt": gt, "arms": {}}
        for a in ARMS:
            r = recs[a].get(sid)
            if not r:
                continue
            mark = "✓" if r.get("correct") else "✗"
            lines += [f"### {a} → {r.get('action')} {mark}",
                      f"- belief: {r.get('task_belief')}",
                      f"- reasoning: {r.get('reasoning')}", ""]
            entry["arms"][a] = {k: r.get(k) for k in ("action", "correct", "task_belief", "reasoning")}
        out_j.append(entry)

    (ev / "trace_examples.md").write_text("\n".join(lines), encoding="utf-8")
    (ev / "trace_examples.json").write_text(json.dumps(out_j, indent=1, ensure_ascii=False))
    print(f"[done] {len(picks)}건 ({relax}) -> {ev/'trace_examples.md'}")


if __name__ == "__main__":
    main()
