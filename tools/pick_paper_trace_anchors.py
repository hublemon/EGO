#!/usr/bin/env python3
"""pick_paper_trace_anchors.py — 논문 Results 절별 정성 trace anchor 추출 (GPU 불필요, 신규 추론 0).

SSOT: docs/paper/2026-07-26_embodied_reasoning_results.tex

arm 매핑 (논문 열 ↔ 산출물):
  Base     = runs/cesft_v2_fp/eval/base.records.jsonl
  GT-only  = runs/cesft_v2_fp/eval/cand_free.records.jsonl      (후보 미노출 · 답만 감독)
  Cand.-CE = runs/cesft_v2_fp/eval/theta_ce.records.jsonl
  EGO      = runs/cesft_v2_fp_gx/eval/sft_r15_gx.records.jsonl  (grounding 과표집 최종)

anchor 필터(논문 절 대응):
  A 판별        WM top-1 ✗ ∧ Base ✗ ∧ GT-only ✗ ∧ Cand.-CE ✓ ∧ EGO ✓   → Candidate Alignment / G2
  B 보존        WM top-1 ✓ ∧ GT-only ✗ ∧ EGO ✓                          → Capability Axes / G1
  C echo        GT-only belief==action ∧ GT-only ✗ ∧ EGO ✓              → Useful Evidence / echo
  D 근거 유용성 GT-only 근거언급∧✗ ∧ EGO 근거언급∧✓                     → Useful Evidence, Not More
  E 이력 인과   Cand.-CE 이력有 ✓ ∧ 이력無 ✗ ∧ Base ✗                    → Causal Test
  F 반례        WM top-1 ✗ ∧ Cand.-CE ✓ ∧ EGO ✗                         → What Is and Is Not Established

주의:
  - 공통 covered 교집합(gt_rank<=10 ∧ 4 arm 모두 존재 ∧ 4 arm non-malformed)에서만 뽑는다.
  - *_nohist 는 Base·GT-only·Cand.-CE·+Replay 에만 존재한다. 최종 EGO 에는 없으므로
    E 의 인과 주장은 EGO 로 전이하지 않는다(논문 본문과 동일한 범위 제한).
  - 각 anchor 는 모집단 크기를 함께 출력한다 — 인용은 필터의 출력이지 체리피킹이 아님을 보이기 위함.

사용: PY tools/pick_paper_trace_anchors.py [--out docs/paper/2026-07-26_paper_trace_anchors]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

# 근거화 정규식 — tools/capability_axes.py 와 동일 정의를 유지할 것.
RE_PATTERN = re.compile(
    r"\b(prior|previous|earlier)\s+(action|actions|step|steps|stirring|pattern)\b"
    r"|\bestablished pattern\b|\brepeatedly\b|\bpattern of (action|behavior)\b"
    r"|\bconsistent with the (prior|previous|established|ongoing)\b", re.I)

ARMS = {
    "base":       "runs/cesft_v2_fp/eval/base.records.jsonl",
    "cand_free":  "runs/cesft_v2_fp/eval/cand_free.records.jsonl",
    "theta_ce":   "runs/cesft_v2_fp/eval/theta_ce.records.jsonl",
    "sft_r15_gx": "runs/cesft_v2_fp_gx/eval/sft_r15_gx.records.jsonl",
}
NOHIST = {
    "base":      "runs/cesft_v2_fp/eval/base_nohist.records.jsonl",
    "cand_free": "runs/cesft_v2_fp/eval/cand_free_nohist.records.jsonl",
    "theta_ce":  "runs/cesft_v2_fp/eval/theta_ce_nohist.records.jsonl",
}
CTX = "runs/cesft_v2_fp_gx/data/context_val.jsonl"
LABEL = {"base": "Base", "cand_free": "GT-only", "theta_ce": "Cand.-CE", "sft_r15_gx": "EGO"}


def load(p: pathlib.Path) -> dict:
    if not p.is_file():
        return {}
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/paper/2026-07-26_paper_trace_anchors")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    recs = {a: load(root / p) for a, p in ARMS.items()}
    nohist = {a: load(root / p) for a, p in NOHIST.items()}
    ctx = {r["sample_id"]: r for r in
           (json.loads(l) for l in open(root / CTX, encoding="utf-8") if l.strip())}
    for a in ARMS:
        if not recs[a]:
            raise SystemExit(f"[fatal] records 없음: {a} -> {ARMS[a]}")

    common = set.intersection(*[set(v) for v in recs.values()])
    pool = sorted(s for s in common
                  if ctx.get(s, {}).get("gt_rank", 99) <= 10
                  and all(not recs[a][s].get("malformed") for a in recs))
    print(f"[pool] common={len(common)} covered∧non-malformed={len(pool)}")

    def ok(s, a):    return bool(recs[a][s].get("correct"))
    def echo(s, a):  return ((recs[a][s].get("task_belief") or "").strip().lower().rstrip(".")
                             == (recs[a][s].get("action") or "").strip().lower())
    def gnd(s, a):   return bool(RE_PATTERN.search(recs[a][s].get("reasoning") or ""))
    def wm_ok(s):    return bool(recs["theta_ce"][s].get("wm_top1_correct"))

    FILTERS = {
        "A_discriminate": ("WM top-1 ✗ ∧ Base ✗ ∧ GT-only ✗ ∧ Cand.-CE ✓ ∧ EGO ✓",
                           lambda s: (not wm_ok(s) and not ok(s, "base") and not ok(s, "cand_free")
                                      and ok(s, "theta_ce") and ok(s, "sft_r15_gx"))),
        "B_retain":       ("WM top-1 ✓ ∧ GT-only ✗ ∧ EGO ✓",
                           lambda s: wm_ok(s) and not ok(s, "cand_free") and ok(s, "sft_r15_gx")),
        "B_retain_strict": ("B ∧ Base ✗ ∧ Cand.-CE ✓",
                            lambda s: (wm_ok(s) and not ok(s, "cand_free") and ok(s, "sft_r15_gx")
                                       and not ok(s, "base") and ok(s, "theta_ce"))),
        "C_echo":         ("GT-only belief==action ∧ GT-only ✗ ∧ EGO ✓",
                           lambda s: echo(s, "cand_free") and not ok(s, "cand_free") and ok(s, "sft_r15_gx")),
        "D_utility":      ("GT-only 근거언급 ∧ ✗ · EGO 근거언급 ∧ ✓",
                           lambda s: (gnd(s, "cand_free") and not ok(s, "cand_free")
                                      and gnd(s, "sft_r15_gx") and ok(s, "sft_r15_gx"))),
        "E_history":      ("Cand.-CE 이력有 ✓ ∧ 이력無 ✗ ∧ Base ✗",
                           lambda s: (ok(s, "theta_ce") and s in nohist["theta_ce"]
                                      and not nohist["theta_ce"][s].get("correct")
                                      and not nohist["theta_ce"][s].get("malformed")
                                      and not ok(s, "base"))),
        "F_counter":      ("WM top-1 ✗ ∧ Cand.-CE ✓ ∧ EGO ✗",
                           lambda s: not wm_ok(s) and ok(s, "theta_ce") and not ok(s, "sft_r15_gx")),
    }

    def brief(sid: str) -> dict:
        c = ctx[sid]
        d = {"sample_id": sid, "video_uid": c["video_uid"], "scenario": c.get("scenario"),
             "gt": f"{c['gt_verb']} {c['gt_noun']}", "gt_rank": c.get("gt_rank"),
             "obs_sec": [c.get("obs_start_sec"), c.get("obs_end_sec")], "target_sec": c.get("target_start_sec"),
             "candidates": c.get("candidates"),
             "history": [f"{h['verb']} {h['noun']}" for h in (c.get("history") or [])],
             "wm_top1": recs["theta_ce"][sid].get("wm_top1"),
             "wm_top1_correct": recs["theta_ce"][sid].get("wm_top1_correct"), "arms": {}}
        for a in ARMS:
            r = recs[a][sid]
            e = {"label": LABEL[a], "action": r.get("action"), "correct": r.get("correct"),
                 "belief": r.get("task_belief"), "reasoning": r.get("reasoning"),
                 "belief_echo": echo(sid, a), "grounding_hit": gnd(sid, a),
                 "reasoning_words": len((r.get("reasoning") or "").split())}
            if sid in nohist.get(a, {}):
                n = nohist[a][sid]
                e["strip"] = {"action": n.get("action"), "correct": n.get("correct"),
                              "belief": n.get("task_belief"), "reasoning": n.get("reasoning"),
                              "reasoning_words": len((n.get("reasoning") or "").split())}
            d["arms"][a] = e
        return d

    out = {"pool_n": len(pool), "arms": ARMS, "context": CTX, "buckets": {}}
    md = ["# 논문 Results 정성 trace anchors — Base / GT-only / Cand.-CE / EGO", "",
          f"공통 covered ∧ non-malformed 모집단 **n={len(pool)}** · 신규 추론 0", ""]
    for key, (desc, fn) in FILTERS.items():
        hits = [s for s in pool if fn(s)]
        print(f"[{key}] n={len(hits)}  ({desc})")
        out["buckets"][key] = {"filter": desc, "n": len(hits),
                               "sample_ids": hits, "examples": [brief(s) for s in hits[:5]]}
        md += [f"## {key} — n={len(hits)}", f"필터: `{desc}`", ""]
        for s in hits[:3]:
            b = brief(s)
            md += [f"### `{s}` · GT **{b['gt']}** (WM {b['gt_rank']}위) · WM top-1 `{b['wm_top1']}`"
                   f" · 이력 {len(b['history'])}",
                   f"- 후보: {', '.join(b['candidates'])}",
                   f"- 이력(최근 6): {' → '.join(b['history'][-6:]) or '(없음)'}", ""]
            for a in ARMS:
                e = b["arms"][a]
                flags = "".join([" [echo]" if e["belief_echo"] else "",
                                 " [근거언급]" if e["grounding_hit"] else ""])
                md += [f"**{e['label']}** → `{e['action']}` {'✓' if e['correct'] else '✗'}{flags}"
                       f" ({e['reasoning_words']}단어)",
                       f"- belief: {e['belief']}", f"- reasoning: {e['reasoning']}"]
                if "strip" in e:
                    t = e["strip"]
                    md += [f"- [이력 제거] → `{t['action']}` {'✓' if t['correct'] else '✗'}"
                           f" ({t['reasoning_words']}단어) · belief: {t['belief']}",
                           f"- [이력 제거] reasoning: {t['reasoning']}"]
                md += [""]

    base = root / args.out
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    base.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] -> {base.with_suffix('.json')} / {base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
