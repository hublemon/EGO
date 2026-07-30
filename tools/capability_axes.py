#!/usr/bin/env python3
"""capability_axes.py — 추론 로그에서 '능력 축' 5종을 재집계 (GPU 불필요, 신규 추론 0).

SSOT: docs/experiments/2026-07-26_results_section_capability_axes_handoff.md

축 채택 기준(불변): **정오와 연결되지 않으면 능력으로 인정하지 않는다.**
  → 축 4·5는 조건부 SelAcc 차이/개입 지표로 검증됨. '대조 접속 표현'은 이 기준에서 기각(Δ+0.9pp).

입력: <run>/eval/{arm}.records.jsonl  (battery — action/reasoning/task_belief/correct/wm_top1_correct)
      <run>/data/context_val.jsonl    (history·candidates·gt_rank 조인용)
출력: <run>/eval/capability_axes.json + stdout 마크다운 표

사용:
  PY tools/capability_axes.py --run runs/cesft_v2_fp --arms base,cand_free,theta_ce,sft_r15
  PY tools/capability_axes.py --run runs/cesft_v2   --arms base,theta_ce,sft_r15

주의:
  - 공통 covered 교집합(gt_rank<=10 ∧ 모든 arm에 존재 ∧ non-malformed)에서만 집계한다.
    파일럿 θ_CE/sft_r15 는 covered_only=false 풀(n=5326)로 평가됐으므로 이 교집합 재집계가 필수.
  - 코호트가 다르면 **절대값 비교 금지**, 코호트 내 base 대비 Δ만 비교한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

# 근거화(이전 행동 패턴을 근거로 지목) — 2026-07-26 신규 정의. 파일럿 잣대(trace_text_metrics.py)와 별개.
RE_PATTERN = re.compile(
    r"\b(prior|previous|earlier)\s+(action|actions|step|steps|stirring|pattern)\b"
    r"|\bestablished pattern\b|\brepeatedly\b|\bpattern of (action|behavior)\b"
    r"|\bconsistent with the (prior|previous|established|ongoing)\b", re.I)
# 축 4-b 궤적 서술(narration) — 2026-07-26 추가. 완료 행동을 '사건'으로 진술한 표현.
#   RE_PATTERN(패턴 인용)과 표면은 같은 'history 언급'이지만 결정 가치가 정반대다:
#   전 arm에서 narration 은 정확도와 음의 연관(-9.7 ~ -16.4pp), citation 은 양의 연관.
RE_NARRATION = re.compile(
    r"\b(have|has|had)\s+(just\s+)?(been\s+)?(finished|completed|added|cut|placed|done)\b"
    r"|\bjust (finished|completed|added|cut|placed|done)\b", re.I)
# 기각된 축 — 보고용으로만 계산한다 (능력으로 인용 금지).
RE_CONTRAST = re.compile(r"\b(rather than|instead of|whereas|as opposed to|but not|however)\b", re.I)
# 기각된 축 2 — 인과 접속. 빈도 패턴은 우리 서사와 정확히 일치하지만(cand_free 5.4% 붕괴 →
#   sft_r15 20.5% 회복) arm 내부 정오 연관이 음수다(sft_r15 -7.7pp CI[-14.3,-0.5]).
#   "정당화의 문법"과 "판단"은 다르다는 반례로 Results에 명시적으로 싣는다.
RE_CAUSAL = re.compile(
    r"\b(since|because|having just|given that|as a result|therefore|thus|so that)\b", re.I)


def load_ctx(run: pathlib.Path) -> dict:
    ctx = {}
    with open(run / "data" / "context_val.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h = r.get("history") or []
            ctx[r["sample_id"]] = {
                "last": (f"{h[-1]['verb']} {h[-1]['noun']}" if h else None),
                "gt": f"{r['gt_verb']} {r['gt_noun']}",
                "rank": r.get("gt_rank", 99),
                "cands": r.get("candidates") or [],
            }
    return ctx


def load_arm(run: pathlib.Path, arm: str) -> dict:
    p = run / "eval" / f"{arm}.records.jsonl"
    if not p.is_file():
        return {}
    return {json.loads(l)["sample_id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def axes(rows: list[dict], ctx: dict) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    g1 = [r for r in rows if r.get("wm_top1_correct")]
    g2 = [r for r in rows if not r.get("wm_top1_correct")]
    rep = [r for r in rows if ctx[r["sample_id"]]["last"] and (r.get("action") or "") == ctx[r["sample_id"]]["last"]]
    pat = [r for r in rows if RE_PATTERN.search(r["reasoning"])]
    npat = [r for r in rows if not RE_PATTERN.search(r["reasoning"])]
    con = [r for r in rows if RE_CONTRAST.search(r["reasoning"])]
    ncon = [r for r in rows if not RE_CONTRAST.search(r["reasoning"])]
    acc = lambda xs: (sum(1 for r in xs if r.get("correct")) / len(xs)) if xs else None
    echo = sum(1 for r in rows
               if (r.get("task_belief") or "").strip().lower().rstrip(".")
               == (r.get("action") or "").strip().lower()) / n
    out = {
        "n": n,
        "sel_acc": round(acc(rows), 4),
        "ax1_g1_retention": round(acc(g1), 4) if g1 else None,
        "ax2_g2_correction": round(acc(g2), 4) if g2 else None,
        "ax3_continuation_recall": None,      # 아래에서 채움 (GT 연속 집합 필요)
        "ax3_continuation_precision": round(acc(rep), 4) if rep else None,
        "ax3_continuation_picks": len(rep),
        "ax4_grounding_rate": round(len(pat) / n, 4),
        "ax4_acc_when_grounded": round(acc(pat), 4) if pat else None,
        "ax4_acc_when_not": round(acc(npat), 4) if npat else None,
        "ax5_belief_echo": round(echo, 4),
        "rejected_contrast_rate": round(len(con) / n, 4),
        "rejected_contrast_acc_delta": None,
    }
    if pat and npat:
        out["ax4_conditional_gain_pp"] = round(100 * (acc(pat) - acc(npat)), 2)

    # 축 4-b: 서술(narration) × 패턴 인용(citation) 2×2 분할.
    #   '이득이 history 를 말한 사실 자체에서 오는가, 규칙을 뽑아낸 데서 오는가' 를 가른다.
    nar = [r for r in rows if RE_NARRATION.search(r["reasoning"])]
    cell = lambda c, n_: [r for r in rows
                          if bool(RE_PATTERN.search(r["reasoning"])) is c
                          and bool(RE_NARRATION.search(r["reasoning"])) is n_]
    q = {"neither": cell(False, False), "narration_only": cell(False, True),
         "citation_only": cell(True, False), "both": cell(True, True)}
    out["ax4b_narration_rate"] = round(len(nar) / n, 4)
    out["ax4b_narration_gain_pp"] = (
        round(100 * (acc(nar) - acc([r for r in rows if r not in nar])), 2)
        if nar and len(nar) < n else None)
    out["ax4b_cells"] = {k: {"n": len(v), "acc": round(acc(v), 4) if v else None}
                         for k, v in q.items()}
    # 서술한 trace 로 한정한 인용 효과 — 'history 를 말했다' 를 고정한 조건부 이득
    a_both, a_nar = q["both"], q["narration_only"]
    out["ax4b_citation_gain_given_narration_pp"] = (
        round(100 * (acc(a_both) - acc(a_nar)), 2) if a_both and a_nar else None)
    if len(con) >= 10 and ncon:
        out["rejected_contrast_acc_delta"] = round(100 * (acc(con) - acc(ncon)), 2)

    cau = [r for r in rows if RE_CAUSAL.search(r["reasoning"])]
    ncau = [r for r in rows if not RE_CAUSAL.search(r["reasoning"])]
    out["rejected_causal_rate"] = round(len(cau) / n, 4)
    out["rejected_causal_acc_delta"] = (round(100 * (acc(cau) - acc(ncau)), 2)
                                        if len(cau) >= 10 and ncau else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cesft_v2_fp")
    ap.add_argument("--arms", default="base,cand_free,theta_ce,sft_r15")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    ctx = load_ctx(run)
    arms = [a for a in args.arms.split(",") if a.strip()]
    data = {a: load_arm(run, a) for a in arms}
    present = [a for a in arms if data[a]]
    missing = [a for a in arms if not data[a]]
    if not present:
        raise SystemExit("no arm records found")

    common = set.intersection(*[set(data[a]) for a in present])
    common = {i for i in common if ctx.get(i, {}).get("rank", 99) <= 10}
    gt_cont = {i for i in common if ctx[i]["last"] and ctx[i]["gt"] == ctx[i]["last"]}

    res = {}
    for a in present:
        rows = [data[a][i] for i in common
                if not data[a][i].get("malformed") and data[a][i].get("reasoning")]
        m = axes(rows, ctx)
        hit = sum(1 for i in gt_cont if data[a][i].get("action") == ctx[i]["gt"])
        m["ax3_continuation_recall"] = round(hit / len(gt_cont), 4) if gt_cont else None
        res[a] = m

    payload = {"run": str(run), "n_common_covered": len(common),
               "gt_continuation_n": len(gt_cont),
               "gt_continuation_rate": round(len(gt_cont) / max(1, len(common)), 4),
               "missing_arms": missing, "per_arm": res,
               "note": "코호트 교차 시 절대값 비교 금지 — base 대비 Δ만 유효. "
                       "축 4·4b·5 문구 분할은 관찰적이며 인과 아님. "
                       "축 채택 기준: 동일 arm 내부에서 정오를 분리하지 못하면 능력으로 인정하지 않는다."}
    out = pathlib.Path(args.out) if args.out else run / "eval" / "capability_axes.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    base = res.get("base")
    print(f"\n# {run}  공통 covered n={len(common)} · GT 연속 비율 {payload['gt_continuation_rate']:.3f}")
    if missing:
        print(f"  (미도착 arm: {', '.join(missing)})")
    hdr = ["축", *present]
    rows_md = []
    LBL = [("ax1_g1_retention", "1. G1 유지"), ("ax2_g2_correction", "2. G2 교정"),
           ("ax3_continuation_recall", "3. 연속 회수율"), ("ax3_continuation_precision", "  └ 연속 정밀도"),
           ("ax4_conditional_gain_pp", "4. 근거 전환력(pp)"), ("ax4_grounding_rate", "  └ 근거화 언급률"),
           ("ax4b_citation_gain_given_narration_pp", "4b. 서술통제 인용효과(pp)"),
           ("ax4b_narration_gain_pp", "  └ 서술 자체 Δacc(pp)"), ("ax4b_narration_rate", "  └ 서술률"),
           ("ax5_belief_echo", "5. belief echo"), ("sel_acc", "(참고) SelAcc"),
           ("rejected_contrast_acc_delta", "[기각] 대조접속 Δacc(pp)"),
           ("rejected_causal_acc_delta", "[기각] 인과접속 Δacc(pp)"),
           ("rejected_causal_rate", "  └ 인과접속률")]
    for k, lbl in LBL:
        cells = []
        for a in present:
            v = res[a].get(k)
            s = "—" if v is None else f"{v:.3f}"
            if base and a != "base" and base.get(k) is not None and v is not None:
                s += f" ({v - base[k]:+.3f})"
            cells.append(s)
        rows_md.append([lbl, *cells])
    w = [max(len(str(r[i])) for r in [hdr] + rows_md) for i in range(len(hdr))]
    print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("-|-".join("-" * x for x in w))
    for r in rows_md:
        print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
