"""DPO preference pair 구성 (Handoff 2 §7–§8, §11–§12).

입력: chosen_train.jsonl(pass만) × base_trace_train.jsonl(usable만)
     + semantic_train.jsonl(있으면 — 없으면 규칙 판정만, semantic="skipped")
출력: runs/retro3/data/pairs_train.jsonl + pairs_report.json (§12 필수 보고)

acceptance: action / belief / (temporal은 v1 미탐지=0) correction 중 하나 이상
rejection: 사실상 동일(belief Jaccard≥0.7 & action 동일) · malformed base ·
          semantic judge가 style_only/belief_equivalent/restates 판정.

taxonomy: BA / B / A (R·T는 v1에서 분류 안 함 — 카운트 0으로 보고).
weight: {BA:1.0, B:1.0, A:0.3}. low-confidence(판정불능)는 제거.
"""
from __future__ import annotations

import argparse
import json

from ego.step2_retrospection.hindsight import quality_gate as qg
from ego.step2_retrospection.runtime import append_jsonl, read_jsonl, runs_root, write_marker

PAIR_TYPE_WEIGHTS = {"BA": 1.0, "B": 1.0, "T": 0.8, "R": 0.6, "A": 0.3,
                     "AO": 1.0}  # action-only 증강 (2026-07-24 G3 대응)
BELIEF_EQUIV_JACCARD = 0.7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_a_ratio", type=float, default=0.34,
                    help="action-only pair 비중 상한 (§8: A 비중 제한)")
    ap.add_argument("--mode", choices=["rule", "sem", "all"], default="rule",
                    help="ablation 3-arm: all=무조건 B≻A(게이트 無) · rule=규칙 판정만 · "
                         "sem=규칙+gemini 판정 필수")
    ap.add_argument("--action_only_aug", action="store_true",
                    help="action_diff pair마다 문맥 고정 action-only(AO) pair 추가: "
                         "rejected = y+ reasoning·belief + base 오답 action. "
                         "belief로 margin을 보상하는 문체-학습 탈출구 차단 (2026-07-24 G3 대응)")
    args = ap.parse_args()

    data_dir = runs_root() / "data"
    chosen = {r["sample_id"]: r for r in read_jsonl(data_dir / "chosen_train.jsonl") if r.get("gate") == "pass"}
    base_all = read_jsonl(data_dir / "base_trace_train.jsonl")
    base = {r["sample_id"]: r for r in base_all if qg.base_trace_usable(r)}
    if args.mode == "sem":
        semantic = {r["sample_id"]: r for r in read_jsonl(data_dir / "semantic_train.jsonl")}
        if not semantic:
            write_marker("S5_PAIRS_SEM_BLOCKED", {"reason": "semantic_train.jsonl 없음 — S4 먼저"})
            raise SystemExit("[S5:sem] semantic 판정 없음 — LETSUR_API_KEY export 후 S4 재실행 필요")
    else:
        semantic = {}  # rule/all: gemini 판정 미사용 (all은 규칙 판정도 미사용)

    report = {
        "mode": args.mode,
        "total_base_traces": len(base_all),
        "usable_base_traces": len(base),
        "chosen_pass": len(chosen),
        "joined": 0, "accepted": 0,
        "rejected": {"identical": 0, "semantic_equiv": 0, "style_only": 0,
                     "restates": 0, "judge_error": 0, "a_capped": 0},
        "types": {"BA": 0, "B": 0, "A": 0, "R": 0, "T": 0, "AO": 0},
        "action_only_aug": bool(args.action_only_aug),
        "semantic_mode": "judged" if semantic else "skipped",
    }

    suffix = "" if args.mode == "rule" else f"_{args.mode}"
    out_path = data_dir / f"pairs_train{suffix}.jsonl"
    out_path.unlink(missing_ok=True)
    a_pairs, other_pairs = [], []

    for sid in sorted(set(chosen) & set(base)):
        c, b = chosen[sid], base[sid]
        report["joined"] += 1
        gt = c["gt"]
        action_diff = b["action"] != gt
        jac = qg.belief_token_jaccard(b["task_belief"], c["task_belief"])
        belief_diff = jac < BELIEF_EQUIV_JACCARD

        if args.mode == "all":
            # 무조건 B≻A: acceptance/rejection 조건 없음, weight 1.0, A-cap 없음.
            # (ablation 목적 — validated preference 없는 naive DPO. B0 문체-학습 재현 예상)
            ptype = "BA" if (action_diff and belief_diff) else ("B" if belief_diff else
                    ("A" if action_diff else "R"))  # R = 사실상 동일 (분석용 태깅)
            report["types"][ptype] += 1
            append_jsonl(out_path, {
                "sample_id": sid, "type": ptype, "weight": 1.0,
                "belief_jaccard": round(jac, 3), "semantic": "ungated",
                "chosen": qg.serialize_trace(c["reasoning"], c["task_belief"], gt),
                "rejected": qg.serialize_trace(b["reasoning"], b["task_belief"], b["action"]),
                "chosen_fields": {"reasoning": c["reasoning"], "task_belief": c["task_belief"], "action": gt},
                "rejected_fields": {"reasoning": b["reasoning"], "task_belief": b["task_belief"],
                                    "action": b["action"]},
                "gt": gt,
            })
            report["accepted"] += 1
            continue

        sem = semantic.get(sid)
        if sem is not None:
            if sem.get("error"):
                report["rejected"]["judge_error"] += 1
                continue
            if sem["belief_restates_action"] or not sem["chosen_grounded"]:
                report["rejected"]["restates"] += 1
                continue
            if sem["style_only"]:
                report["rejected"]["style_only"] += 1
                continue
            if sem["belief_equivalent"]:
                belief_diff = False
                if not action_diff:
                    report["rejected"]["semantic_equiv"] += 1
                    continue

        if not action_diff and not belief_diff:
            report["rejected"]["identical"] += 1
            continue

        ptype = "BA" if (action_diff and belief_diff) else ("B" if belief_diff else "A")
        pair = {
            "sample_id": sid, "type": ptype, "weight": PAIR_TYPE_WEIGHTS[ptype],
            "belief_jaccard": round(jac, 3), "semantic": "judged" if sem else "skipped",
            "chosen": qg.serialize_trace(c["reasoning"], c["task_belief"], gt),
            "rejected": qg.serialize_trace(b["reasoning"], b["task_belief"], b["action"]),
            "chosen_fields": {"reasoning": c["reasoning"], "task_belief": c["task_belief"], "action": gt},
            "rejected_fields": {"reasoning": b["reasoning"], "task_belief": b["task_belief"],
                                "action": b["action"]},
            "gt": gt,
        }
        (a_pairs if ptype == "A" else other_pairs).append(pair)
        # AO 증강: 문맥(y+ reasoning·belief) 고정, action만 GT vs base 오답 —
        # Δ_belief=Δ_reason≡0이라 belief로 margin을 벌 수 없는 순수 action 대비.
        if args.action_only_aug and action_diff:
            other_pairs.append({
                "sample_id": sid, "type": "AO", "weight": PAIR_TYPE_WEIGHTS["AO"],
                "belief_jaccard": 1.0, "semantic": pair["semantic"],
                "chosen": pair["chosen"],
                "rejected": qg.serialize_trace(c["reasoning"], c["task_belief"], b["action"]),
                "chosen_fields": pair["chosen_fields"],
                "rejected_fields": {"reasoning": c["reasoning"], "task_belief": c["task_belief"],
                                    "action": b["action"]},
                "gt": gt,
            })

    # A-type 비중 상한 (§8)
    max_a = int(len(other_pairs) * args.max_a_ratio / max(1e-9, 1 - args.max_a_ratio))
    if len(a_pairs) > max_a:
        report["rejected"]["a_capped"] = len(a_pairs) - max_a
        a_pairs = a_pairs[:max_a]

    if args.mode != "all":  # all 모드는 루프 안에서 직접 기록·집계 완료
        for p in other_pairs + a_pairs:
            report["types"][p["type"]] += 1
            append_jsonl(out_path, p)
        report["accepted"] = (report["types"]["BA"] + report["types"]["B"]
                              + report["types"]["A"] + report["types"]["AO"])
    report["acceptance_rate"] = round(report["accepted"] / max(1, report["joined"]), 4)

    (data_dir / f"pairs_report{suffix}.json").write_text(json.dumps(report, indent=1))
    write_marker(f"S5_PAIRS{suffix.upper()}_DONE", report)
    print(f"[S5:{args.mode}] {json.dumps(report)}")


if __name__ == "__main__":
    main()
