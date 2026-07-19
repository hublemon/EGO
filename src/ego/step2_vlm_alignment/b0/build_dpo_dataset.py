"""build_dpo_dataset.py — offline full-trace DPO pair 오케스트레이션 (핸드오프 §26).

입력:
  - faa_traces.jsonl : frozen FAA online full-trace (generate_faa_traces.py 산출)
  - b0meta.jsonl     : sample 별 gt_action_t + future_gt_actions + candidates + memory
                       (F0 convert 의 *_b0meta.jsonl + candidates 병합)
teacher(TeacherProtocol) 주입 — 서버는 FrozenVLMTeacher, smoke 는 mock.

각 sample 마다:
  raw = teacher.infer_raw_trace(future GT trajectory)
  projected = teacher.project_full_trace(raw, memory, candidates, gt)
  candidate support: gt ∈ D_t 아니면 drop + 집계
  for each FAA trace:
     belief_relation = teacher.equivalence(faa.belief, projected.belief)   # stop-grad
     action_relation = canonical(faa.action) == canonical(gt)
     route → KEEP/DROP → emit or audit
emit 직전 validate_record 로 누설/무결성 재검사 (핸드오프 §15).

출력: b0_dpo_{split}.jsonl (학습) + b0_audit_{split}.jsonl (SAME/SAME·UNCERTAIN/SAME).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .route_pairs import RoutingStats, action_relation, gt_in_candidates, route
from .teacher import TeacherProtocol
from .trace_utils import Trace, build_full_trace, canonical_action, parse_full_trace
from .validate_dpo_dataset import validate_record


def _dedup_valid_traces(raw_traces: list[str]) -> list[Trace]:
    """parse 실패·exact duplicate 제거 (핸드오프 §4)."""
    seen, out = set(), []
    for t in raw_traces:
        tr = parse_full_trace(t)
        if not tr.is_complete():
            continue
        key = tr.raw.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(tr)
    return out


def build_record(context_prompt, image_path, projected: Trace, faa: Trace,
                 belief_relation, act_rel, gt, faa_action, meta_extra: dict) -> dict:
    """DPO record (핸드오프 §14) + 검사 전용 _leak_check (직렬화 시 제거)."""
    chosen = projected.raw
    rejected = faa.raw
    rec = {
        "record_id": meta_extra.get("record_id", ""),
        "prompt": context_prompt,
        "image_path": image_path,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "belief_relation": belief_relation,
            "action_relation": act_rel,
            "gt_in_candidates": True,
            "projection_version": meta_extra.get("projection_version", "v1"),
            "judge_version": meta_extra.get("judge_version", "v1"),
            "faa_checkpoint_hash": meta_extra.get("faa_checkpoint_hash", ""),
        },
        # 검사 전용 — save 시 제거
        "_leak_check": {
            "gt_action": gt, "faa_action": faa_action,
            "gt_action_str": f"{gt['verb']} {gt['noun']}",
            "raw_task": meta_extra.get("raw_task"),
            "projected_belief": projected.belief,
            "faa_belief": faa.belief,
            "belief_relation": belief_relation,
            "future_gt_actions": meta_extra.get("future_gt_actions", []),
            "policy_history": meta_extra.get("policy_history", []),
            "trigger_time": meta_extra.get("trigger_time"),
            "projected_full_trace": chosen,
            "faa_full_trace": rejected,
        },
    }
    return rec


def build_pairs(samples: Iterable[dict], teacher: TeacherProtocol,
                strict_validate: bool = True) -> tuple[list[dict], list[dict], RoutingStats]:
    """samples: 각 항목은
       {sample_id, prompt, image_path, memory_context, candidates[{verb,noun}],
        gt_action{verb,noun}, future_gt_actions[...], faa_traces[str...],
        policy_history[...], trigger_time, faa_checkpoint_hash}
    반환: (train_records, audit_records, stats)."""
    train, audit = [], []
    stats = RoutingStats()
    for s in samples:
        gt = s["gt_action"]
        candidates = s.get("candidates", [])
        if not gt_in_candidates(gt["verb"], gt["noun"], candidates):
            stats.gt_outside_candidates += 1
            continue

        future = s.get("future_gt_actions", [])
        if not future:
            # 영상 말미 등 future 0개 — 빈 시퀀스로 goal 역추론은 무의미. projection 실패로 집계.
            stats.projection_failures += 1
            continue
        raw = teacher.infer_raw_trace(future)
        projected = teacher.project_full_trace(
            raw, s.get("memory_context", ""), candidates, gt["verb"], gt["noun"],
            image_path=s.get("image_path") or None)
        if projected is None or not projected.is_complete():
            stats.projection_failures += 1
            continue

        faa_traces = _dedup_valid_traces(s.get("faa_traces", []))
        for ti, faa in enumerate(faa_traces):
            b_rel = teacher.equivalence(faa.belief, projected.belief)   # stop-gradient
            a_rel = action_relation(faa.verb, faa.noun, gt["verb"], gt["noun"])
            routed = route(b_rel, a_rel)
            stats.account(routed)

            meta_extra = {
                "record_id": f'{s.get("sample_id","")}:{ti}',
                "raw_task": raw,
                "future_gt_actions": s.get("future_gt_actions", []),
                "policy_history": s.get("policy_history", []),
                "trigger_time": s.get("trigger_time"),
                "faa_checkpoint_hash": s.get("faa_checkpoint_hash", ""),
            }
            rec = build_record(s["prompt"], s.get("image_path", ""), projected, faa,
                               b_rel, a_rel, gt,
                               {"verb": faa.verb, "noun": faa.noun}, meta_extra)

            if routed.training:
                errs = validate_record(rec) if strict_validate else []
                if errs:
                    # 누설/무결성 위반은 학습에서 제외하고 audit 으로 (silent drop 금지 — 태그)
                    rec["metadata"]["training_status"] = "DROPPED_VALIDATION"
                    rec["metadata"]["validation_errors"] = errs
                    audit.append(_strip_leakcheck(rec))
                    continue
                rec["metadata"]["training_status"] = "KEEP"
                train.append(_strip_leakcheck(rec))
            else:
                rec["metadata"]["training_status"] = ("DROPPED_SAME_SAME"
                    if routed.decision == "DROP_SAME_SAME" else "DROPPED_UNCERTAIN_SAME")
                rec["metadata"]["drop_reason"] = routed.drop_reason
                audit.append(_strip_leakcheck(rec))
    return train, audit, stats


def _strip_leakcheck(rec: dict) -> dict:
    """_leak_check (검사 전용, 원본 GT/future 포함) 를 제거 — 저장 파일에 절대 남기지 않는다."""
    return {k: v for k, v in rec.items() if k != "_leak_check"}


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help="병합 입력 jsonl (faa_traces + b0meta + candidates + memory)")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_audit", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--no_strict_validate", action="store_true")
    args = ap.parse_args()

    from .teacher import build_teacher
    teacher = build_teacher(args.model_name)
    samples = _load(Path(args.samples))
    train, audit, stats = build_pairs(samples, teacher,
                                      strict_validate=not args.no_strict_validate)

    Path(args.out_train).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    Path(args.out_audit).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in audit) + "\n", encoding="utf-8")
    print(f"[done] train={len(train)} audit={len(audit)}")
    print(f"[stats] {json.dumps(stats.as_log(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
