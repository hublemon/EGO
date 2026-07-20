"""build_dpo_dataset_r1.py — B0-R1: GT-hidden gated teacher 로 DPO pair 재구축.

MVP(build_dpo_dataset)와의 차이 (리팩터 핸드오프 v2 정정 §1·§2·§3):
  1. chosen = teacher 가 **GT 를 못 본 채** goal(미래 suffix, target 제외)만 받고 공동 생성한
     trace 중 canonical(predicted)==canonical(GT) 인 것만 (hard action gate).
  2. goal 추출 입력에서 target action 전 등장 제외 + 누출 시 금지어 재시도 → 실패 드랍.
  3. G1/G2 별 gate 통과율(retention)을 집계 — G2/G1 ≥ 0.5 데이터 게이트 보고용.
FAA rejected 트레이스·프롬프트·routing·validate 는 MVP 와 동일 (변수 = teacher 구성뿐).

입력  : b0_samples.jsonl (MVP 산출 재사용 — FAA 롤아웃 포함)
        train_1f jsonl (G1/G2 판정용 원 topk 순서)
출력  : out_train / out_audit / out_stats(.json)
샤딩  : --shard i --num_shards n (sample 인덱스 모듈러)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_dpo_dataset import _dedup_valid_traces, _strip_leakcheck, build_record
from .route_pairs import RoutingStats, action_relation, gt_in_candidates, route
from .trace_utils import canonical_action


def _load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def sample_group(ex_topk: list[dict], gt: dict) -> str:
    """원 topk 순서 기준 G1(top1==GT)/G2(GT∈top5, top1≠GT)/OUT."""
    keys = [canonical_action(a.get("verb"), a.get("noun")) for a in ex_topk[:5]]
    g = canonical_action(gt.get("verb"), gt.get("noun"))
    if g not in keys:
        return "OUT"
    return "G1" if keys and keys[0] == g else "G2"


def build_pairs_r1(samples, teacher, topk_by_id, attempts=4, strict_validate=True,
                   log_every=25):
    from .validate_dpo_dataset import validate_record
    train, audit = [], []
    stats = RoutingStats()
    r1 = {"goal_leak_dropped": 0, "no_future_suffix": 0,
          "gate_pass": 0, "gate_fail": 0, "gate_attempts_sum": 0,
          "by_group": {g: {"seen": 0, "pass": 0} for g in ("G1", "G2", "OUT")}}
    for si, s in enumerate(samples):
        gt = s["gt_action"]
        candidates = s.get("candidates", [])
        if not gt_in_candidates(gt["verb"], gt["noun"], candidates):
            stats.gt_outside_candidates += 1
            continue
        gt_key = canonical_action(gt["verb"], gt["noun"])
        # v2 §2: goal 소스 = future 중 target 과 다른 액션만 (모든 동일-액션 등장 제거)
        suffix = [a for a in (s.get("future_gt_actions") or [])
                  if canonical_action(a.get("verb"), a.get("noun")) != gt_key]
        if not suffix:
            r1["no_future_suffix"] += 1
            continue
        goal, gmeta = teacher.extract_goal(suffix, gt["verb"], gt["noun"])
        if goal is None:
            r1["goal_leak_dropped"] += 1
            continue

        grp = sample_group(topk_by_id.get(s["sample_id"], []), gt)
        r1["by_group"][grp]["seen"] += 1
        gated, tmeta = teacher.generate_gated_trace(
            goal, s.get("memory_context", ""), candidates, gt["verb"], gt["noun"],
            image_path=s.get("image_path") or None, attempts=attempts)
        r1["gate_attempts_sum"] += tmeta["attempts_used"]
        if gated is None:
            r1["gate_fail"] += 1
            audit.append({"record_id": f'{s["sample_id"]}:GATE_FAIL', "sample_id": s["sample_id"],
                          "metadata": {"training_status": "DROPPED_GATE_FAIL", "goal": goal,
                                       "group": grp, "predictions": tmeta["predictions"],
                                       "goal_retries": gmeta["retries"]}})
            continue
        r1["gate_pass"] += 1
        r1["by_group"][grp]["pass"] += 1

        for ti, faa in enumerate(_dedup_valid_traces(s.get("faa_traces", []))):
            b_rel = teacher.equivalence(faa.belief, gated.belief)   # stop-gradient
            a_rel = action_relation(faa.verb, faa.noun, gt["verb"], gt["noun"])
            routed = route(b_rel, a_rel)
            stats.account(routed)
            meta_extra = {
                "record_id": f'{s["sample_id"]}:{ti}',
                "raw_task": goal,                     # 누설 검사 대상 = goal 문자열
                "projection_version": "r1_gated",
                "future_gt_actions": s.get("future_gt_actions", []),
                "policy_history": s.get("policy_history", []),
                "trigger_time": s.get("trigger_time"),
                "faa_checkpoint_hash": s.get("faa_checkpoint_hash", ""),
            }
            rec = build_record(s["prompt"], s.get("image_path", ""), gated, faa,
                               b_rel, a_rel, gt, {"verb": faa.verb, "noun": faa.noun}, meta_extra)
            rec["metadata"]["goal"] = goal
            rec["metadata"]["group"] = grp
            rec["metadata"]["gate_attempts"] = tmeta["attempts_used"]
            if routed.training:
                errs = validate_record(rec) if strict_validate else []
                if errs:
                    rec["metadata"]["training_status"] = "DROPPED_VALIDATION"
                    rec["metadata"]["validation_errors"] = errs
                    audit.append(_strip_leakcheck(rec))
                    continue
                rec["metadata"]["training_status"] = "KEEP"
                train.append(_strip_leakcheck(rec))
            else:
                rec["metadata"]["training_status"] = ("DROPPED_SAME_SAME"
                    if routed.decision == "DROP_SAME_SAME" else "DROPPED_UNCERTAIN_SAME")
                audit.append(_strip_leakcheck(rec))
        if (si + 1) % log_every == 0:
            print(f"  [{si+1}/{len(samples)}] pass={r1['gate_pass']} fail={r1['gate_fail']} "
                  f"pairs={len(train)}", flush=True)
    return train, audit, stats, r1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="MVP b0_samples.jsonl (FAA 롤아웃 재사용)")
    ap.add_argument("--train_jsonl", required=True, help="grpo train jsonl (G1/G2 판정용 원 topk)")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_audit", required=True)
    ap.add_argument("--out_stats", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from .teacher import build_gated_teacher
    teacher = build_gated_teacher(args.model_name)

    samples = _load(args.samples)[: args.limit or None][args.shard::args.num_shards]
    topk_by_id = {str(r.get("frame_id", "")): (r.get("topk_actions") or [])
                  for r in _load(args.train_jsonl)}
    print(f"[r1] shard {args.shard}/{args.num_shards}: {len(samples)} samples, "
          f"attempts={args.attempts}")
    train, audit, stats, r1 = build_pairs_r1(samples, teacher, topk_by_id,
                                             attempts=args.attempts)
    Path(args.out_train).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + ("\n" if train else ""),
        encoding="utf-8")
    Path(args.out_audit).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in audit) + ("\n" if audit else ""),
        encoding="utf-8")
    summary = {"routing": stats.as_log(), "r1": r1, "n_train_pairs": len(train)}
    Path(args.out_stats).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] train={len(train)} audit={len(audit)}")
    print(json.dumps(summary["r1"], ensure_ascii=False))


if __name__ == "__main__":
    main()
