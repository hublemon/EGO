"""validate_dpo_dataset.py — DPO record 무결성/누설 검사 (순수 로직, 핸드오프 §15·§20).

두 층위:
  1) leakage: policy prompt 에 GT/future/raw·projected trace/faa trace/equivalence label 이
     노출되지 않았는가. history 는 전부 trigger 이전에 끝났는가.
  2) pair invariant: chosen/rejected 가 full-trace 로 파싱되고, chosen.action==GT,
     rejected.action==FAA, 필드 splicing 이 없는가(=완결 trace 두 개), SAME/SAME 은 학습에서 빠졌는가.

model/GPU 불필요 — 문자열·구조 검사만. build_dpo_dataset 이 emit 직전에 호출하고,
독립 실행(check)으로 전체 데이터셋을 재검증할 수도 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

from .trace_utils import canonical_action, has_future_leak_language, parse_full_trace


class LeakageError(AssertionError):
    pass


class PairInvariantError(AssertionError):
    pass


def _prompt_text(record: dict) -> str:
    p = record.get("prompt", "")
    if isinstance(p, (list, dict)):
        return json.dumps(p, ensure_ascii=False)
    return str(p)


def check_prompt_leakage(record: dict) -> list[str]:
    """핸드오프 §15 leakage assertions. record 는 DPO record + 검사용 meta 를 포함할 수 있다.
    prompt 텍스트에 금지 정보가 substring 으로 나타나면 위반."""
    errs = []
    text = _prompt_text(record)
    meta = record.get("_leak_check") or {}   # build 단계가 채워주는 검사용 원본 (직렬화 안 함)
    forbidden = {
        "current_gt_action": meta.get("gt_action_str"),
        "raw_hindsight": meta.get("raw_task"),
        "projected_belief": meta.get("projected_belief"),
        "faa_belief": meta.get("faa_belief"),
        "equivalence_label": meta.get("belief_relation"),
    }
    for name, val in forbidden.items():
        if val and str(val).strip() and str(val).strip() in text:
            errs.append(f"[leak] '{name}' value present in policy prompt")
    for fut in (meta.get("future_gt_actions") or []):
        s = f"{fut.get('verb','')} {fut.get('noun','')}".strip()
        if s and s in text:
            errs.append(f"[leak] future action '{s}' present in policy prompt")
    # history stop_time <= trigger (구조 검사; build 가 numeric 을 넘겨줄 때만)
    trigger = meta.get("trigger_time")
    for h in (meta.get("policy_history") or []):
        st = h.get("stop_time")
        if trigger is not None and st is not None and st > trigger:
            errs.append(f"[leak] history action stop_time {st} > trigger {trigger}")
    return errs


def check_pair_invariants(record: dict) -> list[str]:
    """핸드오프 §15 pair invariants. chosen/rejected 완결성 + action 일치 + no-splicing."""
    errs = []
    chosen = parse_full_trace(record.get("chosen", ""))
    rejected = parse_full_trace(record.get("rejected", ""))
    meta = record.get("metadata", {}) or {}

    if not chosen.is_complete():
        errs.append("[pair] chosen is not a complete full-trace")
    if not rejected.is_complete():
        errs.append("[pair] rejected is not a complete full-trace")

    lc = record.get("_leak_check") or {}
    gt = lc.get("gt_action")     # {"verb","noun"}
    faa = lc.get("faa_action")   # {"verb","noun"}
    if gt and chosen.verb:
        if canonical_action(chosen.verb, chosen.noun) != canonical_action(gt["verb"], gt["noun"]):
            errs.append("[pair] chosen.action != canonical GT action")
    if faa and rejected.verb:
        if canonical_action(rejected.verb, rejected.noun) != canonical_action(faa["verb"], faa["noun"]):
            errs.append("[pair] rejected.action != canonical FAA action")

    # no-splicing: chosen 은 projected 원본, rejected 는 FAA 원본이어야 한다.
    #   splice(예: FAA reasoning + GT action)면 reasoning↔action 정합이 깨진다 — build 가
    #   원본 문자열을 그대로 넣었는지 해시로 확인 (원본 제공 시).
    if lc.get("projected_full_trace") is not None:
        if record.get("chosen", "").strip() != lc["projected_full_trace"].strip():
            errs.append("[pair] chosen != verbatim projected trace (splicing 의심)")
    if lc.get("faa_full_trace") is not None:
        if record.get("rejected", "").strip() != lc["faa_full_trace"].strip():
            errs.append("[pair] rejected != verbatim FAA trace (splicing 의심)")

    # SAME/SAME 은 학습 데이터에 있으면 안 된다.
    if meta.get("belief_relation") == "SAME" and meta.get("action_relation") == "SAME":
        if meta.get("training_status") != "DROPPED_SAME_SAME":
            errs.append("[pair] SAME/SAME pair present in training set (must be dropped)")

    # future leakage 스크리닝 (chosen 은 past-grounded 여야 함)
    if has_future_leak_language(chosen.reasoning):
        errs.append("[pair] chosen reasoning contains future-knowledge language")
    return errs


def validate_record(record: dict) -> list[str]:
    return check_prompt_leakage(record) + check_pair_invariants(record)


def validate_dataset_file(path: str, limit: int | None = None) -> tuple[int, list[str]]:
    """jsonl DPO 데이터셋 전체 재검증. (검사한 레코드 수, 오류 목록) 반환."""
    p = Path(path)
    errs, n = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if limit and n >= limit:
            break
        try:
            rec = json.loads(line)
        except Exception:
            errs.append(f"[parse] line {n} not valid json")
            continue
        for e in validate_record(rec):
            errs.append(f"record {n} ({rec.get('record_id','?')}): {e}")
        n += 1
    return n, errs
