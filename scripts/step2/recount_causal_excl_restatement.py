#!/usr/bin/env python3
"""recount_causal_excl_restatement.py — ③ 를 복창 제외 서브셋에서 재집계한다.

배경 (2026-07-21 야간 결과): reward 를 gt→wm 으로 바꾸자 ③ 가 0.0135→0.0255 로 올랐지만
belief_restatement_rate 도 0.0191→0.0722 로 3.8배 올랐다. belief 가 action 을 그대로
복창하면 belief 를 바꿀 때 action 이 따라 바뀌는 것은 당연하므로, ③ 단독으로는
"인과적 사용"과 "형식적 복사"를 구분하지 못한다.

이 스크립트는 **재학습·재생성 없이** 기존 산출물만으로 그 구분을 만든다:
  - eval  records (`completion` 포함) → 샘플별 복창 여부
  - swap  records (`cond`/`changed`)  → 샘플별 control/swap 쌍

복창 판정은 eval_battery.py:148 의 정의를 그대로 쓴다 (pred verb·noun 이 belief 문자열에
둘 다 등장). 정의를 바꾸면 기존 belief_restatement_rate 와 대조가 깨진다.

출력: 전체 / 복창 제외 / 복창만 세 서브셋의 ③ + paired bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness_v2 import bootstrap_ci, bootstrap_paired_diff  # noqa: E402

BELIEF_RE = re.compile(r"<task_belief>(.*?)</task_belief>", re.DOTALL)


def restatement_flags(eval_records_path: str) -> dict[str, bool | None]:
    """sample_id → 복창 여부. belief 가 없거나 파싱 실패면 None (분류 불가)."""
    flags: dict[str, bool | None] = {}
    for line in open(eval_records_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        mb = BELIEF_RE.search(r.get("completion") or "")
        belief = mb.group(1).strip() if mb else ""
        v, n = r.get("pred_verb"), r.get("pred_noun")
        if not belief or v is None or n is None:
            flags[r["sample_id"]] = None          # belief 부재/파싱 실패 → 분모에서 제외
        else:
            flags[r["sample_id"]] = (v in belief) and (n in belief)
    return flags


def paired_from_swap(swap_records_path: str) -> dict[str, tuple[float, float]]:
    """sample_id → (control_changed, swap_changed). 두 cond 가 다 있는 샘플만."""
    by: dict[str, dict[str, bool]] = {}
    for line in open(swap_records_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault(r["sample_id"], {})[r["cond"]] = bool(r["changed"])
    return {sid: (float(d["control"]), float(d["swap"]))
            for sid, d in by.items() if "control" in d and "swap" in d}


def block(pairs: list[tuple[float, float]], n_boot: int, seed: int):
    if not pairs:
        return {"n": 0, "note": "빈 서브셋"}
    ctrl = [a for a, _ in pairs]
    swp = [b for _, b in pairs]
    return {
        "n": len(pairs),
        "control_action_change": round(sum(ctrl) / len(ctrl), 4),
        "swap_action_change": round(sum(swp) / len(swp), 4),
        # 반올림 전 값의 차 — eval_belief_swap 의 '먼저 반올림 후 뺄셈' 은 정밀도를 잃는다
        "causal_sensitivity": round(sum(swp) / len(swp) - sum(ctrl) / len(ctrl), 4),
        "causal_sensitivity_ci95": bootstrap_paired_diff(pairs, n_boot, seed),
        "control_ci95": bootstrap_ci(ctrl, n_boot, seed),
        "swap_ci95": bootstrap_ci(swp, n_boot, seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_records", required=True, help="eval_*.records.jsonl (completion 포함)")
    ap.add_argument("--swap_records", required=True, help="swap_*.records.jsonl")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    flags = restatement_flags(args.eval_records)
    pairs = paired_from_swap(args.swap_records)

    subsets: dict[str, list[tuple[float, float]]] = {"all": [], "excl_restatement": [],
                                                     "restatement_only": []}
    unclassified = 0
    for sid, pr in pairs.items():
        f = flags.get(sid)
        subsets["all"].append(pr)
        if f is None:
            unclassified += 1
        elif f:
            subsets["restatement_only"].append(pr)
        else:
            subsets["excl_restatement"].append(pr)

    classified = len(pairs) - unclassified
    out = {
        "eval_records": args.eval_records,
        "swap_records": args.swap_records,
        "n_paired": len(pairs),
        "n_unclassified": unclassified,
        "restatement_rate_on_paired": (round(len(subsets["restatement_only"]) / classified, 4)
                                       if classified else None),
        "subsets": {k: block(v, args.n_boot, args.seed) for k, v in subsets.items()},
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {p}")


if __name__ == "__main__":
    main()
