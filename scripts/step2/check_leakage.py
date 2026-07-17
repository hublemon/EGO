#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_leakage.py — F0 freeze 게이트용 자동 누설 검사 (F0 final plan §0-4).

세 가지를 검사한다:
  1. history cutoff: memory_{split}.jsonl 의 각 샘플에서 task_history 로 쓰인 행동이
     전부 trigger_frame 이전에 끝났는가 (strict: stop_frame < trigger_frame).
     frame_aligned_context 도 동일 검사.
  2. future 분리: grpo_train.jsonl(학습 파일)에 future_gt_actions/미래 관련 필드가 없는가.
     b0meta.jsonl 에만 존재하는가.
  3. score 누설: 학습 프롬프트 텍스트에 후보 likelihood 가 노출되지 않았는가 (train_grpo 의
     assert_no_score_leak 과 동일 취지, 데이터 레벨에서 재확인).

CSV(EPIC_100_{split}.csv)를 기준으로 memory 를 대조하므로 EK100 어노테이션이 필요하다.
학습 서버에서 실행. exit code 0 = 통과, 1 = 누설 발견.

사용:
  python scripts/step2/check_leakage.py \
    --memory data/grpo_dataset/memory_train.jsonl \
    --train_jsonl data/grpo_dataset/grpo_train.jsonl \
    --b0_meta data/grpo_dataset/grpo_train_b0meta.jsonl \
    --csv src/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --selected data/grpo_dataset/selected_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_by_id(path: Path, key: str = "sample_id") -> dict[str, dict]:
    return {r[key]: r for r in load_jsonl(path)}


def check_history_cutoff(memory_rows: list[dict], selected: dict[str, dict],
                         csv_rows_by_video: dict[str, list[dict]]) -> list[str]:
    """task_history 의 각 라벨이 trigger_frame 이전에 끝난 행동에서만 왔는지 대조."""
    errs = []
    for m in memory_rows:
        sid = m["sample_id"]
        s = selected.get(sid)
        if not s:
            continue
        trigger = int(s["trigger_frame"])
        vid = m["video_id"]
        # 그 비디오에서 stop < trigger 인 (verb noun) 라벨 집합
        legit = set()
        for r in csv_rows_by_video.get(vid, []):
            try:
                if float(r["stop_frame"]) < trigger:
                    legit.add(f"{r['verb']} {r['noun']}")
            except (ValueError, KeyError):
                continue
        for lab in m.get("task_history", []):
            if lab not in legit:
                errs.append(f"[cutoff] {sid}: history '{lab}' not in pre-trigger set "
                            f"(trigger_frame={trigger})")
        # frame_aligned_context 도 동일 검사
        for k, lab in (m.get("frame_aligned_context") or {}).items():
            if lab and lab not in legit:
                errs.append(f"[cutoff] {sid}: frame_aligned '{k}={lab}' post-trigger leak")
    return errs


def check_future_separation(train_rows: list[dict]) -> list[str]:
    """학습 파일에 future 관련 필드가 물리적으로 없어야 한다."""
    errs = []
    for i, r in enumerate(train_rows):
        for k in r:
            if "future" in k.lower():
                errs.append(f"[future] train row {i}: forbidden key '{k}'")
    return errs


def check_b0_meta_has_future(b0_rows: list[dict]) -> list[str]:
    """b0meta 는 future_gt_actions 를 가져야 한다 (분리가 제대로 됐는지 반대 방향 확인)."""
    if not b0_rows:
        return ["[b0meta] empty — future_gt_actions 분리 파일이 비었다"]
    n_with = sum(1 for r in b0_rows if r.get("future_gt_actions"))
    if n_with == 0:
        return ["[b0meta] no row has future_gt_actions — 추출 실패 의심"]
    return []


def check_score_leak(train_rows: list[dict]) -> list[str]:
    """학습 프롬프트 텍스트에 후보 score/likelihood 가 노출되지 않았는지 (재확인)."""
    errs = []
    for i, r in enumerate(train_rows):
        # convert 출력은 topk_actions_with_score 를 데이터로 갖지만 프롬프트엔 없어야 함.
        # 여기선 memory_context(프롬프트로 가는 문자열)에 소수점 likelihood 흔적만 가볍게 체크.
        mem = str(r.get("memory_context", ""))
        aws = r.get("topk_actions_with_score") or []
        for a in aws:
            lik = a.get("likelihood")
            if lik is None:
                continue
            for fmt in (f"{float(lik):.3f}", f"{float(lik):.4f}"):
                if float(fmt) != 0.0 and fmt in mem:
                    errs.append(f"[score] train row {i}: likelihood {fmt} in memory_context")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--b0_meta", default=None)
    ap.add_argument("--csv", required=True, help="EPIC_100_{split}.csv")
    ap.add_argument("--selected", required=True)
    args = ap.parse_args()

    import csv as csvmod
    memory_rows = load_jsonl(Path(args.memory))
    train_rows = load_jsonl(Path(args.train_jsonl))
    selected = load_by_id(Path(args.selected))
    b0_rows = load_jsonl(Path(args.b0_meta)) if args.b0_meta and Path(args.b0_meta).exists() else []

    csv_by_video: dict[str, list[dict]] = {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csvmod.DictReader(f):
            csv_by_video.setdefault(row["video_id"], []).append(row)

    all_errs = []
    all_errs += check_history_cutoff(memory_rows, selected, csv_by_video)
    all_errs += check_future_separation(train_rows)
    all_errs += check_b0_meta_has_future(b0_rows) if b0_rows else []
    all_errs += check_score_leak(train_rows)

    print(f"[check_leakage] memory={len(memory_rows)} train={len(train_rows)} "
          f"b0meta={len(b0_rows)} selected={len(selected)}")
    if all_errs:
        print(f"[FAIL] {len(all_errs)} leakage issue(s):")
        for e in all_errs[:50]:
            print("  " + e)
        if len(all_errs) > 50:
            print(f"  ... and {len(all_errs) - 50} more")
        sys.exit(1)
    print("[PASS] no leakage detected — freeze 게이트 통과")


if __name__ == "__main__":
    main()
