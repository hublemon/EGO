"""
④ extract_memory_train.py — task_history + temporal context + future_gt_actions 추출.

v2 (2026-07-18, F0 final plan §0-4): history cutoff 를 strict 규칙으로 교정.
  - 기존(legacy) 버그: history 를 `stop_frame < current_stop_frame` 으로 잘랐으나 앵커는
    trigger_frame(= stop_frame - 1s) → 관측 이후 1초에 끝난 직전·중첩 행동이 새었다 (4.6%).
  - strict 규칙: `stop_frame < trigger_frame` (엄격 부등호) + 3개 제외 —
    ⑴ trigger 를 가로지르는 segment (start<trigger<stop)  ⑵ stop == trigger
    ⑶ start/stop timestamp 불완전(NaN) 케이스.
  - legacy 재현은 --legacy_cutoff (기존 run 재현 전용. 신규 데이터 생성 금지).

L2-c (프레임-히스토리 시간 정렬): 4-frame 샘플 시각(trigger-4.0/2.67/1.33/0.0s)과 동일한
  offset 에서 "그 시점에 진행 중이던 **완료된**(stop<trigger) 행동"을 조회해 frame 별로 정렬.
  현재 진행 중 action(GT) 누설 방지: 조회 후보를 완료 집합(stop<trigger)으로 먼저 제한 —
  legacy 의 start_frame 기준 조회보다 강한 보장.

future_gt_actions (B0 데이터 계약, next 3~5): trigger 이후 시작하는 GT 행동 K개.
  **어떤 policy prompt 에도 넣지 않는다** — B0 offline teacher(hindsight)·평가 전용 메타데이터.
  convert 단계에서 학습 jsonl 과 물리적으로 분리된 b0_meta 파일로 빠진다.

입력: selected_{split}.jsonl + EPIC_100_{split}.csv
출력: memory_{split}.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

EGO_ROOT = Path(os.path.expanduser(os.environ.get("EGO_ROOT", "~/work/jihun/EGO")))
ANNOT_DIR = EGO_ROOT / "src/epic-kitchens-100-annotations"
TRAIN_CSV = ANNOT_DIR / "EPIC_100_train.csv"
VIDEO_INFO_CSV = ANNOT_DIR / "EPIC_100_video_info.csv"
SELECTED = EGO_ROOT / "data/grpo_dataset/selected_train.jsonl"
OUT = EGO_ROOT / "data/grpo_dataset/memory_train.jsonl"

LEGACY_OFFSETS = [0.5, 1.0, 2.0]
# 4-frame 샘플 시각과 동일 (extract_frame_train.py FRAME_OFFSETS_SEC 와 일치해야 함)
FRAME_OFFSETS_SEC = [4.0, 2.67, 1.33, 0.0]
MAX_HISTORY = 10
FUTURE_K = 5


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """제외 ⑶: timestamp 불완전(NaN)·역전(stop<=start) 행 제거."""
    d = df.dropna(subset=["start_frame", "stop_frame"])
    return d[d["stop_frame"] > d["start_frame"]]


def get_task_history_strict(df: pd.DataFrame, video_id: str, trigger_frame: int,
                            max_history: int | None = MAX_HISTORY) -> list[str]:
    """strict cutoff: stop_frame < trigger_frame (엄격 부등호).
    가로지르는 segment(⑴)와 stop==trigger(⑵)는 조건식에서 자동 제외된다."""
    video_df = _valid_rows(df[df["video_id"] == video_id]).sort_values("start_frame")
    history = video_df[video_df["stop_frame"] < trigger_frame]
    labels = [f"{row['verb']} {row['noun']}" for _, row in history.iterrows()]
    if max_history is not None:
        labels = labels[-max_history:]
    return labels


def get_task_history_legacy(df: pd.DataFrame, video_id: str, current_stop_frame: int,
                            max_history: int | None = MAX_HISTORY) -> list[str]:
    """기존 run 재현 전용 (버그 보존: stop_frame < current_stop_frame)."""
    video_df = df[df["video_id"] == video_id].sort_values("start_frame")
    history = video_df[video_df["stop_frame"] < current_stop_frame]
    labels = [f"{row['verb']} {row['noun']}" for _, row in history.iterrows()]
    if max_history is not None:
        labels = labels[-max_history:]
    return labels


def get_frame_aligned_context(df: pd.DataFrame, video_id: str, trigger_frame: int,
                              fps: float, offsets_sec: list[float] = FRAME_OFFSETS_SEC,
                              ) -> dict[str, str | None]:
    """L2-c: frame 샘플 시각별로 '그 시점에 진행 중이던 완료된 행동' 조회.
    누설 방지: 후보를 먼저 stop_frame < trigger_frame(완료 집합)으로 제한 —
    현재 진행 중 GT action 은 stop >= trigger 라 구조적으로 나올 수 없다."""
    video_df = _valid_rows(df[df["video_id"] == video_id])
    completed = video_df[video_df["stop_frame"] < trigger_frame]
    result: dict[str, str | None] = {}
    for i, offset in enumerate(offsets_sec, 1):
        target = trigger_frame - int(round(offset * fps))
        key = f"frame{i}_t-{offset}s"
        if target < 0:
            result[key] = None
            continue
        matched = completed[(completed["start_frame"] <= target) &
                            (completed["stop_frame"] >= target)]
        result[key] = (f"{matched.iloc[0]['verb']} {matched.iloc[0]['noun']}"
                       if len(matched) > 0 else None)
    return result


def get_temporal_context_legacy(df: pd.DataFrame, video_id: str, ref_frame: int,
                                fps: float, offsets_sec: list[float] = LEGACY_OFFSETS,
                                ) -> dict[str, str | None]:
    """기존 Type-2 (start_frame 기준 조회) — legacy 재현 전용."""
    video_df = df[df["video_id"] == video_id]
    result: dict[str, str | None] = {}
    for offset in offsets_sec:
        target_frame = ref_frame - int(offset * fps)
        if target_frame < 0:
            result[f"t-{offset}s"] = None
            continue
        matched = video_df[(video_df["start_frame"] <= target_frame) &
                           (video_df["stop_frame"] >= target_frame)]
        result[f"t-{offset}s"] = (f"{matched.iloc[0]['verb']} {matched.iloc[0]['noun']}"
                                  if len(matched) > 0 else None)
    return result


def get_future_gt_actions(df: pd.DataFrame, video_id: str, trigger_frame: int,
                          fps: float, k: int = FUTURE_K) -> list[dict]:
    """B0 hindsight 용 next-K GT actions (start_frame >= trigger_frame, 시간순).
    현재 진행 중 action(start < trigger)은 포함되지 않는다 — gt_label 이 별도 담당.
    ⚠ policy prompt 노출 금지 — b0_meta 전용."""
    video_df = _valid_rows(df[df["video_id"] == video_id]).sort_values("start_frame")
    future = video_df[video_df["start_frame"] >= trigger_frame].head(k)
    out = []
    for offset, (_, row) in enumerate(future.iterrows(), 1):
        out.append({
            "offset": offset,
            "start_sec": round(float(row["start_frame"]) / fps, 3),
            "stop_sec": round(float(row["stop_frame"]) / fps, 3),
            "verb": str(row["verb"]),
            "noun": str(row["noun"]),
            "verb_class": int(row["verb_class"]),
            "noun_class": int(row["noun_class"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-history", type=int, default=MAX_HISTORY)
    ap.add_argument("--future-k", type=int, default=FUTURE_K)
    ap.add_argument("--split", choices=["train", "validation"], default="train",
                    help="validation: EPIC_100_validation.csv 로 history 조회 (held-out)")
    ap.add_argument("--selected", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--legacy_cutoff", action="store_true",
                    help="기존 run 재현 전용 (버그 보존). 신규 데이터 생성에 사용 금지")
    args = ap.parse_args()

    csv_path = ANNOT_DIR / f"EPIC_100_{args.split}.csv"
    selected_path = Path(args.selected) if args.selected else (
        EGO_ROOT / "data/grpo_dataset" /
        ("selected_heldout.jsonl" if args.split == "validation" else "selected_train.jsonl"))
    out_path = Path(args.out) if args.out else (
        EGO_ROOT / "data/grpo_dataset" /
        ("memory_heldout.jsonl" if args.split == "validation" else "memory_train.jsonl"))

    train_df = pd.read_csv(csv_path)
    vinfo = pd.read_csv(VIDEO_INFO_CSV).set_index("video_id")["fps"].to_dict()

    samples = [json.loads(l) for l in selected_path.read_text().splitlines() if l.strip()]
    rule = "legacy" if args.legacy_cutoff else "strict"
    print(f"[load] {len(samples)} samples in {selected_path.name} "
          f"(history csv: {csv_path.name}, cutoff={rule})")

    rows = []
    n_leak_fixed = 0          # legacy 대비 strict 가 제거한 누설 샘플 수 (자동 보고)
    for s in samples:
        vid = s["video_id"]
        start_f = int(s["start_frame"])
        stop_f = int(s["stop_frame"])
        trigger_f = int(s["trigger_frame"])
        fps = float(s.get("fps") or vinfo.get(vid, 60.0))

        if args.legacy_cutoff:
            history = get_task_history_legacy(train_df, vid, stop_f, args.max_history)
            temporal = get_temporal_context_legacy(train_df, vid, start_f, fps)
            frame_ctx: dict[str, str | None] = {}
        else:
            history = get_task_history_strict(train_df, vid, trigger_f, args.max_history)
            legacy_hist = get_task_history_legacy(train_df, vid, stop_f, args.max_history)
            if legacy_hist != history:
                n_leak_fixed += 1
            temporal = {}     # legacy Type-2 는 frame_aligned_context 로 대체
            frame_ctx = get_frame_aligned_context(train_df, vid, trigger_f, fps)

        rows.append({
            "sample_id": s["sample_id"],
            "video_id": vid,
            "cutoff_rule": rule,
            "trigger_frame": trigger_f,
            "fps": fps,
            "task_history": history,
            "temporal_proximity": temporal,               # legacy 호환 필드 (strict 에선 빈 dict)
            "frame_aligned_context": frame_ctx,           # L2-c (strict 전용)
            "frame_offsets_sec": [] if args.legacy_cutoff else FRAME_OFFSETS_SEC,
            # ⚠ policy prompt 노출 금지 — convert 가 b0_meta 파일로 분리한다
            "future_gt_actions": get_future_gt_actions(train_df, vid, trigger_f, fps,
                                                       args.future_k),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} memory contexts → {out_path}")

    hist_lens = [len(r["task_history"]) for r in rows]
    fut_lens = [len(r["future_gt_actions"]) for r in rows]
    print(f"  task_history length: min={min(hist_lens)} mean={sum(hist_lens)/len(rows):.1f} max={max(hist_lens)}")
    print(f"  future_gt_actions:   min={min(fut_lens)} mean={sum(fut_lens)/len(rows):.1f} max={max(fut_lens)}")
    if not args.legacy_cutoff:
        fa_nonnull = [sum(1 for v in r["frame_aligned_context"].values() if v) for r in rows]
        print(f"  frame_aligned non-null per sample: mean={sum(fa_nonnull)/len(rows):.1f}")
        print(f"  [leakage report] legacy 대비 strict 가 history 를 바꾼 샘플: "
              f"{n_leak_fixed}/{len(rows)} ({100*n_leak_fixed/len(rows):.1f}%) — "
              f"이 수치가 곧 수정된 누설 규모")


if __name__ == "__main__":
    main()
