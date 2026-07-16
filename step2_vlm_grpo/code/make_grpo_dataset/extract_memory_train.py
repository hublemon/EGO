"""
④ extract_memory_train.py — task_history + temporal_proximity 추출 (train).

MEMORY_CONTEXT_SPEC.md 의 get_task_history() / get_temporal_context() 로직을
train CSV(EPIC_100_train.csv) 기준으로 동일 적용.

Type 1 (task_history): 현재 stop_frame 이전에 완료된 액션 시퀀스.
Type 2 (temporal_proximity): trigger 기준 0.5/1.0/2.0초 전 시점의 액션 라벨.
  (current action 데이터 누설 방지 위해 start_frame 기준으로 조회 — 기존 spec과 동일)

입력: selected_train.jsonl + EPIC_100_train.csv
출력: data/grpo_dataset/memory_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
ANNOT_DIR = EGO_ROOT / "src/epic-kitchens-100-annotations"
TRAIN_CSV = ANNOT_DIR / "EPIC_100_train.csv"
VIDEO_INFO_CSV = ANNOT_DIR / "EPIC_100_video_info.csv"
SELECTED = EGO_ROOT / "data/grpo_dataset/selected_train.jsonl"
OUT = EGO_ROOT / "data/grpo_dataset/memory_train.jsonl"

DEFAULT_OFFSETS = [0.5, 1.0, 2.0]
MAX_HISTORY = 10


def get_task_history(df: pd.DataFrame, video_id: str, current_stop_frame: int,
                     max_history: int | None = MAX_HISTORY) -> list[str]:
    video_df = df[df["video_id"] == video_id].sort_values("start_frame")
    history = video_df[video_df["stop_frame"] < current_stop_frame]
    labels = [f"{row['verb']} {row['noun']}" for _, row in history.iterrows()]
    if max_history is not None:
        labels = labels[-max_history:]
    return labels


def get_temporal_context(df: pd.DataFrame, video_id: str, ref_frame: int,
                         fps: float, offsets_sec: list[float] = DEFAULT_OFFSETS) -> dict[str, str | None]:
    video_df = df[df["video_id"] == video_id]
    result: dict[str, str | None] = {}
    for offset in offsets_sec:
        target_frame = ref_frame - int(offset * fps)
        if target_frame < 0:
            result[f"t-{offset}s"] = None
            continue
        matched = video_df[
            (video_df["start_frame"] <= target_frame) &
            (video_df["stop_frame"] >= target_frame)
        ]
        if len(matched) > 0:
            row = matched.iloc[0]
            result[f"t-{offset}s"] = f"{row['verb']} {row['noun']}"
        else:
            result[f"t-{offset}s"] = None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-history", type=int, default=MAX_HISTORY)
    ap.add_argument("--split", choices=["train", "validation"], default="train",
                    help="validation: EPIC_100_validation.csv 로 history 조회 (held-out)")
    ap.add_argument("--selected", type=str, default=None,
                    help="입력 selected jsonl. 미지정 시 split 에 따라 자동")
    ap.add_argument("--out", type=str, default=None,
                    help="출력 memory jsonl. 미지정 시 split 에 따라 자동")
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
    print(f"[load] {len(samples)} samples in {selected_path.name} (history csv: {csv_path.name})")

    rows = []
    for s in samples:
        vid = s["video_id"]
        start_f = int(s["start_frame"])
        stop_f = int(s["stop_frame"])
        fps = float(s.get("fps") or vinfo.get(vid, 60.0))
        history = get_task_history(train_df, vid, stop_f, max_history=args.max_history)
        # current action 누설 방지: start_frame 기준 (기존 spec과 동일)
        temporal = get_temporal_context(train_df, vid, start_f, fps)
        rows.append({
            "sample_id": s["sample_id"],
            "video_id": vid,
            "task_history": history,
            "temporal_proximity": temporal,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} memory contexts → {out_path}")

    hist_lens = [len(r["task_history"]) for r in rows]
    tp_nonnull = [sum(1 for v in r["temporal_proximity"].values() if v) for r in rows]
    print(f"  task_history length: min={min(hist_lens)} mean={sum(hist_lens)/len(rows):.1f} max={max(hist_lens)}")
    print(f"  temporal_proximity non-null per sample: min={min(tp_nonnull)} mean={sum(tp_nonnull)/len(rows):.1f} max={max(tp_nonnull)}")


if __name__ == "__main__":
    main()
