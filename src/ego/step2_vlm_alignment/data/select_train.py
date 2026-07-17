"""
① select_train.py — GRPO 학습용 train 샘플 선정.

EPIC_100_train.csv 에서 다음 조건을 통과한 샘플을 random 추출:
  1) 길이 필터: stop_frame - start_frame > fps * 1.5  (최소 1.5초)
  2) trigger_frame = stop_frame - int(1.0 * fps) > 0
  3) 비디오 파일이 실제로 디스크에 존재 (현재 P01 만 보유)

추가:
  - task_goal: 같은 video_id 내 가장 첫 narration 의 "verb noun" (대리 정의)
  - canonical verb/noun (EPIC_100_{verb,noun}_classes.csv key) 동시 저장

출력: data/grpo_dataset/selected_train.jsonl

목표: 5,000 (spec). P01 한정 가용 = 약 2,721 → 가용 전체를 채택.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
ANN = EGO_ROOT / "src/epic-kitchens-100-annotations"
TRAIN_CSV = ANN / "EPIC_100_train.csv"
VALIDATION_CSV = ANN / "EPIC_100_validation.csv"
VINFO_CSV = ANN / "EPIC_100_video_info.csv"
VERB_CSV = ANN / "EPIC_100_verb_classes.csv"
NOUN_CSV = ANN / "EPIC_100_noun_classes.csv"
VIDEOS_BASE = Path(os.environ.get("EK100_VIDEOS", "data/EK100/videos"))  # 비공개 경로는 커밋하지 않는다

OUT_DIR = EGO_ROOT / "data/grpo_dataset"
OUT = OUT_DIR / "selected_train.jsonl"


def video_path(video_id: str) -> Path:
    pid = video_id.split("_")[0]
    for ext in (".MP4", ".mp4"):
        p = VIDEOS_BASE / pid / "videos" / f"{video_id}{ext}"
        if p.exists():
            return p
    return VIDEOS_BASE / pid / "videos" / f"{video_id}.MP4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "validation"], default="train",
                    help="validation: EPIC_100_validation.csv 기반 held-out 셋 (기본 출력 selected_heldout.jsonl)")
    ap.add_argument("--out", type=str, default=None, help="출력 경로 오버라이드")
    args = ap.parse_args()

    csv_path = VALIDATION_CSV if args.split == "validation" else TRAIN_CSV
    out_path = Path(args.out) if args.out else (
        OUT_DIR / ("selected_heldout.jsonl" if args.split == "validation" else "selected_train.jsonl"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    vinfo = pd.read_csv(VINFO_CSV)
    fps_map = vinfo.set_index("video_id")["fps"].to_dict()
    verb_id2key = pd.read_csv(VERB_CSV).set_index("id")["key"].to_dict()
    noun_id2key = pd.read_csv(NOUN_CSV).set_index("id")["key"].to_dict()

    n_total = len(df)
    print(f"[load] train CSV rows: {n_total}")

    # 1. fps 결합
    df["fps"] = df["video_id"].map(fps_map)
    df = df.dropna(subset=["fps"])

    # 2. 디스크 비디오 필터 (P01 만 보유)
    present_videos = set()
    for v in df["video_id"].unique():
        if video_path(v).exists():
            present_videos.add(v)
    n_videos_total = df["video_id"].nunique()
    df = df[df["video_id"].isin(present_videos)]
    print(f"[filter] videos present on disk: {len(present_videos)}/{n_videos_total} → {len(df)} rows")

    # 3. 길이 필터: stop-start > fps*1.5
    df["min_len"] = (df["fps"] * 1.5).astype(int)
    df = df[(df["stop_frame"] - df["start_frame"]) > df["min_len"]]
    print(f"[filter] length > fps*1.5: {len(df)} rows")

    # 4. trigger_frame > 0
    df["trigger_frame"] = df["stop_frame"] - (df["fps"] * 1.0).astype(int)
    df = df[df["trigger_frame"] > 0]
    print(f"[filter] trigger_frame > 0: {len(df)} rows")

    # 5. task_goal: video_id 별 첫 narration (start_frame 최소)
    first = (df.sort_values("start_frame")
               .groupby("video_id")
               .first()
               .reset_index()[["video_id", "verb", "noun"]]
               .rename(columns={"verb": "first_verb", "noun": "first_noun"}))
    df = df.merge(first, on="video_id", how="left")
    df["task_goal"] = df["first_verb"].astype(str) + " " + df["first_noun"].astype(str)

    # 6. random sample
    n_avail = len(df)
    n_take = min(args.target, n_avail)
    if n_take < args.target:
        print(f"[warn] target={args.target} > available={n_avail}, 전체 사용")
    picked = df.sample(n=n_take, random_state=args.seed).reset_index(drop=True)
    print(f"[sample] random_state={args.seed}, picked: {len(picked)}")

    # 7. trigger_timestamp (hh:mm:ss.SS)
    def ts(frame, fps):
        s = float(frame) / float(fps)
        h = int(s // 3600); m = int((s % 3600) // 60); ss = s % 60
        return f"{h:02d}:{m:02d}:{ss:05.2f}"

    rows = []
    for _, r in picked.iterrows():
        tf = int(r["trigger_frame"])
        rows.append({
            "sample_id": r["narration_id"],
            "split": args.split,
            "video_id": r["video_id"],
            "narration_id": r["narration_id"],
            "start_frame": int(r["start_frame"]),
            "stop_frame": int(r["stop_frame"]),
            "trigger_frame": tf,
            "trigger_timestamp": ts(tf, r["fps"]),
            "fps": float(r["fps"]),
            "task_goal": r["task_goal"],
            "gt_label": {
                "action": f"{verb_id2key[int(r['verb_class'])]} {noun_id2key[int(r['noun_class'])]}",
                "verb": verb_id2key[int(r["verb_class"])],
                "noun": noun_id2key[int(r["noun_class"])],
                "verb_class": int(r["verb_class"]),
                "noun_class": int(r["noun_class"]),
                "raw_verb": r["verb"],
                "raw_noun": r["noun"],
            },
        })

    with out_path.open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} samples → {out_path}")

    # 통계
    by_video = picked.groupby("video_id").size()
    print(f"  videos used: {len(by_video)}")
    print(f"  per-video min/mean/max: {by_video.min()}/{by_video.mean():.1f}/{by_video.max()}")


if __name__ == "__main__":
    main()
