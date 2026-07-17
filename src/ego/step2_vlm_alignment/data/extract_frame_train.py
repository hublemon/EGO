"""
③ extract_frame_train.py — trigger_frame JPEG 추출.

selected_train.jsonl 의 각 샘플마다 trigger_frame (= stop_frame - 1s, ②와 동일 anchor)
한 장을 추출해 data/grpo_dataset/frames/{sample_id}.jpg 로 저장.

기존 src/make_samples/extract_frame.py 와 동일 로직 (short-side 768 다운스케일).
속도 최적화: video_id 별로 묶어 VideoReader 를 한 번만 열고 batch 추출
(P01 12개 비디오 → 1348회 open → 12회 open). tqdm 진행률 + --resume 지원.

출력: data/grpo_dataset/frames/{sample_id}.jpg + frames_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
VIDEOS_BASE = os.environ.get("EK100_VIDEOS", "data/EK100/videos")  # 비공개 경로는 커밋하지 않는다
SELECTED = EGO_ROOT / "data/grpo_dataset/selected_train.jsonl"
FRAMES_ROOT = EGO_ROOT / "data/grpo_dataset/frames"
MANIFEST = EGO_ROOT / "data/grpo_dataset/frames_manifest.jsonl"

SHORT_SIDE = 768  # VLM 토큰 절약용 다운스케일


def video_path(video_id: str) -> str:
    pid = video_id.split("_")[0]
    return os.path.join(VIDEOS_BASE, pid, "videos", f"{video_id}.MP4")


def save_frame(frame: np.ndarray, out_path: Path) -> None:
    img = Image.fromarray(frame)
    w, h = img.size
    scale = SHORT_SIDE / min(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    img.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="이미 존재하는 jpg 는 건너뜀")
    ap.add_argument("--selected", type=str, default=str(SELECTED),
                    help="입력 selected jsonl (held-out: selected_heldout.jsonl)")
    ap.add_argument("--manifest", type=str, default=str(MANIFEST),
                    help="출력 manifest jsonl (frames 디렉토리는 공용 — sample_id 전역 유일)")
    args = ap.parse_args()
    selected_path = Path(args.selected)
    manifest_path = Path(args.manifest)

    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(l) for l in selected_path.read_text().splitlines() if l.strip()]
    if args.limit:
        samples = samples[: args.limit]

    # video_id 별로 묶어 VideoReader 를 한 번만 open
    by_video: dict[str, list] = defaultdict(list)
    for s in samples:
        by_video[s["video_id"]].append(s)

    manifest = []
    n_done = 0
    n_skip = 0
    n_err = 0
    pbar = tqdm(total=len(samples), desc="extract_frame", unit="frame")
    for vid, vsamples in by_video.items():
        # resume: 이미 있는 건 건너뛰되 manifest 엔 포함
        todo = []
        for s in vsamples:
            sid = s["sample_id"]
            out = FRAMES_ROOT / f"{sid}.jpg"
            if args.resume and out.exists():
                n_skip += 1
                manifest.append({"sample_id": sid, "video_id": vid,
                                 "frame_path": str(out), "trigger_frame": int(s["trigger_frame"])})
                pbar.update(1)
            else:
                todo.append(s)
        if not todo:
            continue
        try:
            vr = VideoReader(video_path(vid), num_threads=-1, ctx=cpu(0))
            nframes = len(vr)
        except Exception as e:
            n_err += len(todo)
            pbar.write(f"[err] open {vid}: {type(e).__name__}: {e}")
            pbar.update(len(todo))
            continue
        for s in todo:
            sid = s["sample_id"]
            out = FRAMES_ROOT / f"{sid}.jpg"
            try:
                idx = max(0, min(int(s["trigger_frame"]), nframes - 1))
                frame = vr[idx].asnumpy()
                save_frame(frame, out)
                manifest.append({"sample_id": sid, "video_id": vid,
                                 "frame_path": str(out), "trigger_frame": idx})
                n_done += 1
            except Exception as e:
                n_err += 1
                pbar.write(f"[err] {sid}: {type(e).__name__}: {e}")
            pbar.update(1)
            pbar.set_postfix(ok=n_done, skip=n_skip, err=n_err)
        del vr
    pbar.close()

    manifest_path.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n")
    print(f"[done] extracted: {n_done}, skipped: {n_skip}, errors: {n_err} → {FRAMES_ROOT}")
    print(f"[done] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
