"""
③ extract_frame_train.py — trigger 기준 frame JPEG 추출 (1-frame / 4-frame).

v2 (2026-07-18, F0 final plan): --num_frames 4 지원.
  - 샘플 시각: trigger - [4.0, 2.67, 1.33, 0.0]s  (모든 frame timestamp <= trigger — 미래 금지)
  - 4장을 2x2 grid 하나의 JPEG 로 합성해 {sid}.jpg 저장.
    합성 이유: 학습 코드(train_grpo_action.py)의 검증된 "단일 image 컬럼" 경로를
    무수정 유지하기 위해 — 프레임별 시각 라벨은 프롬프트 텍스트가 담당 (L2-c).
    배치: [1(-4.0s) 2(-2.67s) / 3(-1.33s) 4(now)], 흰색 2px 구분선.
  - manifest 에 n_frames / offsets_sec / frame_indices 기록 (3중 비교·timestamp 검증용).
  - --num_frames 1 은 기존과 동일 (1f-base 비교 기준선 유지).

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

SHORT_SIDE_1F = 768   # 1-frame: 기존과 동일
SHORT_SIDE_4F = 448   # 4-frame: 각 frame 448 → grid 약 896 (VLM 토큰 예산 고려)
GRID_GAP = 2          # frame 구분선 (흰색, px)
# extract_memory_train.py FRAME_OFFSETS_SEC 와 반드시 일치 (L2-c 정렬 계약)
FRAME_OFFSETS_SEC = [4.0, 2.67, 1.33, 0.0]


def video_path(video_id: str) -> str:
    pid = video_id.split("_")[0]
    return os.path.join(VIDEOS_BASE, pid, "videos", f"{video_id}.MP4")


def _resize_short(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    scale = short_side / min(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def save_frame(frame: np.ndarray, out_path: Path) -> None:
    img = _resize_short(Image.fromarray(frame), SHORT_SIDE_1F)
    img.save(out_path, quality=92)


def save_grid(frames: list[np.ndarray], out_path: Path) -> None:
    """4 frame → 2x2 grid. 순서: [0 1 / 2 3] = 과거→현재 (row-major)."""
    imgs = [_resize_short(Image.fromarray(f), SHORT_SIDE_4F) for f in frames]
    w = min(i.size[0] for i in imgs)
    h = min(i.size[1] for i in imgs)
    imgs = [i.resize((w, h), Image.BILINEAR) if i.size != (w, h) else i for i in imgs]
    canvas = Image.new("RGB", (w * 2 + GRID_GAP, h * 2 + GRID_GAP), (255, 255, 255))
    pos = [(0, 0), (w + GRID_GAP, 0), (0, h + GRID_GAP), (w + GRID_GAP, h + GRID_GAP)]
    for img, p in zip(imgs, pos):
        canvas.paste(img, p)
    canvas.save(out_path, quality=92)


def frame_indices(trigger: int, fps: float, nframes: int, num: int) -> list[int]:
    """샘플 인덱스. 전부 <= trigger (미래 프레임 금지), 0 이상으로 클램프."""
    if num == 1:
        return [max(0, min(trigger, nframes - 1))]
    idxs = []
    for off in FRAME_OFFSETS_SEC:
        idx = trigger - int(round(off * fps))
        idxs.append(max(0, min(idx, min(trigger, nframes - 1))))
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="이미 존재하는 jpg 는 건너뜀")
    ap.add_argument("--selected", type=str, default=str(SELECTED))
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--num_frames", type=int, default=1, choices=[1, 4],
                    help="4: trigger-4.0/2.67/1.33/0.0s 4장을 2x2 grid 합성 (F0 final plan)")
    args = ap.parse_args()
    selected_path = Path(args.selected)
    manifest_path = Path(args.manifest)

    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(l) for l in selected_path.read_text().splitlines() if l.strip()]
    if args.limit:
        samples = samples[: args.limit]

    by_video: dict[str, list] = defaultdict(list)
    for s in samples:
        by_video[s["video_id"]].append(s)

    manifest = []
    n_done = n_skip = n_err = 0
    pbar = tqdm(total=len(samples), desc=f"extract_frame(x{args.num_frames})", unit="sample")
    for vid, vsamples in by_video.items():
        todo = []
        for s in vsamples:
            sid = s["sample_id"]
            out = FRAMES_ROOT / f"{sid}.jpg"
            if args.resume and out.exists():
                n_skip += 1
                manifest.append({"sample_id": sid, "video_id": vid, "frame_path": str(out),
                                 "trigger_frame": int(s["trigger_frame"]),
                                 "n_frames": args.num_frames,
                                 "offsets_sec": FRAME_OFFSETS_SEC if args.num_frames == 4 else [0.0]})
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
        fps = float(vr.get_avg_fps()) or 60.0
        for s in todo:
            sid = s["sample_id"]
            out = FRAMES_ROOT / f"{sid}.jpg"
            try:
                trigger = int(s["trigger_frame"])
                idxs = frame_indices(trigger, fps, nframes, args.num_frames)
                assert all(i <= trigger for i in idxs), f"future frame sampled: {idxs} > {trigger}"
                if args.num_frames == 1:
                    save_frame(vr[idxs[0]].asnumpy(), out)
                else:
                    save_grid([vr[i].asnumpy() for i in idxs], out)
                manifest.append({"sample_id": sid, "video_id": vid, "frame_path": str(out),
                                 "trigger_frame": trigger, "n_frames": args.num_frames,
                                 "offsets_sec": FRAME_OFFSETS_SEC if args.num_frames == 4 else [0.0],
                                 "frame_indices": idxs, "fps": fps})
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
    print(f"[done] manifest → {manifest_path}  (n_frames={args.num_frames})")
    if n_err:
        print(f"[note] 디코딩 실패 {n_err}건 — 3중 비교 아티팩트에 제외 목록으로 보고할 것")


if __name__ == "__main__":
    main()
