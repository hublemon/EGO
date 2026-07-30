"""frame_cache 사전 생성 (PyAV) — vlm._load_cached가 읽는 경로/규칙 그대로.

vlm.extract_frames(decord)와 동일한 프레임 선택을 재현한다:
  idx[i] = clamp(round((t0 + (t1-t0)*i/(n-1)) * fps), 0, n_total-1)  → 시각 idx/fps 로 seek
  짧은 변 336으로 축소(확대 안 함)
decord는 aarch64 휠이 없어 설치 불가 → 캐시 히트로 디코드 경로를 우회한다.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import av
from PIL import Image

REPO = Path("/home/hogun/Project/EGO")
VIDEO_DIR = REPO / "data/Ego4D/v2/goalstep_videos"
CACHE = REPO / "runs/cesft_v2/frame_cache"
N_FRAMES = 8
SHORT_SIDE = 336


def save(im: Image.Image, sid: str, vid: str, slot: int) -> None:
    w, h = im.size
    s = SHORT_SIDE / min(w, h)
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)))
    d = CACHE / vid / sid
    d.mkdir(parents=True, exist_ok=True)
    im.save(d / f"f{slot}.jpg", quality=92)


def main() -> None:
    recs = [json.loads(l) for l in open(REPO / "runs/cesft_v2/data/context_val.jsonl",
                                        encoding="utf-8") if l.strip()]
    by_vid = defaultdict(list)
    for r in recs:
        by_vid[r["video_uid"]].append(r)

    n_ok = n_fail = 0
    for vid, rows in sorted(by_vid.items()):
        path = str(VIDEO_DIR / f"{vid}.mp4")
        with av.open(path) as c:
            st = c.streams.video[0]
            fps = float(st.average_rate)
            n_total = st.frames or int(float(st.duration * st.time_base) * fps)
            print(f"{vid}: fps={fps:.3f} frames={n_total} samples={len(rows)}", flush=True)

            want: dict[float, list] = defaultdict(list)
            for r in rows:
                t0 = max(0.0, r["obs_start_sec"])
                t1 = max(t0 + 1e-3, r["obs_end_sec"])
                for i in range(N_FRAMES):
                    t = t0 + (t1 - t0) * i / max(1, N_FRAMES - 1)
                    idx = min(n_total - 1, max(0, round(t * fps)))
                    want[idx / fps].append((r["sample_id"], i))

            for tsec in sorted(want):
                try:
                    c.seek(int(tsec / st.time_base), stream=st, backward=True, any_frame=False)
                    frame = None
                    for f in c.decode(st):
                        frame = f
                        if float(f.pts * st.time_base) >= tsec - 1e-3:
                            break
                    if frame is None:
                        raise RuntimeError("no frame decoded")
                    im = frame.to_image()
                except Exception as e:  # noqa: BLE001
                    print(f"[FAIL] {vid} @{tsec:.3f}s: {e}", file=sys.stderr)
                    n_fail += len(want[tsec])
                    continue
                for sid, slot in want[tsec]:
                    save(im, sid, vid, slot)
                    n_ok += 1

    complete = 0
    for r in recs:
        d = CACHE / r["video_uid"] / r["sample_id"]
        ps = [d / f"f{i}.jpg" for i in range(N_FRAMES)]
        if all(p.is_file() for p in ps) and len({Image.open(p).size for p in ps}) == 1:
            complete += 1
    print(f"frames written={n_ok} failed={n_fail} | complete samples={complete}/{len(recs)}")


if __name__ == "__main__":
    main()
