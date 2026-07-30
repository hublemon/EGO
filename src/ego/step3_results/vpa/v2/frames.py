"""VPA v2 프레임 추출 — 4초 창에서 8프레임(2fps), 짧은 변 336.

step2(`step2_retrospection/vlm.py`)의 검증된 디코드 로직(영상별 락 · 개수/크기 검증)을
따르되 torch 의존을 뺐다(frontier arm 은 GPU 불필요).

**캐시 키에 창 규격을 포함한다.** step2 캐시는 `{uid}/{sid}/f{i}.jpg` 뿐이라 창을 8s→4s로
바꿔도 조용히 옛 프레임을 재사용하는 함정이 있었다(plan §2 B2). 여기서는 디렉토리 이름에
창·간격·프레임수·해상도를 박아 서로 다른 계약이 절대 섞이지 않게 한다.

사용:
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.frames \
      --gt runs/vpa_v2/vpa_v2_T3.json --video-root data/Ego4D/v2/goalstep_videos
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from ego.step3_results.vpa.v2 import common as C

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def cache_dirname() -> str:
    """계약이 다르면 디렉토리가 달라진다 — 캐시 교차오염 원천 차단."""
    return (f"frame_cache_w{C.OBS_WINDOW_SEC:g}_g{C.SAFETY_GAP_SEC:g}"
            f"_n{C.N_FRAMES}_s{C.FRAME_SHORT_SIDE}")


@lru_cache(maxsize=4)  # 장영상 리더는 무겁다 — step2와 동일하게 4개로 제한
def _reader(path: str):
    import decord
    return decord.VideoReader(path, num_threads=2)


def _lock(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def frame_paths(cache_root: Path, sample: dict) -> list[Path]:
    d = cache_root / sample["video_uid"] / sample["sample_id"]
    return [d / f"f{i}.jpg" for i in range(C.N_FRAMES)]


def extract_one(video_root: Path, cache_root: Path, sample: dict, overwrite: bool = False) -> tuple[bool, str]:
    """샘플 하나의 8프레임을 추출·저장. (성공여부, 사유) 반환."""
    from PIL import Image

    paths = frame_paths(cache_root, sample)
    if not overwrite and all(p.is_file() for p in paths):
        return True, "cached"

    video = video_root / f"{sample['video_uid']}.mp4"
    if not video.is_file():
        return False, "video_missing"

    t0, t1 = float(sample["obs_start_sec"]), float(sample["obs_end_sec"])
    try:
        with _lock(str(video)):
            vr = _reader(str(video))
            fps, n_total = vr.get_avg_fps(), len(vr)
            # 균등 샘플: 4초 창의 양 끝을 포함하는 8점 → 2 fps
            idxs = [min(n_total - 1, max(0, round((t0 + (t1 - t0) * i / (C.N_FRAMES - 1)) * fps)))
                    for i in range(C.N_FRAMES)]
            batch = vr.get_batch(idxs).asnumpy()
    except Exception as e:  # noqa: BLE001 — 깨진 영상은 스킵 태깅 후 계속
        return False, f"decode_error:{str(e)[:80]}"

    paths[0].parent.mkdir(parents=True, exist_ok=True)
    sizes = set()
    for i in range(batch.shape[0]):
        im = Image.fromarray(batch[i])
        w, h = im.size
        s = C.FRAME_SHORT_SIDE / min(w, h)
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)))
        sizes.add(im.size)
        im.save(paths[i], quality=90)
    if len(sizes) != 1:
        return False, "size_mismatch"
    return True, "extracted"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--video-root", default="data/Ego4D/v2/goalstep_videos")
    p.add_argument("--out-dir", default="runs/vpa_v2")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    samples = C.load_json(args.gt)
    if args.limit:
        samples = samples[: args.limit]
    video_root = Path(args.video_root)
    cache_root = Path(args.out_dir) / cache_dirname()

    # 로컬에 영상이 있는 샘플만 대상 — 없는 것은 미측정으로 남기고 오답 처리하지 않는다
    present = {f.stem for f in video_root.glob("*.mp4")}
    todo = [s for s in samples if s["video_uid"] in present]
    print(f"[info] {len(todo)}/{len(samples)} samples have local video "
          f"({len({s['video_uid'] for s in todo})}/{len({s['video_uid'] for s in samples})} videos)")
    print(f"[info] cache: {cache_root}")

    # 같은 영상 샘플을 붙여 리더 캐시 히트를 올린다
    todo.sort(key=lambda s: (s["video_uid"], s["obs_start_sec"]))
    results: dict[str, int] = {}

    def one(s):
        ok, why = extract_one(video_root, cache_root, s, args.overwrite)
        return s["sample_id"], ok, why.split(":")[0]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (sid, ok, why) in enumerate(ex.map(one, todo), 1):
            results[why] = results.get(why, 0) + 1
            if i % 50 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {results}")

    manifest = {
        "cache_dir": str(cache_root), "gt": args.gt,
        "n_requested": len(samples), "n_with_local_video": len(todo),
        "outcomes": results,
        "contract": {"obs_window_sec": C.OBS_WINDOW_SEC, "safety_gap_sec": C.SAFETY_GAP_SEC,
                     "n_frames": C.N_FRAMES, "short_side": C.FRAME_SHORT_SIDE},
    }
    C.dump_json(Path(args.out_dir) / f"frames_manifest_{Path(args.gt).stem}.json", manifest)
    print(f"\n{results}")


if __name__ == "__main__":
    main()
