"""에피소드 전 스텝의 관측 프레임 추출 — VPA v2 와 **동일 캐시**를 공유한다.

계약(4s창 · 1s gap · 8프레임 · 336px)이 VPA v2 와 글자 그대로 같으므로
`runs/vpa_v2/frame_cache_w4_g1_n8_s336` 를 그대로 쓰고, 이미 뽑혀 있는 샘플은 건너뛴다.
캐시 디렉토리 이름에 계약이 박혀 있어(vpa.v2.frames.cache_dirname) 계약이 바뀌면
자동으로 다른 디렉토리가 되어 옛 프레임이 섞이지 않는다.

사용:
  PYTHONPATH=src python -m ego.step3_results.dynamic.extract_frames --episodes runs/dynamic_v1/episodes.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ego.step3_results.dynamic import common as C
from ego.step3_results.vpa.v2 import frames as F


def flatten(episodes: list[dict]) -> list[dict]:
    out = []
    for e in episodes:
        for s in e["steps"]:
            out.append({"sample_id": s["sample_id"], "video_uid": e["video_uid"],
                        "obs_start_sec": s["obs_start_sec"], "obs_end_sec": s["obs_end_sec"]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", default="runs/dynamic_v1/episodes.json")
    p.add_argument("--video-root", default="data/Ego4D/v2/goalstep_videos")
    p.add_argument("--cache-root", default=f"runs/vpa_v2/{F.cache_dirname()}")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out-dir", default="runs/dynamic_v1")
    args = p.parse_args()

    data = C.load_json(args.episodes)
    samples = flatten(data["episodes"])
    cache_root, video_root = Path(args.cache_root), Path(args.video_root)
    cached = {s["sample_id"] for s in samples if all(q.is_file() for q in F.frame_paths(cache_root, s))}
    todo = [s for s in samples if s["sample_id"] not in cached]
    todo.sort(key=lambda s: (s["video_uid"], s["obs_start_sec"]))  # 리더 캐시 히트 최대화
    print(f"[info] steps={len(samples)} · 이미 캐시됨={len(cached)} · 추출 대상={len(todo)}")
    print(f"[info] cache={cache_root}")

    res = Counter()
    if todo:
        def one(s):
            ok, why = F.extract_one(video_root, cache_root, s)
            return ok, why.split(":")[0]

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, (ok, why) in enumerate(ex.map(one, todo), 1):
                res[why] += 1
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {dict(res)}", flush=True)

    ready = [s for s in samples if all(q.is_file() for q in F.frame_paths(cache_root, s))]
    C.dump_json(Path(args.out_dir) / "frames_manifest.json", {
        "cache_root": str(cache_root), "episodes": args.episodes,
        "n_steps": len(samples), "n_ready": len(ready), "outcomes": dict(res),
        "contract": {"obs_window_sec": C.OBS_WINDOW_SEC, "safety_gap_sec": C.SAFETY_GAP_SEC,
                     "n_frames": C.N_FRAMES, "short_side": C.FRAME_SHORT_SIDE},
    })
    print(f"\n{dict(res)}\n프레임 준비 완료: {len(ready)}/{len(samples)} 스텝")


if __name__ == "__main__":
    main()
