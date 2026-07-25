"""train 코퍼스 서브셋 선정 — video당 상한을 두고 seed 고정 샘플링.

사용: PYTHONPATH=src python -m ego.step2_retrospection.data.make_subset --n 6000 --per_video 30
출력: runs/retro3/data/train_subset.json {"sample_ids": [...]}
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from ego.step2_retrospection.runtime import read_jsonl, runs_root, write_marker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--per_video", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(runs_root() / "data" / "context_train.jsonl")
    rng = random.Random(args.seed)
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video_uid"]].append(r["sample_id"])
    picked: list[str] = []
    for vid in sorted(by_vid):
        ids = by_vid[vid]
        rng.shuffle(ids)
        picked.extend(ids[: args.per_video])
    rng.shuffle(picked)
    picked = sorted(picked[: args.n])
    out = runs_root() / "data" / "train_subset.json"
    out.write_text(json.dumps({"seed": args.seed, "per_video": args.per_video,
                               "n": len(picked), "sample_ids": picked}))
    write_marker("S1B_SUBSET_DONE", {"n": len(picked)})
    print(f"[S1b] train subset: {len(picked)} samples ({len(by_vid)} videos)")


if __name__ == "__main__":
    main()
