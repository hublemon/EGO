"""Task 1 -- contamination check: GoalStep train vs val video_uid overlap.

VPA evaluation must run only on val videos and must never touch train videos.
This verifies the two splits are disjoint at the video level and writes
overlap_report.json. A non-empty intersection is a loud warning.

Usage:
    python scripts/vpa/check_overlap.py \
        --train-json data/Ego4D/v2/annotations/goalstep_train.json \
        --val-json   data/Ego4D/v2/annotations/goalstep_val.json \
        --output outputs/goalstep/vpa/overlap_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpa_common import dump_json, load_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-json", default="data/Ego4D/v2/annotations/goalstep_train.json")
    p.add_argument("--val-json", default="data/Ego4D/v2/annotations/goalstep_val.json")
    p.add_argument("--output", default="outputs/goalstep/vpa/overlap_report.json")
    args = p.parse_args()

    train_uids = {v["video_uid"] for v in load_json(args.train_json)["videos"]}
    val_uids = {v["video_uid"] for v in load_json(args.val_json)["videos"]}
    overlap = sorted(train_uids & val_uids)

    report = {
        "n_train_videos": len(train_uids),
        "n_val_videos": len(val_uids),
        "n_overlap": len(overlap),
        "overlap_uids": overlap,
        "clean": len(overlap) == 0,
    }
    dump_json(args.output, report)

    print(f"train videos : {len(train_uids)}")
    print(f"val videos   : {len(val_uids)}")
    print(f"overlap      : {len(overlap)}")
    if overlap:
        print("!" * 60)
        print(f"!!! WARNING: {len(overlap)} videos appear in BOTH splits !!!")
        for u in overlap[:20]:
            print("   ", u)
        print("!" * 60)
    else:
        print("OK: train/val are disjoint at the video level.")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
