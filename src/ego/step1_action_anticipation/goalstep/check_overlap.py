"""Task 1 -- GoalStep train/val contamination check.

Verifies that ``goalstep_train.json`` and ``goalstep_val.json`` share **zero**
video_uids (hard stop if they don't), then cross-checks both splits against the
Phase-1 flattened annotation dump
(``outputs/goalstep/inspection/goalstep_annotations_flat.csv``) so that a
video present in a split JSON but missing from the CSV -- or duplicated in
either -- is caught before any index is built.

Writes ``overlap_report.json`` and prints a per-split row/video/step summary.

Usage:
    python src/ego/step1_action_anticipation/goalstep/check_overlap.py \
        --annotations-dir data/Ego4D/v2/annotations \
        --flat-csv outputs/goalstep/inspection/goalstep_annotations_flat.csv \
        --output-dir src/ego/step1_action_anticipation/goalstep/index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# parents[3] is <repo>/src (this file lives at src/ego/step1_action_anticipation/goalstep/)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from ego.common.io import ensure_dir, read_json, write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402

PHASE = "GoalStepOverlap"


def _iter_segments(node: dict):
    """Yield every (sub)segment under a GoalStep video/segment node, depth-first.

    GoalStep nests ``segments`` (steps) inside a video and ``segments``
    (substeps) inside a step; both levels carry the same field schema.
    """
    for seg in node.get("segments") or []:
        yield seg
        yield from _iter_segments(seg)


def load_split(path: Path) -> dict:
    data = read_json(path)
    videos = data["videos"]
    uids = [v["video_uid"] for v in videos]
    n_steps = sum(len(v.get("segments") or []) for v in videos)
    n_all = sum(sum(1 for _ in _iter_segments(v)) for v in videos)
    return {
        "path": str(path),
        "num_videos": len(videos),
        "num_unique_video_uids": len(set(uids)),
        "duplicate_video_uids": sorted({u for u in uids if uids.count(u) > 1}),
        "num_steps": n_steps,
        "num_steps_and_substeps": n_all,
        "video_uids": uids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-dir", default="data/Ego4D/v2/annotations")
    parser.add_argument("--flat-csv", default="outputs/goalstep/inspection/goalstep_annotations_flat.csv")
    parser.add_argument("--output-dir", default="src/ego/step1_action_anticipation/goalstep/index")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    out_dir = ensure_dir(args.output_dir)

    train = load_split(ann_dir / "goalstep_train.json")
    val = load_split(ann_dir / "goalstep_val.json")
    train_uids, val_uids = set(train["video_uids"]), set(val["video_uids"])

    for name, split in (("train", train), ("val", val)):
        step_log(
            1, PHASE,
            f"{name}: videos={split['num_videos']} unique={split['num_unique_video_uids']} "
            f"steps={split['num_steps']} steps+substeps={split['num_steps_and_substeps']}",
        )
        if split["duplicate_video_uids"]:
            step_log(1, PHASE, f"WARNING {name}: duplicate video_uids {split['duplicate_video_uids']}")

    overlap = sorted(train_uids & val_uids)
    report = {
        "train": {k: v for k, v in train.items() if k != "video_uids"},
        "val": {k: v for k, v in val.items() if k != "video_uids"},
        "overlap_video_uids": overlap,
        "num_overlap": len(overlap),
    }

    # Cross-check against the Phase-1 flattened dump (covers all 7219 annotated
    # videos, so both splits must be fully contained in it).
    flat_path = Path(args.flat_csv)
    if flat_path.is_file():
        flat = pd.read_csv(flat_path, low_memory=False)
        cross = {}
        for name, uids in (("train", train_uids), ("val", val_uids)):
            sub = flat[flat["video_uid"].isin(uids)]
            missing = sorted(uids - set(flat["video_uid"]))
            cross[name] = {
                "flat_rows": int(len(sub)),
                "flat_videos": int(sub["video_uid"].nunique()),
                "video_uids_missing_from_flat_csv": missing,
                "num_missing": len(missing),
            }
            step_log(
                1, PHASE,
                f"{name} vs flat CSV: rows={len(sub)} videos={sub['video_uid'].nunique()} "
                f"missing_from_csv={len(missing)}",
            )
            if missing:
                step_log(1, PHASE, f"WARNING {name}: {len(missing)} video_uids absent from {flat_path}")
        report["flat_csv"] = {"path": str(flat_path), "total_rows": int(len(flat)), **cross}
    else:
        step_log(1, PHASE, f"WARNING: flat CSV not found at {flat_path}; skipping cross-validation")
        report["flat_csv"] = None

    report_path = out_dir / "overlap_report.json"
    write_json(report_path, report)
    step_log(1, PHASE, f"Wrote {report_path}")

    if overlap:
        step_log(1, PHASE, f"*** CONTAMINATION: {len(overlap)} video_uids in BOTH train and val ***")
        for uid in overlap:
            print(f"  {uid}")
        raise SystemExit(1)
    step_log(1, PHASE, "OK: train/val video_uid intersection == 0")


if __name__ == "__main__":
    main()
