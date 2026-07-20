"""Flatten Ego4D GoalStep annotations (goal -> step -> substep) into a
human-reviewable CSV. Phase 1 of the GoalStep verb/noun taxonomy effort.

This step does NOT parse verbs/nouns and does NOT download any video. It only
reads the already-downloaded GoalStep annotation JSONs and writes:

  * goalstep_annotations_flat.csv  -- one row per goal/step/substep segment,
    ORIGINAL text preserved verbatim (no parsing).
  * goalstep_annotations_sample.csv -- a seeded random sample (default 300 rows)
    for a quick eyeball pass.
  * printed summary statistics (unique videos, #steps, #substeps, total hours,
    official train/val/test/trainval split distribution).

Input files (Ego4D v2.1 annotations, benchmark=goalstep):
  * goalstep_train.json          -- 583 cooking videos, dense goal->step->substep
  * goalstep_val.json            -- 134 cooking videos, dense goal->step->substep
  * goalstep_trainval.json       -- 7219 videos, GOAL-LEVEL only (superset that
                                    contains train+val); different schema
                                    (video.annotations[], list-valued fields).
  * goalstep_test_unannotated.json -- 134 video_uids only, NO annotations.

"All annotated data" policy (per user): the CSV covers every video that carries
any annotation. Dense train/val videos contribute goal+step+substep rows; the
remaining trainval-only videos contribute a single goal row each. Overlapping
videos are taken from the dense files (richest detail) and skipped in trainval
to avoid duplicate goal rows. test_unannotated carries no annotations, so it is
reported in the split stats but produces no annotation rows.

Usage:
    python src/ego/step1_action_anticipation/goalstep/dump_goalstep_annotations.py \
        --annotations-dir data/Ego4D/v2/annotations \
        --output-dir outputs/goalstep/inspection \
        --sample-size 300 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

# Column order for the flattened CSV. Kept deliberately close to the Phase-1
# spec; source_file/parent_step_idx/duration_sec/summary added for traceability
# and to help the human pick a parsing source in Phase 2.
COLUMNS = [
    "video_uid",
    "split",
    "source_file",
    "level",            # goal | step | substep
    "parent_step_idx",  # for substeps: index of the parent step within the video
    "goal_category",
    "goal_description",
    "step_category",
    "step_description",
    "is_relevant",      # essential | optional | irrelevant (string) -- steps/substeps
    "is_procedural",
    "is_continued",
    "is_partial",       # only present in the trainval goal schema
    "start_time",
    "end_time",
    "duration_sec",
    "summary",
]


def _unwrap(value):
    """GoalStep's trainval schema stores goal_category/description/summary as
    lists; the dense train/val schema stores them as scalars. Normalise both to
    a single string without altering the original text."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(v) for v in value if v not in (None, "")]
        return " | ".join(parts)
    return str(value)


def _dur(seg: dict) -> str:
    try:
        return f"{float(seg['end_time']) - float(seg['start_time']):.3f}"
    except (KeyError, TypeError, ValueError):
        return ""


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def flatten_dense(video: dict, split: str, source_file: str) -> list[dict]:
    """Flatten a dense train/val video into goal + step + substep rows."""
    rows = []
    goal_cat = _unwrap(video.get("goal_category"))
    goal_desc = _unwrap(video.get("goal_description"))
    uid = video["video_uid"]

    rows.append({
        "video_uid": uid, "split": split, "source_file": source_file,
        "level": "goal", "parent_step_idx": "",
        "goal_category": goal_cat, "goal_description": goal_desc,
        "step_category": "", "step_description": "",
        "is_relevant": "", "is_procedural": video.get("is_procedural", ""),
        "is_continued": "", "is_partial": "",
        "start_time": video.get("start_time", ""), "end_time": video.get("end_time", ""),
        "duration_sec": _dur(video), "summary": _unwrap(video.get("summary")),
    })

    for step_idx, step in enumerate(video.get("segments", []) or []):
        rows.append({
            "video_uid": uid, "split": split, "source_file": source_file,
            "level": "step", "parent_step_idx": "",
            "goal_category": goal_cat, "goal_description": goal_desc,
            "step_category": _unwrap(step.get("step_category")),
            "step_description": _unwrap(step.get("step_description")),
            "is_relevant": _unwrap(step.get("is_relevant")),
            "is_procedural": step.get("is_procedural", ""),
            "is_continued": step.get("is_continued", ""),
            "is_partial": step.get("is_partial", ""),
            "start_time": step.get("start_time", ""), "end_time": step.get("end_time", ""),
            "duration_sec": _dur(step), "summary": _unwrap(step.get("summary")),
        })
        for sub in step.get("segments", []) or []:
            rows.append({
                "video_uid": uid, "split": split, "source_file": source_file,
                "level": "substep", "parent_step_idx": step_idx,
                "goal_category": goal_cat, "goal_description": goal_desc,
                "step_category": _unwrap(sub.get("step_category")),
                "step_description": _unwrap(sub.get("step_description")),
                "is_relevant": _unwrap(sub.get("is_relevant")),
                "is_procedural": sub.get("is_procedural", ""),
                "is_continued": sub.get("is_continued", ""),
                "is_partial": sub.get("is_partial", ""),
                "start_time": sub.get("start_time", ""), "end_time": sub.get("end_time", ""),
                "duration_sec": _dur(sub), "summary": _unwrap(sub.get("summary")),
            })
    return rows


def flatten_goal_only(video: dict, split: str, source_file: str) -> list[dict]:
    """Flatten a trainval (goal-level only) video: one row per goal annotation."""
    rows = []
    uid = video["video_uid"]
    for ann in video.get("annotations", []) or []:
        rows.append({
            "video_uid": uid, "split": split, "source_file": source_file,
            "level": "goal", "parent_step_idx": "",
            "goal_category": _unwrap(ann.get("goal_category")),
            "goal_description": _unwrap(ann.get("goal_description")),
            "step_category": "", "step_description": "",
            "is_relevant": "", "is_procedural": ann.get("is_procedural", ""),
            "is_continued": ann.get("is_continued", ""),
            "is_partial": ann.get("is_partial", ""),
            "start_time": ann.get("start_time", ""), "end_time": ann.get("end_time", ""),
            "duration_sec": _dur(ann), "summary": _unwrap(ann.get("summary")),
        })
    return rows


def build(args: argparse.Namespace) -> None:
    ann_dir = Path(args.annotations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_files = [("goalstep_train.json", "train"), ("goalstep_val.json", "val")]
    all_rows: list[dict] = []
    dense_uids: set[str] = set()

    # --- dense cooking step set: goal -> step -> substep ---
    stats = {}
    for fname, split in dense_files:
        path = ann_dir / fname
        if not path.exists():
            print(f"[warn] missing {path}, skipping")
            continue
        data = _load(path)
        vids = data["videos"]
        for v in vids:
            dense_uids.add(v["video_uid"])
            all_rows.extend(flatten_dense(v, split, fname))
        stats[split] = {"file": fname, "videos": len(vids)}

    # --- trainval goal-level superset: only videos NOT already covered ---
    tv_path = ann_dir / "goalstep_trainval.json"
    trainval_extra_videos = 0
    if tv_path.exists():
        tv = _load(tv_path)
        for v in tv["videos"]:
            if v["video_uid"] in dense_uids:
                continue  # richer dense rows already emitted for this video
            rows = flatten_goal_only(v, "trainval", "goalstep_trainval.json")
            if rows:
                trainval_extra_videos += 1
                all_rows.extend(rows)
        stats["trainval"] = {
            "file": "goalstep_trainval.json",
            "videos_total": len(tv["videos"]),
            "videos_extra_goal_only": trainval_extra_videos,
        }
    else:
        print(f"[warn] missing {tv_path}")

    # --- test_unannotated: uids only, no annotation rows ---
    test_path = ann_dir / "goalstep_test_unannotated.json"
    test_uids = 0
    if test_path.exists():
        test_uids = len(_load(test_path)["videos"])
        stats["test"] = {"file": "goalstep_test_unannotated.json",
                         "videos": test_uids, "note": "unannotated (uids only, no rows)"}

    # --- write full CSV ---
    flat_path = out_dir / "goalstep_annotations_flat.csv"
    with open(flat_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    # --- write seeded random sample ---
    rng = random.Random(args.seed)
    sample = all_rows if len(all_rows) <= args.sample_size else rng.sample(all_rows, args.sample_size)
    sample_path = out_dir / "goalstep_annotations_sample.csv"
    with open(sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sample)

    _print_summary(all_rows, stats, dense_uids, flat_path, sample_path, test_uids)


def _print_summary(all_rows, stats, dense_uids, flat_path, sample_path, test_uids):
    by_level = {}
    hours_by_level = {}
    uids = set()
    for r in all_rows:
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
        uids.add(r["video_uid"])
        try:
            hours_by_level[r["level"]] = hours_by_level.get(r["level"], 0.0) + float(r["duration_sec"]) / 3600.0
        except (TypeError, ValueError):
            pass
    split_counts = {}
    for r in all_rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    print("\n" + "=" * 68)
    print("GoalStep annotation flatten -- summary")
    print("=" * 68)
    print(f"Output CSV     : {flat_path}  ({len(all_rows)} rows)")
    print(f"Sample CSV     : {sample_path}")
    print(f"Unique videos in CSV : {len(uids)}")
    print(f"  (dense cooking step videos train+val: {len(dense_uids)})")
    print("\nRows by level:")
    for lvl in ("goal", "step", "substep"):
        h = hours_by_level.get(lvl, 0.0)
        print(f"  {lvl:8s}: {by_level.get(lvl,0):7d} rows   ~{h:8.1f} h (sum of segment durations)")
    print("\nRows by split:")
    for sp, n in sorted(split_counts.items()):
        print(f"  {sp:9s}: {n:7d} rows")
    print("\nSource-file / official-split breakdown:")
    for k, v in stats.items():
        print(f"  {k:9s}: {v}")
    # 430h cooking-step sanity check: sum of top-level STEP durations in train+val
    step_h = hours_by_level.get("step", 0.0)
    print(f"\n[430h check] sum of top-level step-segment durations = {step_h:.1f} h "
          f"(GoalStep paper: ~430 h dense cooking steps)")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-dir", default="data/Ego4D/v2/annotations",
                        help="Directory containing goalstep_*.json")
    parser.add_argument("--output-dir", default="outputs/goalstep/inspection",
                        help="Where to write the flattened + sample CSVs")
    parser.add_argument("--sample-size", type=int, default=300, help="Rows in the random sample CSV")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random sample")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
