"""Build an endpoint-shifted GoalStep index while preserving sample identity.

The canonical GoalStep index observes up to ``target_start - tau_a``.  This
utility produces the diagnostic protocol used by the released V-JEPA EK100
loader: observe up to ``target_end - tau_a``.  It deliberately starts from an
existing canonical index so row order, sample count, labels, scenarios, and
the action registry remain identical; only the observation timestamps change.

For duplicate step/substep annotations with the same start and label, the
canonical builder retains the first row after sorting by end time.  We mirror
that rule by joining the shortest matching target segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


KEY_DECIMALS = 6


def _read_index(index_dir: Path, split: str) -> tuple[pd.DataFrame, Path]:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = index_dir / f"{split}{suffix}"
        if path.is_file():
            return reader(path), path
    raise FileNotFoundError(f"No {split}.parquet or {split}.csv under {index_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_key(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["_time_key"] = result[time_column].astype(float).round(KEY_DECIMALS)
    result["verb_label"] = result["verb_label"].astype(int)
    result["noun_label"] = result["noun_label"].astype(int)
    return result


def build_split(
    source: pd.DataFrame,
    labels: pd.DataFrame,
    source_tau_a: float,
    tau_a: float,
    l_obs: float,
) -> tuple[pd.DataFrame, dict]:
    source = source.reset_index(drop=True).copy()
    source["source_row"] = source.index
    source["target_start_sec"] = source["obs_end_sec"].astype(float) + source_tau_a
    source = _join_key(source, "target_start_sec")

    candidates = _join_key(labels, "start_time")
    candidates = candidates.sort_values(
        ["video_uid", "_time_key", "verb_label", "noun_label", "end_time", "level"]
    ).drop_duplicates(["video_uid", "_time_key", "verb_label", "noun_label"], keep="first")
    candidates = candidates[[
        "video_uid", "_time_key", "verb_label", "noun_label", "start_time", "end_time", "level"
    ]].rename(columns={
        "start_time": "annotation_target_start_sec",
        "end_time": "target_end_sec",
        "level": "matched_level",
    })

    keys = ["video_uid", "_time_key", "verb_label", "noun_label"]
    merged = source.merge(candidates, on=keys, how="left", validate="many_to_one", sort=False)
    missing = merged["target_end_sec"].isna()
    if missing.any():
        preview = merged.loc[missing, keys + ["target_start_sec"]].head(10).to_dict("records")
        raise RuntimeError(f"Failed to match {int(missing.sum())} source rows to target end timestamps: {preview}")

    # Restore the canonical row order explicitly after the merge.
    merged = merged.sort_values("source_row").reset_index(drop=True)
    raw_obs_end = merged["target_end_sec"].astype(float) - tau_a
    # Match the released EK100 loader's negative-frame handling: an endpoint
    # before the video begins is represented by repeated frame zero rather
    # than dropping the labelled sample.  This matters for larger tau values
    # such as end-6s, while leaving the end-1s index unchanged.
    merged["anchor_boundary_flag"] = raw_obs_end < 0.0
    merged["obs_end_sec"] = raw_obs_end.clip(lower=0.0)
    unclamped_start = merged["obs_end_sec"] - l_obs
    merged["boundary_flag"] = (unclamped_start < 0.0) | merged["anchor_boundary_flag"]
    merged["obs_start_sec"] = unclamped_start.clip(lower=0.0)
    merged["observation_anchor"] = "action_end_minus_tau"

    if (merged["obs_end_sec"] < merged["obs_start_sec"]).any():
        raise RuntimeError("Endpoint index contains a negative observation window")
    if (merged["target_end_sec"] < merged["target_start_sec"]).any():
        raise RuntimeError("Annotation contains target_end before target_start")

    output_columns = [
        "video_uid", "clip_uid", "obs_start_sec", "obs_end_sec",
        "verb_label", "noun_label", "action_label", "scenario", "boundary_flag",
        "target_start_sec", "target_end_sec", "matched_level", "observation_anchor", "source_row",
    ]
    output = merged[output_columns].copy()

    # The protocol shift must not alter sample identity or supervision.
    label_columns = ["video_uid", "clip_uid", "verb_label", "noun_label", "action_label", "scenario"]
    if not output[label_columns].reset_index(drop=True).equals(source[label_columns].reset_index(drop=True)):
        raise RuntimeError("Sample identity or labels changed while shifting the observation endpoint")

    duration = output["obs_end_sec"] - output["obs_start_sec"]
    target_duration = output["target_end_sec"] - output["target_start_sec"]
    stats = {
        "samples": int(len(output)),
        "boundary_truncated": int(output["boundary_flag"].sum()),
        "anchor_clamped_to_video_start": int(merged["anchor_boundary_flag"].sum()),
        "observation_duration_min": float(duration.min()),
        "observation_duration_median": float(duration.median()),
        "observation_duration_max": float(duration.max()),
        "target_duration_min": float(target_duration.min()),
        "target_duration_median": float(target_duration.median()),
        "target_duration_max": float(target_duration.max()),
        "target_action_visible_seconds_median": float((target_duration - tau_a).clip(lower=0.0).median()),
        "target_action_visible_fraction": float((target_duration > tau_a).mean()),
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-tau-a", type=float, default=1.0)
    parser.add_argument("--tau-a", type=float, default=1.0)
    parser.add_argument("--l-obs", type=float, default=8.0)
    args = parser.parse_args()

    source_dir = Path(args.source_index_dir).resolve()
    labels_path = Path(args.labels_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_path)

    all_stats: dict[str, object] = {
        "protocol": "action_end_minus_tau",
        "source_index_dir": str(source_dir),
        "source_tau_a": args.source_tau_a,
        "tau_a": args.tau_a,
        "l_obs": args.l_obs,
        "labels_csv": str(labels_path),
        "labels_csv_sha256": _sha256(labels_path),
    }

    for split in ("train", "val"):
        source, source_path = _read_index(source_dir, split)
        split_labels = labels[labels["split"] == split].reset_index(drop=True)
        output, stats = build_split(source, split_labels, args.source_tau_a, args.tau_a, args.l_obs)
        output_path = output_dir / f"{split}.parquet"
        output.to_parquet(output_path, index=False)
        all_stats[split] = {
            **stats,
            "source_index": str(source_path),
            "source_index_sha256": _sha256(source_path),
            "output_index": str(output_path),
            "output_index_sha256": _sha256(output_path),
        }
        print(json.dumps({"split": split, **stats}, ensure_ascii=False))

    for filename in ("action_registry.json", "video_uids.txt"):
        source_path = source_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, output_dir / filename)

    stats_path = output_dir / "build_stats.json"
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
