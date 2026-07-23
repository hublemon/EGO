"""Relabel endpoint observations with the next strictly future GoalStep action.

Input rows describe an observed action A2 and an eight-second window ending at
``A2.end - tau``.  The output keeps that exact observation/cache identity but
uses the first same-level annotation whose start is at or after ``A2.end`` as
the classification target A3.

GoalStep contains overlapping step and substep annotations.  A plain next-row
shift can therefore select an action which is already in progress (or even far
in the past) at the observation endpoint.  Requiring the same annotation level
and ``A3.start >= A2.end`` makes every retained label strictly anticipatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


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


def build_split(source: pd.DataFrame, tau_a: float) -> tuple[pd.DataFrame, dict]:
    required = {
        "video_uid", "clip_uid", "obs_start_sec", "obs_end_sec", "verb_label",
        "noun_label", "action_label", "scenario", "boundary_flag",
        "target_start_sec", "target_end_sec", "matched_level",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Endpoint index is missing required columns: {missing}")

    source = source.reset_index(drop=True).copy()
    source["cache_sample_id"] = [
        f"{clip_uid}_{row_position}"
        for row_position, clip_uid in enumerate(source["clip_uid"].astype(str))
    ]

    rows: list[dict] = []
    excluded_no_strict_next = 0
    for _, video in source.groupby("video_uid", sort=False):
        video = video.sort_values(
            ["target_start_sec", "target_end_sec", "matched_level", "cache_sample_id"],
            kind="stable",
        )
        for _, observed in video.iterrows():
            candidates = video[
                (video["matched_level"] == observed["matched_level"])
                & (video["target_start_sec"] >= float(observed["target_end_sec"]) - 1e-6)
            ]
            if candidates.empty:
                excluded_no_strict_next += 1
                continue
            target = candidates.iloc[0]
            target_horizon = float(target["target_start_sec"]) - float(observed["obs_end_sec"])
            if target_horizon < tau_a - 1e-5:
                raise RuntimeError(
                    f"Non-anticipatory target for {observed['cache_sample_id']}: horizon={target_horizon}"
                )
            rows.append({
                "video_uid": observed["video_uid"],
                "clip_uid": observed["clip_uid"],
                "obs_start_sec": float(observed["obs_start_sec"]),
                "obs_end_sec": float(observed["obs_end_sec"]),
                "verb_label": int(target["verb_label"]),
                "noun_label": int(target["noun_label"]),
                "action_label": int(target["action_label"]),
                "scenario": observed["scenario"],
                "boundary_flag": bool(observed["boundary_flag"]),
                "cache_sample_id": observed["cache_sample_id"],
                "observed_action_start_sec": float(observed["target_start_sec"]),
                "observed_action_end_sec": float(observed["target_end_sec"]),
                "observed_verb_label": int(observed["verb_label"]),
                "observed_noun_label": int(observed["noun_label"]),
                "observed_action_label": int(observed["action_label"]),
                "target_start_sec": float(target["target_start_sec"]),
                "target_end_sec": float(target["target_end_sec"]),
                "target_horizon_sec": target_horizon,
                "annotation_level": observed["matched_level"],
            })

    output = pd.DataFrame(rows)
    horizons = output["target_horizon_sec"]
    gaps = output["target_start_sec"] - output["observed_action_end_sec"]
    changed = output["action_label"] != output["observed_action_label"]
    stats = {
        "source_samples": int(len(source)),
        "retained_samples": int(len(output)),
        "excluded_no_strict_next_same_level": int(excluded_no_strict_next),
        "target_horizon_sec_min": float(horizons.min()),
        "target_horizon_sec_median": float(horizons.median()),
        "target_horizon_sec_p90": float(horizons.quantile(0.9)),
        "target_horizon_sec_max": float(horizons.max()),
        "inter_action_gap_sec_median": float(gaps.median()),
        "same_label_as_observed": int((~changed).sum()),
        "different_label_from_observed": int(changed.sum()),
        "boundary_truncated": int(output["boundary_flag"].sum()),
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau-a", type=float, default=1.0)
    parser.add_argument("--l-obs", type=float, default=8.0)
    args = parser.parse_args()

    source_dir = Path(args.source_index_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, object] = {
        "protocol": "observe_action_end_minus_tau_predict_next_strict_future_same_level",
        "source_index_dir": str(source_dir),
        "tau_a": args.tau_a,
        "l_obs": args.l_obs,
        "target_rule": "first same-level annotation with target_start >= observed_action_end",
        "cache_reuse": "cache_sample_id preserves source endpoint-index row identity",
    }

    for split in ("train", "val"):
        source, source_path = _read_index(source_dir, split)
        output, stats = build_split(source, args.tau_a)
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
        shutil.copy2(source_dir / filename, output_dir / filename)
    (output_dir / "build_stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
