"""Attach leakage-safe visual history references to the next-action index.

The target cohort and current observation are copied byte-for-byte (at the
dataframe level) from ``index_end_m1_lobs8_next_action``.  Each target row is
augmented with the most recent ``K`` *completed* same-video, same-level action
segments from the full endpoint index.  A history segment is eligible only
when its annotated action end is no later than the current observed action
``A2.start``.

Only cache identities, masks, temporal distances, and level IDs are history
model inputs.  GT verb/noun/action labels for history segments are deliberately
never written.
Columns prefixed by ``audit_`` contain boundaries solely for offline contract
checks and must not be passed to the model.

History positions are left padded and chronological (oldest -> newest).  For
example, with K=8 and three eligible segments, slots 0..4 are padding and slots
5..7 contain the selected segments in temporal order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


EPSILON_SEC = 1e-6


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


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _endpoint_with_cache_ids(endpoint: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the immutable cache identity used by next-action builder."""
    result = endpoint.reset_index(drop=True).copy()
    result["_cache_sample_id"] = [
        f"{clip_uid}_{row_position}"
        for row_position, clip_uid in enumerate(result["clip_uid"].astype(str))
    ]
    if result["_cache_sample_id"].duplicated().any():
        duplicates = result.loc[
            result["_cache_sample_id"].duplicated(keep=False), "_cache_sample_id"
        ].head(10).tolist()
        raise RuntimeError(f"Endpoint cache identities are not unique: {duplicates}")
    return result


def build_split(
    endpoint: pd.DataFrame,
    targets: pd.DataFrame,
    history_length: int = 8,
    action_registry: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the unchanged target cohort plus fixed-width visual-history refs."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")

    _require_columns(
        endpoint,
        {
            "video_uid",
            "clip_uid",
            "obs_end_sec",
            "target_start_sec",
            "target_end_sec",
            "matched_level",
        },
        "endpoint index",
    )
    _require_columns(
        targets,
        {
            "video_uid",
            "clip_uid",
            "obs_end_sec",
            "cache_sample_id",
            "observed_action_start_sec",
            "observed_action_end_sec",
            "target_start_sec",
            "target_end_sec",
            "annotation_level",
        },
        "next-action index",
    )

    endpoint = _endpoint_with_cache_ids(endpoint)
    targets = targets.reset_index(drop=True).copy()
    base_columns = list(targets.columns)
    # The trainer consumes explicit model IDs while the original raw label
    # columns remain available for cohort/audit equality checks.
    registry = action_registry or {
        "verb_classes": {str(value): value for value in targets["verb_label"].unique()},
        "noun_classes": {str(value): value for value in targets["noun_label"].unique()},
        "action_classes": {
            f"{int(row.verb_label)}|{int(row.noun_label)}": int(row.action_label)
            for row in targets.itertuples()
        },
    }
    verb_classes = registry["verb_classes"]
    noun_classes = registry["noun_classes"]
    action_classes = registry["action_classes"]
    targets["sample_id"] = targets["cache_sample_id"].astype(str)
    targets["current_cache_sample_id"] = targets["cache_sample_id"].astype(str)
    targets["verb_id"] = targets["verb_label"].map(lambda value: int(verb_classes[str(int(value))]))
    targets["noun_id"] = targets["noun_label"].map(lambda value: int(noun_classes[str(int(value))]))
    targets["action_id"] = targets.apply(
        lambda row: int(action_classes[f"{int(row['verb_label'])}|{int(row['noun_label'])}"]),
        axis=1,
    )
    targets["audit_current_observation_end_sec"] = targets["obs_end_sec"].astype(float)
    targets["audit_target_start_sec"] = targets["target_start_sec"].astype(float)
    endpoint_ids = set(endpoint["_cache_sample_id"].astype(str))

    missing_current_ids = sorted(set(targets["cache_sample_id"].astype(str)) - endpoint_ids)
    if missing_current_ids:
        raise RuntimeError(
            "Next-action index references cache IDs absent from the full endpoint index: "
            f"{missing_current_ids[:10]}"
        )

    grouped_endpoint: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in endpoint.groupby(["video_uid", "matched_level"], sort=False):
        grouped_endpoint[(str(key[0]), str(key[1]))] = group.sort_values(
            ["target_end_sec", "target_start_sec", "_cache_sample_id"], kind="stable"
        )

    records: list[dict[str, object]] = []
    history_lengths: list[int] = []
    max_history_obs_end = float("-inf")
    min_target_margin = float("inf")

    for row_number, current in targets.iterrows():
        current_id = str(current["cache_sample_id"])
        current_obs_end = float(current["obs_end_sec"])
        current_action_start = float(current["observed_action_start_sec"])
        target_start = float(current["target_start_sec"])
        level = str(current["annotation_level"])
        group = grouped_endpoint.get((str(current["video_uid"]), level))
        if group is None:
            raise RuntimeError(
                f"No endpoint group for target row {row_number}: "
                f"video={current['video_uid']} level={level}"
            )

        # ``current_id`` is excluded explicitly so that even a malformed
        # zero-duration annotation can never enter its own history.
        eligible = group[
            (group["target_end_sec"].astype(float) <= current_action_start + EPSILON_SEC)
            & (group["_cache_sample_id"].astype(str) != current_id)
        ].tail(history_length)
        history = eligible.to_dict("records")
        history_lengths.append(len(history))
        padding = history_length - len(history)

        additions: dict[str, object] = {"history_length": len(history)}
        for slot_zero_based in range(history_length):
            # Public manifest slots are 1-based to match the history trainer.
            slot = slot_zero_based + 1
            prefix = f"history_{slot}"
            history_index = slot_zero_based - padding
            if history_index < 0:
                additions[f"{prefix}_cache_sample_id"] = ""
                additions[f"{prefix}_mask"] = False
                additions[f"{prefix}_delta_t_sec"] = 0.0
                additions[f"{prefix}_level_id"] = -1
                additions[f"audit_{prefix}_action_start_sec"] = float("nan")
                additions[f"audit_{prefix}_action_end_sec"] = float("nan")
                additions[f"audit_{prefix}_obs_end_sec"] = float("nan")
                continue

            past = history[history_index]
            history_id = str(past["_cache_sample_id"])
            history_action_start = float(past["target_start_sec"])
            history_action_end = float(past["target_end_sec"])
            history_obs_end = float(past["obs_end_sec"])
            delta_sec = current_obs_end - history_obs_end
            history_level_id = {"step": 0, "substep": 1}.get(str(past["matched_level"]))
            if history_level_id is None:
                raise RuntimeError(f"Unknown history annotation level: {past['matched_level']!r}")

            if history_id not in endpoint_ids:
                raise RuntimeError(f"History cache ID does not exist: {history_id}")
            if history_action_end > current_action_start + EPSILON_SEC:
                raise RuntimeError(
                    f"History action crosses A2.start for {current_id}: "
                    f"{history_action_end} > {current_action_start}"
                )
            if history_obs_end > current_obs_end + EPSILON_SEC:
                raise RuntimeError(
                    f"History observation crosses current observation for {current_id}: "
                    f"{history_obs_end} > {current_obs_end}"
                )
            if delta_sec <= EPSILON_SEC:
                raise RuntimeError(f"Non-positive history delta for {current_id}: {delta_sec}")

            additions[f"{prefix}_cache_sample_id"] = history_id
            additions[f"{prefix}_mask"] = True
            additions[f"{prefix}_delta_t_sec"] = delta_sec
            additions[f"{prefix}_level_id"] = history_level_id
            additions[f"audit_{prefix}_action_start_sec"] = history_action_start
            additions[f"audit_{prefix}_action_end_sec"] = history_action_end
            additions[f"audit_{prefix}_obs_end_sec"] = history_obs_end
            max_history_obs_end = max(max_history_obs_end, history_obs_end)

        target_margin = target_start - current_obs_end
        if current_obs_end >= target_start - EPSILON_SEC:
            raise RuntimeError(
                f"Current observation is not strictly before A3 for {current_id}: "
                f"obs_end={current_obs_end}, target_start={target_start}"
            )
        min_target_margin = min(min_target_margin, target_margin)
        records.append(additions)

    additions_frame = pd.DataFrame(records, index=targets.index)
    output = pd.concat([targets, additions_frame], axis=1)

    # Adding the manifest must never reorder, remove, or relabel target rows.
    if len(output) != len(targets) or not output[base_columns].equals(targets[base_columns]):
        raise RuntimeError("History attachment changed the next-action target cohort")
    if output["sample_id"].duplicated().any():
        raise RuntimeError("History manifest sample_id values are not unique")

    # Model-facing history columns intentionally contain no GT labels.
    history_model_columns = [
        column for column in output.columns
        if column.startswith("history_") and not column.startswith("history_length")
    ]
    forbidden = [
        column for column in history_model_columns
        if any(token in column for token in ("verb", "noun", "action_label"))
    ]
    if forbidden:
        raise RuntimeError(f"History GT labels leaked into model columns: {forbidden}")

    length_series = pd.Series(history_lengths, dtype="int64")
    histogram = {
        str(length): int((length_series == length).sum())
        for length in range(history_length + 1)
    }
    stats: dict[str, object] = {
        "source_endpoint_samples": int(len(endpoint)),
        "target_samples": int(len(targets)),
        "retained_samples": int(len(output)),
        "history_k": int(history_length),
        "history_length_min": int(length_series.min()),
        "history_length_mean": float(length_series.mean()),
        "history_length_median": float(length_series.median()),
        "history_length_p90": float(length_series.quantile(0.9)),
        "history_length_max": int(length_series.max()),
        "history_length_histogram": histogram,
        "samples_with_no_history": histogram["0"],
        "samples_with_full_history": histogram[str(history_length)],
        "history_model_columns_contain_gt_labels": False,
        "minimum_current_obs_to_target_margin_sec": float(min_target_margin),
        "maximum_history_obs_end_sec": (
            None if max_history_obs_end == float("-inf") else float(max_history_obs_end)
        ),
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-endpoint-index-dir", required=True)
    parser.add_argument("--target-next-action-index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-length", type=int, default=8)
    args = parser.parse_args()

    endpoint_dir = Path(args.source_endpoint_index_dir).resolve()
    target_dir = Path(args.target_next_action_index_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, object] = {
        "protocol": "visual_history_of_completed_same_video_same_level_actions",
        "source_endpoint_index_dir": str(endpoint_dir),
        "target_next_action_index_dir": str(target_dir),
        "history_k": int(args.history_length),
        "history_order": "left_padded_then_oldest_to_newest",
        "history_eligibility": (
            "same video_uid and annotation level; history action end <= current A2.start"
        ),
        "history_temporal_feature": "current obs_end - history obs_end",
        "target_contract": "unchanged next-action A3 cohort from target index",
        "leakage_contract": "history visual cache IDs only; no history GT labels",
        "audit_column_contract": "audit_* boundary columns are forbidden model inputs",
        "builder_sha256": _sha256(Path(__file__).resolve()),
    }

    for split in ("train", "val"):
        endpoint, endpoint_path = _read_index(endpoint_dir, split)
        targets, target_path = _read_index(target_dir, split)
        registry_path = target_dir / "action_registry.json"
        if not registry_path.is_file():
            raise FileNotFoundError(registry_path)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        output, stats = build_split(
            endpoint, targets, history_length=args.history_length, action_registry=registry
        )
        output_path = output_dir / f"{split}.parquet"
        output.to_parquet(output_path, index=False)
        all_stats[split] = {
            **stats,
            "source_endpoint_index": str(endpoint_path),
            "source_endpoint_index_sha256": _sha256(endpoint_path),
            "target_next_action_index": str(target_path),
            "target_next_action_index_sha256": _sha256(target_path),
            "output_index": str(output_path),
            "output_index_sha256": _sha256(output_path),
        }
        print(json.dumps({"split": split, **stats}, ensure_ascii=False))

    copied_assets: dict[str, dict[str, str]] = {}
    for filename in ("action_registry.json", "video_uids.txt"):
        source_path = target_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = output_dir / filename
        shutil.copy2(source_path, destination)
        copied_assets[filename] = {
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "output": str(destination),
            "output_sha256": _sha256(destination),
        }
    all_stats["copied_assets"] = copied_assets

    stats_path = output_dir / "build_stats.json"
    stats_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
