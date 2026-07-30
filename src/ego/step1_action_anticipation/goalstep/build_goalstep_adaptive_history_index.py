"""Attach K=8 visual history to the adaptive-transition A1 -> A2 cohort.

Each adaptive index row already has the complete causal contract:

* cached/current observation: A1 through ``A1.end - 0.25s``;
* supervised target: the close, immediate same-level successor A2; and
* cache identity: ``<clip_uid>_<row_position>``.

This builder reuses the leakage checks and fixed-width history schema from
``build_goalstep_history_index``.  For history membership, however, an
adaptive row is interpreted as its *observed* action A1 rather than as its
target A2.  A past cached action is eligible only when it is from the same
video and annotation level and its observed action has completed no later
than the current A1 start.

Only cache identities, masks, temporal deltas, and level IDs are exposed to
the model.  Past GT class labels are never copied into history columns.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from ego.step1_action_anticipation.goalstep.build_goalstep_history_index import (
    _read_index,
    _sha256,
    build_split,
)


REQUIRED_COLUMNS = {
    "video_uid",
    "clip_uid",
    "obs_end_sec",
    "verb_label",
    "noun_label",
    "action_label",
    "annotation_level",
    "observed_action_start_sec",
    "observed_action_end_sec",
    "target_start_sec",
    "target_end_sec",
}


def _cache_sample_ids(frame: pd.DataFrame) -> list[str]:
    return [
        f"{clip_uid}_{row_position}"
        for row_position, clip_uid in enumerate(frame["clip_uid"].astype(str))
    ]


def build_adaptive_split(
    adaptive: pd.DataFrame,
    *,
    history_length: int = 8,
    action_registry: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a history manifest without changing the adaptive A1 -> A2 rows."""
    missing = sorted(REQUIRED_COLUMNS - set(adaptive.columns))
    if missing:
        raise ValueError(f"adaptive transition index is missing required columns: {missing}")

    adaptive = adaptive.reset_index(drop=True).copy()
    observed_identity = [
        "video_uid",
        "annotation_level",
        "observed_action_start_sec",
        "observed_action_end_sec",
    ]
    if adaptive.duplicated(observed_identity).any():
        raise ValueError("adaptive transition index contains duplicate observed actions")

    cache_ids = _cache_sample_ids(adaptive)
    if len(cache_ids) != len(set(cache_ids)):
        raise ValueError("adaptive transition cache identities are not unique")

    # ``build_split`` expects the source endpoint table's target_start/end to
    # describe the action represented by each cached visual segment.  In the
    # adaptive table that action is A1, held in observed_action_* columns.
    endpoint_proxy = adaptive.copy()
    endpoint_proxy["target_start_sec"] = endpoint_proxy[
        "observed_action_start_sec"
    ].astype(float)
    endpoint_proxy["target_end_sec"] = endpoint_proxy[
        "observed_action_end_sec"
    ].astype(float)
    endpoint_proxy["matched_level"] = endpoint_proxy["annotation_level"].astype(str)

    targets = adaptive.copy()
    targets["cache_sample_id"] = cache_ids
    output, stats = build_split(
        endpoint_proxy,
        targets,
        history_length=history_length,
        action_registry=action_registry,
    )

    if not output["cache_sample_id"].astype(str).tolist() == cache_ids:
        raise RuntimeError("adaptive history builder changed current cache identities")
    if not (
        output["audit_current_observation_end_sec"].astype(float)
        < output["audit_target_start_sec"].astype(float)
    ).all():
        raise RuntimeError("adaptive history output violates current-before-target causality")
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-length", type=int, default=8)
    args = parser.parse_args()

    adaptive_dir = Path(args.adaptive_index_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = adaptive_dir / "action_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    all_stats: dict[str, object] = {
        "protocol": "adaptive_transition_visual_history_of_completed_cached_actions",
        "adaptive_index_dir": str(adaptive_dir),
        "history_k": int(args.history_length),
        "current_contract": "A1 through A1.end-0.25s from adaptive MR24+8 cache",
        "target_contract": "close immediate same-level successor A2",
        "history_order": "left_padded_then_oldest_to_newest",
        "history_eligibility": (
            "same video_uid and annotation level; cached history observed action end "
            "<= current A1 start"
        ),
        "history_temporal_feature": "current adaptive obs_end - history adaptive obs_end",
        "leakage_contract": "history visual cache IDs only; no history GT labels",
        "source_builder_sha256": _sha256(Path(__file__).resolve()),
    }

    for split in ("train", "val"):
        adaptive, adaptive_path = _read_index(adaptive_dir, split)
        output, stats = build_adaptive_split(
            adaptive,
            history_length=args.history_length,
            action_registry=registry,
        )
        output_path = output_dir / f"{split}.parquet"
        output.to_parquet(output_path, index=False)
        all_stats[split] = {
            **stats,
            "adaptive_index": str(adaptive_path),
            "adaptive_index_sha256": _sha256(adaptive_path),
            "output_index": str(output_path),
            "output_index_sha256": _sha256(output_path),
        }
        print(json.dumps({"split": split, **stats}, ensure_ascii=False), flush=True)

    copied_assets: dict[str, dict[str, str]] = {}
    for filename in ("action_registry.json", "video_uids.txt"):
        source = adaptive_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / filename
        shutil.copy2(source, destination)
        copied_assets[filename] = {
            "source": str(source),
            "source_sha256": _sha256(source),
            "output": str(destination),
            "output_sha256": _sha256(destination),
        }
    all_stats["copied_assets"] = copied_assets

    stats_path = output_dir / "build_stats.json"
    stats_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
