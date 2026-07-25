"""Build a training-only Ego4D LTA auxiliary index in the GoalStep label space.

Each sample starts from an LTA action A2.  The fixed observation window ends
at ``A2.end - 1s`` and spans at most eight seconds.  Its target A3 is the
first later ``action_idx`` in the same clip whose start is at or after A2's
end, matching GoalStep's strict-next contract while skipping LTA's overlapping
eight-second annotations.  Operational ``*_sec`` columns are clip-relative so
the existing clip-file decoder can consume them directly.  Explicit
``*_video_sec`` audit columns preserve the parent-video conversion:

``video_time = clip_parent_start_sec + action_clip_time``.

LTA target text is conservatively exact-matched to the existing GoalStep
taxonomy.  Unmatched verb/noun heads receive ``-1`` and a false loss mask; the
GoalStep action vocabulary is never extended.  By default only verb+noun
targets are emitted (the handoff's A1 arm), while ``--match-policy any`` is
available for a later partial-label ablation.

This builder is deliberately fail-closed:

* the GoalStep validation-video set must be non-empty and every overlapping
  LTA video is excluded;
* ``A3.start >= A2.end`` and ``obs_end_sec < target_start_sec`` are enforced,
  because otherwise the fixed LTA endpoint would turn next-action
  anticipation into target recognition;
* malformed coordinates, duplicate cache IDs, and taxonomy collisions raise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


REQUIRED_ANNOTATION_FIELDS = {
    "video_uid",
    "clip_uid",
    "clip_parent_start_sec",
    "action_clip_start_sec",
    "action_clip_end_sec",
    "action_idx",
    "verb",
    "noun",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lta_text(value: object) -> str:
    """Apply the exact head-token normalization fixed by the handoff."""

    return " ".join(
        str(value).split("_(", 1)[0].lower().replace("_", " ").strip().split()
    )


def load_registry(path: str | Path) -> dict:
    path = Path(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {"verb_classes", "noun_classes", "action_classes"}
    missing = sorted(required - set(registry))
    if missing:
        raise ValueError(f"{path}: registry is missing fields {missing}")
    for key in required:
        if not isinstance(registry[key], dict):
            raise ValueError(f"{path}: registry field {key!r} must be an object")
    return registry


def load_taxonomy_lookup(
    csv_path: str | Path,
    registered_classes: Mapping[str, int],
) -> dict[str, str]:
    """Map normalized class keys and members to registered raw class IDs.

    This intentionally reproduces handoff §3.1 literally: members are split
    on commas and pipes only.  The checked-in CSV happens to use semicolons in
    a few rows, so those compound strings do *not* expand into extra aliases
    in the conservative A1 cohort.  ``class_key`` is always included.
    """

    csv_path = Path(csv_path)
    lookup: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"class_id", "class_key", "members"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path}: expected columns {sorted(required)}, got {reader.fieldnames}"
            )
        for row in reader:
            raw_class_id = str(row["class_id"])
            if raw_class_id not in registered_classes:
                continue
            members = [
                row["class_key"],
                *row.get("members", "").replace("|", ",").split(","),
            ]
            for member in members:
                normalized = normalize_lta_text(member)
                if not normalized:
                    continue
                previous = lookup.get(normalized)
                if previous is not None and previous != raw_class_id:
                    raise ValueError(
                        f"{csv_path}: normalized taxonomy surface {normalized!r} "
                        f"maps to both class {previous} and {raw_class_id}"
                    )
                lookup[normalized] = raw_class_id
    if not lookup:
        raise ValueError(f"{csv_path}: no taxonomy surfaces matched the registry")
    return lookup


def load_lta_annotations(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    seen_clip_splits: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("clips") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a top-level 'clips' list")
        split = str(payload.get("split") or path.stem.removeprefix("fho_lta_"))
        frame = pd.DataFrame(records)
        missing = sorted(REQUIRED_ANNOTATION_FIELDS - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: annotation rows are missing fields {missing}")
        frame = frame.copy()
        frame["source_split"] = split
        frame["source_row"] = range(len(frame))
        for clip_uid in frame["clip_uid"].astype(str).unique():
            previous = seen_clip_splits.get(clip_uid)
            if previous is not None and previous != split:
                raise ValueError(
                    f"clip_uid {clip_uid!r} occurs in both {previous!r} and {split!r}"
                )
            seen_clip_splits[clip_uid] = split
        frames.append(frame)
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "split": split,
                "actions": int(len(frame)),
                "clips": int(frame["clip_uid"].nunique()),
                "videos": int(frame["video_uid"].nunique()),
            }
        )
    if not frames:
        raise ValueError("At least one LTA annotation JSON is required")
    actions = pd.concat(frames, ignore_index=True)
    return actions, sources


def load_goalstep_val_videos(path: str | Path) -> set[str]:
    """Load the authoritative validation video set and reject empty inputs."""

    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["video_uid"])
        videos = set(frame["video_uid"].dropna().astype(str))
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, usecols=["video_uid"])
        videos = set(frame["video_uid"].dropna().astype(str))
    else:
        videos = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if not videos:
        raise ValueError(
            f"{path}: GoalStep validation-video set is empty; refusing leakage filtering"
        )
    return videos


def load_available_clip_videos(path: str | Path) -> set[str]:
    """Return clip UID stems from the Stage-1 media directory, fail-closed."""

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"LTA clip-video directory does not exist: {path}")
    clip_uids = {item.stem for item in path.glob("*.mp4") if item.is_file()}
    if not clip_uids:
        raise ValueError(f"{path}: no .mp4 clip videos found")
    return clip_uids


def copy_registry_for_output(registry_path: str | Path, output_dir: str | Path) -> Path:
    """Copy the GoalStep registry byte-for-byte and verify its fingerprint."""

    source = Path(registry_path).resolve()
    destination = Path(output_dir).resolve() / "action_registry.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if source.read_bytes() != destination.read_bytes() or _sha256(source) != _sha256(
        destination
    ):
        raise RuntimeError(f"Registry copy verification failed: {source} -> {destination}")
    return destination


def _mapped_target(
    target: pd.Series,
    verb_lookup: Mapping[str, str],
    noun_lookup: Mapping[str, str],
    registry: Mapping[str, object],
) -> dict:
    raw_verb = verb_lookup.get(normalize_lta_text(target["verb"]))
    raw_noun = noun_lookup.get(normalize_lta_text(target["noun"]))

    verb_id = (
        int(registry["verb_classes"][raw_verb])  # type: ignore[index]
        if raw_verb is not None
        else -1
    )
    noun_id = (
        int(registry["noun_classes"][raw_noun])  # type: ignore[index]
        if raw_noun is not None
        else -1
    )
    action_id = -1
    if raw_verb is not None and raw_noun is not None:
        action_id = int(
            registry["action_classes"].get(f"{raw_verb}|{raw_noun}", -1)  # type: ignore[union-attr]
        )
    return {
        # Raw GoalStep taxonomy IDs.  They happen to be identity-mapped in the
        # current registry, but keeping both forms makes that assumption
        # explicit and avoids coupling future trainers to it.
        "verb_label": int(raw_verb) if raw_verb is not None else -1,
        "noun_label": int(raw_noun) if raw_noun is not None else -1,
        # Dense head IDs used for loss computation.
        "verb_id": verb_id,
        "noun_id": noun_id,
        "action_id": action_id,
        # Compatibility alias for existing GoalStep data paths.
        "action_label": action_id,
        "verb_mask": raw_verb is not None,
        "noun_mask": raw_noun is not None,
        "action_mask": action_id >= 0,
    }


def build_lta_aux_index(
    actions: pd.DataFrame,
    *,
    verb_lookup: Mapping[str, str],
    noun_lookup: Mapping[str, str],
    registry: Mapping[str, object],
    goalstep_val_videos: set[str],
    available_clip_uids: set[str] | None = None,
    match_policy: str = "both",
    tau_a: float = 1.0,
    l_obs: float = 8.0,
) -> tuple[pd.DataFrame, dict]:
    """Construct the fixed-window, strict-future, training-only aux cohort."""

    if match_policy not in {"both", "any"}:
        raise ValueError(f"match_policy must be 'both' or 'any', got {match_policy!r}")
    if not goalstep_val_videos:
        raise ValueError("GoalStep validation-video set must be non-empty")
    if tau_a <= 0 or l_obs <= 0:
        raise ValueError(f"tau_a and l_obs must be positive, got {tau_a}, {l_obs}")
    missing = sorted(REQUIRED_ANNOTATION_FIELDS - set(actions.columns))
    if missing:
        raise ValueError(f"LTA annotations are missing fields {missing}")

    counters = {
        "source_actions": int(len(actions)),
        "source_clips": int(actions["clip_uid"].nunique()),
        "source_videos": int(actions["video_uid"].nunique()),
        "source_a2_candidates": 0,
        "source_action_idx_gaps": 0,
        "excluded_invalid_coordinates": 0,
        "excluded_nonpositive_observation": 0,
        "excluded_no_strict_later_target": 0,
        "rows_skipping_overlapping_annotations": 0,
        "skipped_overlapping_annotations_total": 0,
        "mapped_verb_candidates": 0,
        "mapped_noun_candidates": 0,
        "mapped_any_candidates": 0,
        "mapped_both_candidates": 0,
        "mapped_action_candidates": 0,
        "excluded_match_policy": 0,
        "excluded_goalstep_val_rows": 0,
        "excluded_goalstep_val_any_rows": 0,
        "excluded_goalstep_val_both_rows": 0,
        "excluded_missing_clip_video_rows": 0,
    }
    rows: list[dict] = []
    skipped_overlapping_counts: list[int] = []
    inter_action_gaps: list[float] = []

    for clip_uid, clip in actions.groupby("clip_uid", sort=False):
        clip = clip.sort_values(["action_idx", "source_row"], kind="stable").reset_index(
            drop=True
        )
        if clip["action_idx"].duplicated().any():
            raise ValueError(f"clip {clip_uid!r} contains duplicate action_idx values")
        if clip["video_uid"].astype(str).nunique() != 1:
            raise ValueError(f"clip {clip_uid!r} maps to multiple video_uid values")
        if clip["clip_parent_start_sec"].astype(float).nunique() != 1:
            raise ValueError(f"clip {clip_uid!r} has inconsistent parent start timestamps")
        action_indices = clip["action_idx"].astype(int).tolist()
        counters["source_action_idx_gaps"] += sum(
            int(after != before + 1)
            for before, after in zip(action_indices[:-1], action_indices[1:])
        )

        for pair_position in range(max(0, len(clip) - 1)):
            observed = clip.iloc[pair_position]
            counters["source_a2_candidates"] += 1
            a2_idx = int(observed["action_idx"])
            parent_start = float(observed["clip_parent_start_sec"])
            a2_start_local = float(observed["action_clip_start_sec"])
            a2_end_local = float(observed["action_clip_end_sec"])
            if (
                not all(
                    math.isfinite(value)
                    for value in (parent_start, a2_start_local, a2_end_local)
                )
                or a2_end_local <= a2_start_local
                or a2_start_local < 0
            ):
                counters["excluded_invalid_coordinates"] += 1
                continue

            target_position: int | None = None
            for candidate_position in range(pair_position + 1, len(clip)):
                candidate_start = float(
                    clip.iloc[candidate_position]["action_clip_start_sec"]
                )
                if math.isfinite(candidate_start) and candidate_start >= a2_end_local:
                    target_position = candidate_position
                    break
            if target_position is None:
                counters["excluded_no_strict_later_target"] += 1
                continue

            target = clip.iloc[target_position]
            a3_idx = int(target["action_idx"])
            a3_start_local = float(target["action_clip_start_sec"])
            a3_end_local = float(target["action_clip_end_sec"])
            coordinates = (
                parent_start,
                a2_start_local,
                a2_end_local,
                a3_start_local,
                a3_end_local,
            )
            if (
                not all(math.isfinite(value) for value in coordinates)
                or a2_end_local <= a2_start_local
                or a3_end_local <= a3_start_local
                or a2_start_local < 0
                or a3_start_local < 0
            ):
                counters["excluded_invalid_coordinates"] += 1
                continue
            skipped_overlapping = target_position - pair_position - 1
            inter_action_gap = a3_start_local - a2_end_local
            if skipped_overlapping:
                counters["rows_skipping_overlapping_annotations"] += 1
            counters["skipped_overlapping_annotations_total"] += skipped_overlapping
            skipped_overlapping_counts.append(skipped_overlapping)
            inter_action_gaps.append(inter_action_gap)

            obs_end_local = a2_end_local - tau_a
            obs_start_local = max(0.0, obs_end_local - l_obs)
            if obs_end_local <= obs_start_local:
                counters["excluded_nonpositive_observation"] += 1
                continue
            obs_start_video = parent_start + obs_start_local
            obs_end_video = parent_start + obs_end_local
            target_start_video = parent_start + a3_start_local
            target_end_video = parent_start + a3_end_local
            if target_start_video < parent_start + a2_end_local:
                raise RuntimeError("Strict-next target starts before A2 ends")
            if not obs_end_video < target_start_video:
                raise RuntimeError("Strict-next target is not after the observation")

            mapped = _mapped_target(
                target,
                verb_lookup=verb_lookup,
                noun_lookup=noun_lookup,
                registry=registry,
            )
            verb_mask = bool(mapped["verb_mask"])
            noun_mask = bool(mapped["noun_mask"])
            action_mask = bool(mapped["action_mask"])
            counters["mapped_verb_candidates"] += int(verb_mask)
            counters["mapped_noun_candidates"] += int(noun_mask)
            counters["mapped_any_candidates"] += int(verb_mask or noun_mask)
            counters["mapped_both_candidates"] += int(verb_mask and noun_mask)
            counters["mapped_action_candidates"] += int(action_mask)

            selected = verb_mask and noun_mask if match_policy == "both" else verb_mask or noun_mask
            if not selected:
                counters["excluded_match_policy"] += 1
                continue

            video_uid = str(observed["video_uid"])
            if video_uid in goalstep_val_videos:
                counters["excluded_goalstep_val_rows"] += 1
                counters["excluded_goalstep_val_any_rows"] += int(verb_mask or noun_mask)
                counters["excluded_goalstep_val_both_rows"] += int(verb_mask and noun_mask)
                continue
            if available_clip_uids is not None and str(clip_uid) not in available_clip_uids:
                counters["excluded_missing_clip_video_rows"] += 1
                continue

            cache_sample_id = f"ltaaux_{clip_uid}_{a2_idx}_{a3_idx}"
            rows.append(
                {
                    "video_uid": video_uid,
                    "clip_uid": str(clip_uid),
                    "cache_sample_id": cache_sample_id,
                    "clip_parent_start_sec": parent_start,
                    "obs_start_sec": obs_start_local,
                    "obs_end_sec": obs_end_local,
                    "obs_start_clip_sec": obs_start_local,
                    "obs_end_clip_sec": obs_end_local,
                    "obs_start_video_sec": obs_start_video,
                    "obs_end_video_sec": obs_end_video,
                    **mapped,
                    "scenario": "unknown",
                    "boundary_flag": bool(obs_start_local == 0.0),
                    "observation_anchor": "lta_a2_end_minus_tau_fixed_window",
                    "observed_action_idx": a2_idx,
                    "target_action_idx": a3_idx,
                    "target_action_rank_gap": int(target_position - pair_position),
                    "skipped_overlapping_actions": skipped_overlapping,
                    "observed_action_start_sec": a2_start_local,
                    "observed_action_end_sec": a2_end_local,
                    "target_start_sec": a3_start_local,
                    "target_end_sec": a3_end_local,
                    "observed_action_start_video_sec": parent_start + a2_start_local,
                    "observed_action_end_video_sec": parent_start + a2_end_local,
                    "target_start_video_sec": target_start_video,
                    "target_end_video_sec": target_end_video,
                    "target_horizon_sec": a3_start_local - obs_end_local,
                    "inter_action_gap_sec": inter_action_gap,
                    "observed_verb_text": str(observed["verb"]),
                    "observed_noun_text": str(observed["noun"]),
                    "target_verb_text": str(target["verb"]),
                    "target_noun_text": str(target["noun"]),
                    "source_split": str(observed.get("source_split", "unknown")),
                    "source_row_a2": int(observed.get("source_row", pair_position)),
                    "source_row_a3": int(target.get("source_row", target_position)),
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("LTA aux construction retained zero rows")
    if output["cache_sample_id"].duplicated().any():
        duplicates = output.loc[
            output["cache_sample_id"].duplicated(keep=False), "cache_sample_id"
        ].tolist()
        raise RuntimeError(f"Duplicate cache_sample_id values: {duplicates[:5]}")
    leaked = set(output["video_uid"].astype(str)) & set(goalstep_val_videos)
    if leaked:
        raise RuntimeError(f"GoalStep validation leakage remained after filtering: {sorted(leaked)[:5]}")
    if not bool((output["obs_end_sec"] < output["target_start_sec"]).all()):
        raise RuntimeError("Non-anticipatory rows remained after strict-future filtering")
    if not bool(
        (output["target_start_sec"] >= output["observed_action_end_sec"]).all()
    ):
        raise RuntimeError("Output contains a target which begins before A2 ends")
    if match_policy == "both" and not bool(
        (output["verb_mask"].astype(bool) & output["noun_mask"].astype(bool)).all()
    ):
        raise RuntimeError("both-match output contains a partial-label row")

    counters.update(
        {
            "match_policy": match_policy,
            "output_rows": int(len(output)),
            "output_clips": int(output["clip_uid"].nunique()),
            "output_videos": int(output["video_uid"].nunique()),
            "output_verb_mask_rows": int(output["verb_mask"].sum()),
            "output_noun_mask_rows": int(output["noun_mask"].sum()),
            "output_action_mask_rows": int(output["action_mask"].sum()),
            "output_covered_actions": int(output.loc[output["action_mask"], "action_id"].nunique()),
            "output_boundary_truncated_rows": int(output["boundary_flag"].sum()),
            "output_target_horizon_sec_min": float(output["target_horizon_sec"].min()),
            "output_target_horizon_sec_median": float(output["target_horizon_sec"].median()),
            "output_target_horizon_sec_max": float(output["target_horizon_sec"].max()),
            "strict_target_skipped_overlaps_min": int(min(skipped_overlapping_counts)),
            "strict_target_skipped_overlaps_median": float(
                pd.Series(skipped_overlapping_counts).median()
            ),
            "strict_target_skipped_overlaps_p90": float(
                pd.Series(skipped_overlapping_counts).quantile(0.9)
            ),
            "strict_target_skipped_overlaps_max": int(max(skipped_overlapping_counts)),
            "strict_target_inter_action_gap_sec_min": float(min(inter_action_gaps)),
            "strict_target_inter_action_gap_sec_median": float(
                pd.Series(inter_action_gaps).median()
            ),
            "strict_target_inter_action_gap_sec_p90": float(
                pd.Series(inter_action_gaps).quantile(0.9)
            ),
            "strict_target_inter_action_gap_sec_max": float(max(inter_action_gaps)),
            "goalstep_val_video_count": int(len(goalstep_val_videos)),
            "output_source_split_rows": {
                str(key): int(value)
                for key, value in output["source_split"].value_counts().sort_index().items()
            },
        }
    )
    return output.reset_index(drop=True), counters


def build_and_write(args: argparse.Namespace) -> dict:
    train_json = Path(args.lta_train_json).resolve()
    val_json = Path(args.lta_val_json).resolve()
    verb_csv = Path(args.verb_taxonomy_csv).resolve()
    noun_csv = Path(args.noun_taxonomy_csv).resolve()
    registry_path = Path(args.registry).resolve()
    val_index_path = Path(args.goalstep_val_index).resolve()
    clip_video_dir = Path(args.clip_video_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    registry = load_registry(registry_path)
    verb_lookup = load_taxonomy_lookup(verb_csv, registry["verb_classes"])
    noun_lookup = load_taxonomy_lookup(noun_csv, registry["noun_classes"])
    actions, sources = load_lta_annotations([train_json, val_json])
    goalstep_val_videos = load_goalstep_val_videos(val_index_path)
    available_clip_uids = load_available_clip_videos(clip_video_dir)
    output, stats = build_lta_aux_index(
        actions,
        verb_lookup=verb_lookup,
        noun_lookup=noun_lookup,
        registry=registry,
        goalstep_val_videos=goalstep_val_videos,
        available_clip_uids=available_clip_uids,
        match_policy=args.match_policy,
        tau_a=args.tau_a,
        l_obs=args.l_obs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.parquet"
    if (output_dir / "val.parquet").exists():
        raise RuntimeError(
            f"{output_dir / 'val.parquet'} exists, but the LTA aux contract is training-only"
        )
    output.to_parquet(output_path, index=False)
    output_registry_path = copy_registry_for_output(registry_path, output_dir)

    manifest = {
        "protocol": "lta_first_strict_later_a3_fixed_end_minus_tau_next_action_aux",
        "coordinate_contract": (
            "operational *_sec fields are clip-relative for clip_256ss decoding; "
            "audit *_video_sec fields equal clip_parent_start_sec+clip_time"
        ),
        "observation_contract": {
            "anchor": "A2.action_clip_end_sec - tau_a",
            "tau_a_sec": float(args.tau_a),
            "max_observation_sec": float(args.l_obs),
            "sampling": "32_uniform_frames (performed by Stage 1, not this builder)",
            "target_rule": (
                "first later action_idx in the same clip with A3.start >= A2.end"
            ),
            "strict_future": (
                "A3.target_start_sec >= A2.observed_action_end_sec and "
                "obs_end_sec < A3.target_start_sec"
            ),
            "adaptive_transition_window": False,
        },
        "label_contract": {
            "goalstep_registry": str(registry_path),
            "goalstep_registry_sha256": _sha256(registry_path),
            "output_registry": str(output_registry_path),
            "output_registry_sha256": _sha256(output_registry_path),
            "output_registry_byte_identical": True,
            "num_verbs": int(registry.get("num_verbs", len(registry["verb_classes"]))),
            "num_nouns": int(registry.get("num_nouns", len(registry["noun_classes"]))),
            "num_actions": int(registry.get("num_actions", len(registry["action_classes"]))),
            "match_policy": args.match_policy,
            "taxonomy_synonym_policy": (
                "literal handoff section 3.1: class_key plus members split on "
                "comma/pipe only; semicolon is not expanded"
            ),
            "unmatched_id": -1,
            "partial_labels_are_masked": True,
        },
        "leakage_contract": {
            "goalstep_val_index": str(val_index_path),
            "goalstep_val_index_sha256": _sha256(val_index_path),
            "goalstep_val_videos_excluded": True,
        },
        "media_contract": {
            "clip_video_dir": str(clip_video_dir),
            "available_mp4_clips": int(len(available_clip_uids)),
            "missing_selected_rows_excluded": True,
        },
        "sources": sources,
        "taxonomy": {
            "verb_csv": str(verb_csv),
            "verb_csv_sha256": _sha256(verb_csv),
            "noun_csv": str(noun_csv),
            "noun_csv_sha256": _sha256(noun_csv),
            "verb_lookup_surfaces": int(len(verb_lookup)),
            "noun_lookup_surfaces": int(len(noun_lookup)),
        },
        **stats,
        "output_index": str(output_path),
        "output_index_sha256": _sha256(output_path),
    }
    stats_path = output_dir / "build_stats.json"
    stats_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_rows": manifest["output_rows"],
                "output_videos": manifest["output_videos"],
                "match_policy": manifest["match_policy"],
                "excluded_no_strict_later_target": manifest[
                    "excluded_no_strict_later_target"
                ],
                "skipped_overlapping_annotations_total": manifest[
                    "skipped_overlapping_annotations_total"
                ],
                "excluded_goalstep_val_rows": manifest["excluded_goalstep_val_rows"],
                "excluded_missing_clip_video_rows": manifest[
                    "excluded_missing_clip_video_rows"
                ],
            },
            ensure_ascii=False,
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lta-train-json",
        default="../datasets/Ego4D/v2/annotations/fho_lta_train.json",
    )
    parser.add_argument(
        "--lta-val-json",
        default="../datasets/Ego4D/v2/annotations/fho_lta_val.json",
    )
    parser.add_argument(
        "--verb-taxonomy-csv",
        default="src/ego/step1_action_anticipation/goalstep/taxonomy/verb_classes.csv",
    )
    parser.add_argument(
        "--noun-taxonomy-csv",
        default="src/ego/step1_action_anticipation/goalstep/taxonomy/noun_classes.csv",
    )
    parser.add_argument(
        "--registry",
        default=(
            "src/ego/step1_action_anticipation/goalstep/"
            "index_end_m1_lobs8/action_registry.json"
        ),
    )
    parser.add_argument(
        "--goalstep-val-index",
        default=(
            "src/ego/step1_action_anticipation/goalstep/"
            "index_end_m1_lobs8/val.parquet"
        ),
    )
    parser.add_argument(
        "--clip-video-dir",
        default="../datasets/Ego4D/v2/clip_256ss",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "src/ego/step1_action_anticipation/goalstep/"
            "index_lta_aux_end_m1_lobs8"
        ),
    )
    parser.add_argument("--match-policy", choices=("both", "any"), default="both")
    parser.add_argument("--tau-a", type=float, default=1.0)
    parser.add_argument("--l-obs", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    build_and_write(parse_args())
