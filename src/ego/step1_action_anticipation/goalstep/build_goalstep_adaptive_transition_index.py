"""Build the GoalStep adaptive predecessor-to-next-action transition index.

For a same-level consecutive pair ``A1 -> A2``, observe only A1, ending a
small guard interval before A1's annotated end, and classify A2.  Pair
selection keeps close transitions with an adaptive gap rule based only on A1:

    0 <= A2.start - A1.end <= min(max_gap_sec, gap_ratio * A1.duration)

The future gap is recorded for audit and cohort construction, but is never an
input to the model.  The V-JEPA predictor horizon remains a fixed config value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_map(annotations_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "val"):
        path = annotations_dir / f"goalstep_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for video in _read_json(path)["videos"]:
            result[str(video["video_uid"])] = video.get("goal_category") or "COOKING:UNKNOWN"
    return result


def _action_map(registry: dict) -> dict[tuple[int, int], int]:
    mapping = {}
    for key, action_id in registry["action_classes"].items():
        verb, noun = key.split("|")
        mapping[(int(verb), int(noun))] = int(action_id)
    return mapping


def build_split(
    labels: pd.DataFrame,
    split: str,
    scenarios: dict[str, str],
    actions: dict[tuple[int, int], int],
    *,
    gap_ratio: float,
    max_gap_sec: float,
    min_action_sec: float,
    guard_sec: float,
    max_observation_sec: float,
) -> tuple[pd.DataFrame, dict]:
    split_labels = labels[labels["split"].eq(split)].copy()
    split_labels = split_labels[split_labels["level"].isin(["step", "substep"])]

    rows: list[dict] = []
    counters = {
        "candidate_pairs": 0,
        "excluded_overlap": 0,
        "excluded_short_observed_action": 0,
        "excluded_adaptive_gap": 0,
        "excluded_unknown_target_class": 0,
    }
    groups = split_labels.groupby(["video_uid", "level"], sort=False)
    for (video_uid, level), group in groups:
        group = group.sort_values(["start_time", "end_time"], kind="stable").reset_index(drop=True)
        for position in range(len(group) - 1):
            observed = group.iloc[position]
            target = group.iloc[position + 1]
            counters["candidate_pairs"] += 1

            observed_start = float(observed["start_time"])
            observed_end = float(observed["end_time"])
            target_start = float(target["start_time"])
            target_end = float(target["end_time"])
            observed_duration = observed_end - observed_start
            gap = target_start - observed_end

            if gap < -1e-6:
                counters["excluded_overlap"] += 1
                continue
            if observed_duration < min_action_sec:
                counters["excluded_short_observed_action"] += 1
                continue
            allowed_gap = min(max_gap_sec, gap_ratio * observed_duration)
            if gap > allowed_gap + 1e-6:
                counters["excluded_adaptive_gap"] += 1
                continue

            target_pair = (int(target["verb_label"]), int(target["noun_label"]))
            if target_pair not in actions:
                counters["excluded_unknown_target_class"] += 1
                continue

            obs_end = observed_end - guard_sec
            obs_start = max(observed_start, obs_end - max_observation_sec)
            if obs_end <= obs_start:
                counters["excluded_short_observed_action"] += 1
                continue
            target_horizon = target_start - obs_end
            if target_horizon < guard_sec - 1e-5:
                raise RuntimeError(
                    f"Target leakage for {video_uid}/{level}/{position}: horizon={target_horizon}"
                )

            observed_pair = (int(observed["verb_label"]), int(observed["noun_label"]))
            rows.append({
                "video_uid": str(video_uid),
                "clip_uid": str(video_uid),
                "obs_start_sec": obs_start,
                "obs_end_sec": obs_end,
                "verb_label": target_pair[0],
                "noun_label": target_pair[1],
                "action_label": actions[target_pair],
                "scenario": scenarios.get(str(video_uid), "COOKING:UNKNOWN"),
                "boundary_flag": False,
                "annotation_level": str(level),
                "observed_action_start_sec": observed_start,
                "observed_action_end_sec": observed_end,
                "observed_action_duration_sec": observed_duration,
                "observed_verb_label": observed_pair[0],
                "observed_noun_label": observed_pair[1],
                "observed_action_label": actions.get(observed_pair, -1),
                "target_start_sec": target_start,
                "target_end_sec": target_end,
                "target_horizon_sec": target_horizon,
                "inter_action_gap_sec": max(0.0, gap),
                "allowed_gap_sec": allowed_gap,
                "observation_duration_sec": obs_end - obs_start,
                "guard_sec": guard_sec,
                "sampling_strategy": "adaptive_multirate_24_8",
            })

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError(f"Adaptive transition index for {split} is empty")
    if not (output["target_start_sec"] > output["obs_end_sec"]).all():
        raise RuntimeError(f"Adaptive transition index for {split} contains target leakage")
    if not (output["inter_action_gap_sec"] <= output["allowed_gap_sec"] + 1e-6).all():
        raise RuntimeError(f"Adaptive transition index for {split} violates the adaptive gap rule")

    same_class = output["action_label"].eq(output["observed_action_label"])
    stats = {
        **counters,
        "retained_samples": int(len(output)),
        "videos": int(output["video_uid"].nunique()),
        "step_samples": int(output["annotation_level"].eq("step").sum()),
        "substep_samples": int(output["annotation_level"].eq("substep").sum()),
        "same_class_transition_samples": int(same_class.sum()),
        "different_class_transition_samples": int((~same_class).sum()),
        "gap_sec_median": float(output["inter_action_gap_sec"].median()),
        "gap_sec_p90": float(output["inter_action_gap_sec"].quantile(0.9)),
        "target_horizon_sec_min": float(output["target_horizon_sec"].min()),
        "target_horizon_sec_median": float(output["target_horizon_sec"].median()),
        "target_horizon_sec_max": float(output["target_horizon_sec"].max()),
        "observation_duration_sec_min": float(output["observation_duration_sec"].min()),
        "observation_duration_sec_median": float(output["observation_duration_sec"].median()),
        "observation_duration_sec_max": float(output["observation_duration_sec"].max()),
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-csv",
        default="src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_step_labels.csv",
    )
    parser.add_argument("--annotations-dir", default="../datasets/Ego4D/v2/annotations")
    parser.add_argument(
        "--registry",
        default="src/ego/step1_action_anticipation/goalstep/index_start_m1_lobs8/action_registry.json",
    )
    parser.add_argument(
        "--output-dir",
        default="src/ego/step1_action_anticipation/goalstep/index_adaptive_transition_mr24x8",
    )
    parser.add_argument("--gap-ratio", type=float, default=0.20)
    parser.add_argument("--max-gap-sec", type=float, default=2.0)
    parser.add_argument("--min-action-sec", type=float, default=1.0)
    parser.add_argument("--guard-sec", type=float, default=0.25)
    parser.add_argument("--max-observation-sec", type=float, default=32.0)
    args = parser.parse_args()

    labels_path = Path(args.labels_csv).resolve()
    annotations_dir = Path(args.annotations_dir).resolve()
    registry_path = Path(args.registry).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path)
    registry = _read_json(registry_path)
    actions = _action_map(registry)
    scenarios = _scenario_map(annotations_dir)
    build_stats: dict[str, object] = {
        "protocol": "adaptive_predecessor_boundary_to_next_same_level_action",
        "labels_csv": str(labels_path),
        "labels_csv_sha256": _sha256(labels_path),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "annotations_dir": str(annotations_dir),
        "gap_ratio": args.gap_ratio,
        "max_gap_sec": args.max_gap_sec,
        "min_action_sec": args.min_action_sec,
        "guard_sec": args.guard_sec,
        "max_observation_sec": args.max_observation_sec,
        "pair_rule": "same-level immediate successor; 0 <= gap <= min(max_gap, gap_ratio*A1.duration)",
        "target": "A2; A2 timing/gap are audit-only and never model inputs",
        "predictor_horizon": "fixed by training config; not the annotated A2 gap",
    }

    outputs = {}
    for split in ("train", "val"):
        output, stats = build_split(
            labels,
            split,
            scenarios,
            actions,
            gap_ratio=args.gap_ratio,
            max_gap_sec=args.max_gap_sec,
            min_action_sec=args.min_action_sec,
            guard_sec=args.guard_sec,
            max_observation_sec=args.max_observation_sec,
        )
        path = output_dir / f"{split}.parquet"
        output.to_parquet(path, index=False)
        outputs[split] = output
        build_stats[split] = {**stats, "output": str(path), "output_sha256": _sha256(path)}
        print(json.dumps({"split": split, **stats}, ensure_ascii=False))

    shutil.copy2(registry_path, output_dir / "action_registry.json")
    video_uids = sorted(set(outputs["train"]["video_uid"]) | set(outputs["val"]["video_uid"]))
    (output_dir / "video_uids.txt").write_text("\n".join(video_uids) + "\n", encoding="utf-8")
    (output_dir / "build_stats.json").write_text(
        json.dumps(build_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'build_stats.json'}")


if __name__ == "__main__":
    main()
