"""Task 3 -- build the GoalStep Z=1 anticipation index (FHO-identical rules).

Reuses ``ego.datasets.ego4d.build_z1_index`` / ``register_action_labels`` /
``filter_to_known_pairs`` **unchanged** -- the only GoalStep-specific work here
is turning ``goalstep_step_labels.csv`` (Task 2) into the per-action DataFrame
those functions already expect:

    video_uid | clip_uid | action_idx | action_clip_start_sec | verb_label | noun_label

Differences from FHO, all mechanical:
  * GoalStep has no clip layer -- timestamps are already video-relative, so
    ``clip_uid := video_uid`` (logged) and features are read from the full video
    (``video_source: full_scale``).
  * ``scenario`` comes from GoalStep's ``goal_category`` (cooking goals such as
    ``COOKING:MAKE_BREAD``) instead of the Ego4D scenario tag list, so the
    existing scenario-stratified sampler and per-scenario breakdown keep working.
  * Split layout is ``train.parquet`` / ``val.parquet`` (no dev/heldout re-split;
    goalstep_val.json is the evaluation set).
  * An extra ``action_label`` column (dense action id from the registry) is
    emitted alongside FHO's columns.

Window rule is FHO's, unchanged: ``obs_end_sec = step_start - tau_a``,
``obs_start_sec = obs_end_sec - l_obs``, boundary handling per
``--boundary-policy``, first action of each video dropped.

Usage:
    python src/ego/step1_action_anticipation/goalstep/build_goalstep_z1_index.py \
        --taxonomy-dir src/ego/step1_action_anticipation/goalstep/taxonomy \
        --annotations-dir data/Ego4D/v2/annotations \
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
from ego.datasets.ego4d import build_z1_index, register_action_labels  # noqa: E402
from ego.datasets.label_mapping import filter_to_known_pairs  # noqa: E402

PHASE = "BuildGoalStepIndex"


def load_goal_categories(annotations_dir: Path) -> dict[str, str]:
    """video_uid -> goal_category, used as the ``scenario`` column."""
    mapping: dict[str, str] = {}
    for split in ("train", "val"):
        path = annotations_dir / f"goalstep_{split}.json"
        if not path.is_file():
            continue
        for video in read_json(path)["videos"]:
            mapping[video["video_uid"]] = video.get("goal_category") or "COOKING:UNKNOWN"
    return mapping


def to_actions_df(labels: pd.DataFrame, scenario_map: dict[str, str]) -> pd.DataFrame:
    """Shape Task 2's per-step labels into the per-action frame ``build_z1_index`` expects."""
    df = labels.sort_values(["video_uid", "start_time", "end_time"]).reset_index(drop=True)
    df["clip_uid"] = df["video_uid"]  # GoalStep has no clip layer
    df["action_idx"] = df.groupby("video_uid").cumcount()
    df["action_clip_start_sec"] = df["start_time"].astype(float)
    return df[["video_uid", "clip_uid", "action_idx", "action_clip_start_sec", "verb_label", "noun_label"]]


def _write_index(df: pd.DataFrame, path_stem: Path) -> Path:
    try:
        out_path = path_stem.with_suffix(".parquet")
        df.to_parquet(out_path)
    except ImportError:
        out_path = path_stem.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    step_log(1, PHASE, f"wrote {out_path} ({len(df)} rows)")
    return out_path


def _add_action_label(df: pd.DataFrame, mapping) -> pd.DataFrame:
    df = df.copy()
    df["action_label"] = [
        mapping.action_classes.get((int(v), int(n)), -1)
        for v, n in zip(df["verb_label"], df["noun_label"])
    ]
    return df[[
        "video_uid", "clip_uid", "obs_start_sec", "obs_end_sec",
        "verb_label", "noun_label", "action_label", "scenario", "boundary_flag",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxonomy-dir", default="src/ego/step1_action_anticipation/goalstep/taxonomy")
    parser.add_argument("--labels-csv", default=None, help="Override goalstep_step_labels.csv path")
    parser.add_argument("--annotations-dir", default="data/Ego4D/v2/annotations")
    parser.add_argument("--level", choices=["step", "substep", "both"], default="both",
                        help="Annotation level to index; default matches the level the verb/noun classes were built at")
    parser.add_argument("--tau-a", type=float, default=1.0, help="Anticipation horizon in seconds")
    parser.add_argument("--l-obs", type=float, default=3.5, help="Observation window length in seconds")
    parser.add_argument("--min-obs-sec", type=float, default=0.5, help="Minimum usable observation length")
    parser.add_argument("--boundary-policy", choices=["truncate", "exclude"], default="truncate")
    parser.add_argument("--drop-duplicate-windows", action="store_true", default=True,
                        help="Drop exact (video, window, verb, noun) duplicates -- a step and its first substep "
                             "usually start at the same timestamp when --level both")
    parser.add_argument("--keep-duplicate-windows", dest="drop_duplicate_windows", action="store_false")
    parser.add_argument("--video-uid-subset", default=None,
                        help="Newline-delimited file of video_uids to restrict the index to (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="src/ego/step1_action_anticipation/goalstep/index")
    args = parser.parse_args()

    tax_dir = Path(args.taxonomy_dir)
    out_dir = ensure_dir(args.output_dir)

    taxonomy = read_json(tax_dir / "goalstep_verbnoun_taxonomy.json")
    step_log(1, PHASE, f"N_verb={len(taxonomy['verbs'])} N_noun={len(taxonomy['nouns'])} (taxonomy space)")

    labels = pd.read_csv(Path(args.labels_csv or tax_dir / "goalstep_step_labels.csv"))
    if args.level != "both":
        labels = labels[labels["level"] == args.level].reset_index(drop=True)
    step_log(1, PHASE, f"Loaded {len(labels)} labelled step instances (level={args.level})")

    if args.video_uid_subset:
        keep = {line.strip() for line in Path(args.video_uid_subset).read_text().splitlines() if line.strip()}
        labels = labels[labels["video_uid"].isin(keep)].reset_index(drop=True)
        step_log(1, PHASE, f"--video-uid-subset: restricted to {labels['video_uid'].nunique()} videos, {len(labels)} instances")

    scenario_map = load_goal_categories(Path(args.annotations_dir))
    step_log(1, PHASE, f"Loaded goal_category (=scenario) for {len(scenario_map)} videos; "
                       f"{len(set(scenario_map.values()))} distinct cooking goals")
    step_log(1, PHASE, "GoalStep has no clip layer: clip_uid := video_uid for every row")

    indices, stats = {}, {}
    for split in ("train", "val"):
        split_labels = labels[labels["split"] == split].reset_index(drop=True)
        actions = to_actions_df(split_labels, scenario_map)
        idx, st = build_z1_index(
            actions,
            tau_a=args.tau_a,
            l_obs=args.l_obs,
            min_obs_sec=args.min_obs_sec,
            boundary_policy=args.boundary_policy,
            scenario_map=scenario_map,
        )
        step_log(1, PHASE, f"{split} Z=1 samples: {st.to_dict()}")
        if args.drop_duplicate_windows:
            before = len(idx)
            idx = idx.drop_duplicates(
                subset=["video_uid", "obs_start_sec", "obs_end_sec", "verb_label", "noun_label"]
            ).reset_index(drop=True)
            step_log(1, PHASE, f"{split}: dropped {before - len(idx)} duplicate (window, label) rows -> {len(idx)}")
        indices[split], stats[split] = idx, st

    mapping = register_action_labels(indices["train"])
    step_log(1, PHASE, f"Registered verb classes: {mapping.num_verbs}")
    step_log(1, PHASE, f"Registered noun classes: {mapping.num_nouns}")
    step_log(1, PHASE, f"Registered (verb,noun) action combinations: {mapping.num_actions}")

    known_pairs = set(mapping.action_classes.keys())
    val_rows = filter_to_known_pairs(
        indices["val"].to_dict("records"), known_pairs, verb_key="verb_label", noun_key="noun_label"
    )
    before_val = len(indices["val"])
    indices["val"] = (
        pd.DataFrame(val_rows, columns=indices["val"].columns) if val_rows else indices["val"].iloc[0:0]
    )
    step_log(1, PHASE, f"Val samples after restricting to train-seen combinations: {len(indices['val'])} "
                       f"(dropped {before_val - len(indices['val'])})")

    for split in ("train", "val"):
        _write_index(_add_action_label(indices[split], mapping), out_dir / split)

    write_json(out_dir / "action_registry.json", {
        "num_verbs": mapping.num_verbs,
        "num_nouns": mapping.num_nouns,
        "num_actions": mapping.num_actions,
        "verb_classes": {str(k): v for k, v in mapping.verb_classes.items()},
        "noun_classes": {str(k): v for k, v in mapping.noun_classes.items()},
        "action_classes": {f"{v}|{n}": a for (v, n), a in mapping.action_classes.items()},
    })

    uid_path = out_dir / "video_uids.txt"
    all_uids = sorted(set(indices["train"]["video_uid"]) | set(indices["val"]["video_uid"]))
    uid_path.write_text("\n".join(all_uids) + "\n")
    step_log(1, PHASE, f"Wrote {uid_path} ({len(all_uids)} video_uids to download)")

    write_json(out_dir / "build_stats.json", {
        "level": args.level,
        "tau_a": args.tau_a,
        "l_obs": args.l_obs,
        "min_obs_sec": args.min_obs_sec,
        "boundary_policy": args.boundary_policy,
        "drop_duplicate_windows": args.drop_duplicate_windows,
        "seed": args.seed,
        "num_verbs_taxonomy": len(taxonomy["verbs"]),
        "num_nouns_taxonomy": len(taxonomy["nouns"]),
        "num_registered_actions": mapping.num_actions,
        "train": stats["train"].to_dict(),
        "val": stats["val"].to_dict(),
        "train_samples": int(len(indices["train"])),
        "val_samples": int(len(indices["val"])),
        "train_videos": int(indices["train"]["video_uid"].nunique()),
        "val_videos": int(indices["val"]["video_uid"].nunique()),
        "num_scenarios": int(pd.concat([indices["train"], indices["val"]])["scenario"].nunique()),
    })
    step_log(1, PHASE, f"Wrote action_registry.json and build_stats.json to {out_dir}")


if __name__ == "__main__":
    main()
