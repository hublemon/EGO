"""Build the Ego4D LTA Z=1 anticipation index from fho_lta_{train,val}.json + taxonomy.

Converts each action segment into one anticipation sample (observe
[obs_start_sec, obs_end_sec], predict the action starting tau_a seconds
later), registers a dense (verb, noun) -> action_id table from train-only
combinations, and re-splits val into an internal dev/heldout pair. See
``ego.datasets.ego4d`` for the parsing/index-building logic this wraps and
``PILOT.md`` for the recommended validation order.

Usage:
    python scripts/step1/ego4d_lta/build_lta_z1_index.py \
        --taxonomy /path/to/fho_lta_taxonomy.json \
        --train-json /path/to/fho_lta_train.json \
        --val-json /path/to/fho_lta_val.json \
        --ego4d-json /path/to/ego4d.json \
        --output-dir outputs/ego4d_lta/index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ego.common.io import ensure_dir, write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.datasets.ego4d import (  # noqa: E402
    build_z1_index,
    load_lta_taxonomy,
    load_video_scenarios,
    parse_lta_annotations,
    register_action_labels,
    split_dev_heldout,
)
from ego.datasets.ego4d_stats import build_pilot_taxonomy  # noqa: E402
from ego.datasets.label_mapping import filter_to_known_pairs  # noqa: E402


def _write_index(df, path_stem: Path) -> Path:
    try:
        out_path = path_stem.with_suffix(".parquet")
        df.to_parquet(out_path)
    except ImportError:
        out_path = path_stem.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    step_log(1, "BuildLTAIndex", f"wrote {out_path} ({len(df)} rows)")
    return out_path


def build_index(args: argparse.Namespace) -> dict:
    output_dir = ensure_dir(args.output_dir)

    step_log(1, "BuildLTAIndex", f"Taxonomy: {args.taxonomy}")
    taxonomy = load_lta_taxonomy(args.taxonomy)
    step_log(1, "BuildLTAIndex", f"N_verb={taxonomy.num_verbs} N_noun={taxonomy.num_nouns}")

    scenario_map = load_video_scenarios(args.ego4d_json) if args.ego4d_json else {}
    if args.ego4d_json:
        step_log(1, "BuildLTAIndex", f"Loaded scenario tags for {len(scenario_map)} videos from {args.ego4d_json}")
    else:
        step_log(1, "BuildLTAIndex", "No --ego4d-json given; scenario column will be 'unknown' for all samples")

    step_log(1, "BuildLTAIndex", f"Train annotations: {args.train_json}")
    train_actions = parse_lta_annotations(args.train_json)
    step_log(1, "BuildLTAIndex", f"Val annotations: {args.val_json}")
    val_actions = parse_lta_annotations(args.val_json)

    if args.train_clip_fraction < 1.0:
        import random

        clip_uids = sorted(train_actions["clip_uid"].unique().tolist())
        rng = random.Random(args.seed)
        rng.shuffle(clip_uids)
        n_keep = max(1, round(len(clip_uids) * args.train_clip_fraction))
        keep_clips = set(clip_uids[:n_keep])
        train_actions = train_actions[train_actions["clip_uid"].isin(keep_clips)].reset_index(drop=True)
        step_log(
            1, "BuildLTAIndex",
            f"--train-clip-fraction {args.train_clip_fraction}: kept {len(keep_clips)}/{len(clip_uids)} "
            f"train clips ({len(train_actions)} action rows)",
        )

    train_index, train_stats = build_z1_index(
        train_actions,
        tau_a=args.tau_a,
        l_obs=args.l_obs,
        min_obs_sec=args.min_obs_sec,
        boundary_policy=args.boundary_policy,
        scenario_map=scenario_map,
    )
    val_index, val_stats = build_z1_index(
        val_actions,
        tau_a=args.tau_a,
        l_obs=args.l_obs,
        min_obs_sec=args.min_obs_sec,
        boundary_policy=args.boundary_policy,
        scenario_map=scenario_map,
    )
    step_log(1, "BuildLTAIndex", f"Train Z=1 samples: {train_stats.to_dict()}")
    step_log(1, "BuildLTAIndex", f"Val Z=1 samples: {val_stats.to_dict()}")

    pilot_info = None
    if args.top_verb is not None or args.top_noun is not None:
        top_verb = args.top_verb or train_index["verb_label"].nunique()
        top_noun = args.top_noun or train_index["noun_label"].nunique()
        train_index, pilot_info = build_pilot_taxonomy(
            train_index, top_verb=top_verb, top_noun=top_noun, mode=args.pilot_mode
        )
        # Val must be restricted to the SAME kept raw ids the pilot train
        # taxonomy uses, or verb/noun ids downstream code doesn't know about
        # would leak into dev/heldout.
        kept_verbs = set(pilot_info["kept_verb_ids"])
        kept_nouns = set(pilot_info["kept_noun_ids"])
        if args.pilot_mode == "exclude":
            val_index = val_index[
                val_index["verb_label"].isin(kept_verbs) & val_index["noun_label"].isin(kept_nouns)
            ].reset_index(drop=True)
        else:
            val_index = val_index.copy()
            val_index["verb_label"] = val_index["verb_label"].where(val_index["verb_label"].isin(kept_verbs), -1)
            val_index["noun_label"] = val_index["noun_label"].where(val_index["noun_label"].isin(kept_nouns), -1)
        step_log(
            1, "BuildLTAIndex",
            "*** PILOT TAXONOMY ACTIVE: this index is NOT comparable to a full-taxonomy index "
            f"(top_verb={top_verb}, top_noun={top_noun}, mode={args.pilot_mode}, "
            f"train rows {pilot_info['rows_before']} -> {pilot_info['rows_after']}) ***",
        )

    mapping = register_action_labels(train_index)
    step_log(1, "BuildLTAIndex", f"Registered verb classes: {mapping.num_verbs}")
    step_log(1, "BuildLTAIndex", f"Registered noun classes: {mapping.num_nouns}")
    step_log(1, "BuildLTAIndex", f"Registered (verb,noun) action combinations: {mapping.num_actions}")

    known_pairs = set(mapping.action_classes.keys())
    val_rows = filter_to_known_pairs(
        val_index.to_dict("records"), known_pairs, verb_key="verb_label", noun_key="noun_label"
    )
    val_index = type(val_index)(val_rows, columns=val_index.columns) if val_rows else val_index.iloc[0:0]
    step_log(1, "BuildLTAIndex", f"Val samples after restricting to train-seen combinations: {len(val_index)}")

    dev_index, heldout_index = split_dev_heldout(val_index, dev_fraction=args.dev_fraction, seed=args.seed)
    step_log(
        1, "BuildLTAIndex",
        f"Val split -> dev={len(dev_index)} heldout={len(heldout_index)} (seed={args.seed})",
    )

    _write_index(train_index, output_dir / "train")
    _write_index(dev_index, output_dir / "dev")
    _write_index(heldout_index, output_dir / "heldout")

    write_json(
        output_dir / "action_registry.json",
        {
            "num_verbs": mapping.num_verbs,
            "num_nouns": mapping.num_nouns,
            "num_actions": mapping.num_actions,
            "verb_classes": {str(k): v for k, v in mapping.verb_classes.items()},
            "noun_classes": {str(k): v for k, v in mapping.noun_classes.items()},
            "action_classes": {f"{v}|{n}": a for (v, n), a in mapping.action_classes.items()},
        },
    )
    summary = {
        "tau_a": args.tau_a,
        "l_obs": args.l_obs,
        "min_obs_sec": args.min_obs_sec,
        "boundary_policy": args.boundary_policy,
        "pilot_taxonomy": pilot_info,
        "dev_fraction": args.dev_fraction,
        "train_clip_fraction": args.train_clip_fraction,
        "seed": args.seed,
        "num_verbs_taxonomy": taxonomy.num_verbs,
        "num_nouns_taxonomy": taxonomy.num_nouns,
        "num_registered_actions": mapping.num_actions,
        "train": train_stats.to_dict(),
        "val": val_stats.to_dict(),
        "dev_samples": len(dev_index),
        "heldout_samples": len(heldout_index),
    }
    write_json(output_dir / "build_stats.json", summary)
    step_log(1, "BuildLTAIndex", f"Wrote action_registry.json and build_stats.json to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxonomy", required=True, help="Path to fho_lta_taxonomy.json")
    parser.add_argument("--train-json", required=True, help="Path to fho_lta_train.json")
    parser.add_argument("--val-json", required=True, help="Path to fho_lta_val.json")
    parser.add_argument("--ego4d-json", default=None, help="Path to ego4d.json (for scenario tags); optional")
    parser.add_argument("--tau-a", type=float, default=1.0, help="Anticipation horizon in seconds")
    parser.add_argument("--l-obs", type=float, default=3.5, help="Observation window length in seconds")
    parser.add_argument("--min-obs-sec", type=float, default=0.5, help="Minimum usable observation length")
    parser.add_argument(
        "--boundary-policy", choices=["truncate", "exclude"], default="truncate",
        help="How to handle obs_start_sec < clip start",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.8, help="Fraction of val clips kept as internal dev")
    parser.add_argument(
        "--train-clip-fraction", type=float, default=1.0,
        help="Subsample this fraction of train clips (by clip_uid, seeded) -- for pilot runs, e.g. 0.15",
    )
    parser.add_argument("--top-verb", type=int, default=None, help="Pilot taxonomy: keep only top-N verb classes")
    parser.add_argument("--top-noun", type=int, default=None, help="Pilot taxonomy: keep only top-N noun classes")
    parser.add_argument("--pilot-mode", choices=["exclude", "other"], default="exclude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/ego4d_lta/index")
    args = parser.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
