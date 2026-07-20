"""Task 2 -- rebuild the GoalStep verb/noun taxonomy + action registry from the
Phase-2 class tables, in the exact schema FHO-LTA uses.

Inputs (Phase 2 outputs of ``parse_goalstep_to_verbnoun.py``):
  * ``verb_classes.csv`` / ``noun_classes.csv`` -- one row per class:
    ``class_id, class_key, members, segment_count``.
  * ``goalstep_parsed_segments.csv`` -- one row per step/substep instance with
    the per-instance ``verb_class`` / ``noun_class`` assignment already made at
    parse time. That assignment is REUSED when present; instances lacking one
    are re-mapped through the class ``members`` lists, and anything still
    unmatched is logged as OTHER.

Outputs (into ``--output-dir``):
  * ``goalstep_verbnoun_taxonomy.json`` -- ``{"verbs": [...], "nouns": [...]}``,
    list index == class id. Identical schema to ``fho_lta_taxonomy.json``.
  * ``goalstep_step_labels.csv`` -- per-instance
    ``video_uid, split, level, start_time, end_time, verb_label, noun_label``
    (integer taxonomy ids); this is what ``build_goalstep_z1_index.py`` consumes.
  * ``action_registry.json`` -- dense (verb, noun) -> action id over the
    combinations seen in TRAIN only (identical rule to FHO
    ``register_action_labels``).
  * ``taxonomy_build_stats.json`` -- counts + OTHER breakdown.

Usage:
    python scripts/step1/goalstep/build_goalstep_taxonomy.py \
        --class-dir outputs/goalstep/taxonomy \
        --output-dir outputs/goalstep/taxonomy_rebuilt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pandas as pd  # noqa: E402

from ego.common.io import ensure_dir, write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.datasets.label_mapping import build_label_mapping  # noqa: E402

PHASE = "GoalStepTaxonomy"
OTHER = "OTHER"


def _fho_class_name(class_key: str, members: str) -> str:
    """Render a class the way ``fho_lta_taxonomy.json`` does: bare key when the
    class has a single member, ``key_(m1,_m2)`` when it merged synonyms."""
    extras = [m for m in _members(members) if m != class_key]
    if not extras:
        return class_key
    return f"{class_key}_({',_'.join(sorted(extras))})"


def _members(members: str) -> list[str]:
    """Phase 2 writes the member list ``;``-separated (see
    ``parse_goalstep_to_verbnoun.py``); tolerate ``,`` too."""
    raw = str(members).replace(",", ";")
    return [m.strip() for m in raw.split(";") if m.strip()]


def load_classes(path: Path) -> tuple[list[str], dict[str, int]]:
    """Return (taxonomy names ordered by class_id, member/key -> class_id)."""
    df = pd.read_csv(path).sort_values("class_id").reset_index(drop=True)
    if list(df["class_id"]) != list(range(len(df))):
        raise ValueError(f"{path}: class_id column must be a dense 0..N-1 range")
    names = [_fho_class_name(r.class_key, r.members) for r in df.itertuples()]
    lookup: dict[str, int] = {}
    for r in df.itertuples():
        lookup[r.class_key] = int(r.class_id)
        for member in _members(r.members):
            lookup.setdefault(member, int(r.class_id))
    return names, lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-dir", default="outputs/goalstep/taxonomy",
                        help="Directory holding verb_classes.csv / noun_classes.csv / goalstep_parsed_segments.csv")
    parser.add_argument("--segments-csv", default=None, help="Override path to goalstep_parsed_segments.csv")
    parser.add_argument("--level", choices=["step", "substep", "both"], default="both",
                        help="Annotation level to emit (default 'both' == the level the classes were built at)")
    parser.add_argument("--output-dir", default="outputs/goalstep/taxonomy")
    args = parser.parse_args()

    class_dir = Path(args.class_dir)
    out_dir = ensure_dir(args.output_dir)

    verbs, verb_lookup = load_classes(class_dir / "verb_classes.csv")
    nouns, noun_lookup = load_classes(class_dir / "noun_classes.csv")
    step_log(1, PHASE, f"N_verb={len(verbs)} N_noun={len(nouns)}")

    taxonomy_path = out_dir / "goalstep_verbnoun_taxonomy.json"
    write_json(taxonomy_path, {"verbs": verbs, "nouns": nouns})
    step_log(1, PHASE, f"Wrote {taxonomy_path} (FHO schema: verbs[]/nouns[], index == class id)")

    seg_path = Path(args.segments_csv or class_dir / "goalstep_parsed_segments.csv")
    seg = pd.read_csv(seg_path, low_memory=False)
    step_log(1, PHASE, f"Loaded {len(seg)} parsed step instances from {seg_path}")
    if args.level != "both":
        seg = seg[seg["level"] == args.level].reset_index(drop=True)
        step_log(1, PHASE, f"--level {args.level}: {len(seg)} instances kept")

    # Long-tail pruning and OTHER instances must never reach the training index.
    # A pruned instance can still have a surviving verb AND a surviving noun
    # (its *combination* is what was rare), so filtering by the class lookups
    # alone would silently let it through -- drop it explicitly here.
    for flag, why in (("is_other", "unparsed (OTHER)"), ("is_pruned", "long-tail pruned")):
        if flag in seg.columns:
            before = len(seg)
            seg = seg[seg[flag].fillna(0).astype(int) == 0].reset_index(drop=True)
            if before != len(seg):
                step_log(1, PHASE, f"dropped {before - len(seg)} {why} instances -> {len(seg)} kept")

    # Reuse the per-instance assignment made at parse time; fall back to the
    # members lookup for anything that only carries raw verb/noun text.
    reused = remapped = 0
    other_rows: list[dict] = []
    other_reasons: Counter = Counter()
    records: list[dict] = []
    for row in seg.itertuples():
        v_key, n_key = str(getattr(row, "verb_class", "")), str(getattr(row, "noun_class", ""))
        source = "parsed"
        if v_key not in verb_lookup or n_key not in noun_lookup:
            source = "remapped"
            v_key = str(getattr(row, "verb_raw", "")).strip().lower()
            n_key = str(getattr(row, "noun_raw", "")).strip().lower()
        v_id, n_id = verb_lookup.get(v_key), noun_lookup.get(n_key)
        if v_id is None or n_id is None:
            other_reasons["verb" if v_id is None else "noun"] += 1
            other_rows.append({
                "video_uid": row.video_uid, "split": row.split, "level": row.level,
                "start_time": row.start_time, "end_time": row.end_time,
                "step_category": getattr(row, "step_category", ""),
                "verb_class": getattr(row, "verb_class", ""), "noun_class": getattr(row, "noun_class", ""),
                "reason": f"unmapped_verb={v_id is None} unmapped_noun={n_id is None}",
            })
            continue
        reused += source == "parsed"
        remapped += source == "remapped"
        records.append({
            "video_uid": row.video_uid, "split": row.split, "level": row.level,
            "start_time": float(row.start_time), "end_time": float(row.end_time),
            "verb_label": v_id, "noun_label": n_id,
        })

    labels = pd.DataFrame(records)
    n_other = len(other_rows)
    step_log(
        1, PHASE,
        f"Mapped instances: {len(labels)} (reused parse-time assignment={reused}, re-mapped via members={remapped}); "
        f"OTHER/unmapped={n_other} ({100.0 * n_other / max(1, len(seg)):.2f}%)",
    )
    if other_rows:
        other_path = out_dir / "taxonomy_other_segments.csv"
        pd.DataFrame(other_rows).to_csv(other_path, index=False)
        step_log(1, PHASE, f"Logged {n_other} OTHER instances to {other_path}")

    labels_path = out_dir / "goalstep_step_labels.csv"
    labels.to_csv(labels_path, index=False)
    step_log(1, PHASE, f"Wrote {labels_path} ({len(labels)} rows)")

    # action registry: TRAIN-seen (verb, noun) combinations only -- same rule as FHO.
    train_labels = labels[labels["split"] == "train"]
    mapping = build_label_mapping(list(zip(train_labels["verb_label"], train_labels["noun_label"])))
    step_log(
        1, PHASE,
        f"Registered (train-seen): verbs={mapping.num_verbs} nouns={mapping.num_nouns} "
        f"actions={mapping.num_actions}",
    )
    registry_path = out_dir / "action_registry.json"
    write_json(registry_path, {
        "num_verbs": mapping.num_verbs,
        "num_nouns": mapping.num_nouns,
        "num_actions": mapping.num_actions,
        "verb_classes": {str(k): v for k, v in mapping.verb_classes.items()},
        "noun_classes": {str(k): v for k, v in mapping.noun_classes.items()},
        "action_classes": {f"{v}|{n}": a for (v, n), a in mapping.action_classes.items()},
    })
    step_log(1, PHASE, f"Wrote {registry_path}")

    val_labels = labels[labels["split"] == "val"]
    known = set(mapping.action_classes.keys())
    val_unseen = sum(1 for p in zip(val_labels["verb_label"], val_labels["noun_label"]) if p not in known)
    write_json(out_dir / "taxonomy_build_stats.json", {
        "level": args.level,
        "num_verbs_taxonomy": len(verbs),
        "num_nouns_taxonomy": len(nouns),
        "num_verbs_registered": mapping.num_verbs,
        "num_nouns_registered": mapping.num_nouns,
        "num_actions_registered": mapping.num_actions,
        "instances_total": int(len(seg)),
        "instances_mapped": int(len(labels)),
        "instances_other": n_other,
        "other_rate": n_other / max(1, len(seg)),
        "reused_parse_time_assignment": reused,
        "remapped_via_members": remapped,
        "train_instances": int(len(train_labels)),
        "val_instances": int(len(val_labels)),
        "val_instances_with_unseen_action_pair": val_unseen,
    })
    step_log(1, PHASE, f"val instances whose (verb,noun) pair is unseen in train: {val_unseen}")


if __name__ == "__main__":
    main()
