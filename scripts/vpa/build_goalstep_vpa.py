"""Task 2 -- build GoalStep VPA samples (val 134 videos ONLY).

For every val video (each video = one cooking goal) we slide a stop point t over
step boundaries. At each t with >= T valid future steps we emit one VPA sample:
observation history (goal text + steps observed up to t) -> next T step labels.

LABEL MODE (research-aligned):
  * --label-mode action  (default) -- each step's label is the "<verb> <noun>"
    action CLASS from the Phase-2 taxonomy (goalstep_parsed_segments.csv), i.e.
    the SAME label space the Step-1 anticipation model is trained/scored on.
    Steps whose parse was OTHER (no verb/noun) are dropped.
  * --label-mode step_category -- raw normalized step_category strings (the
    original controlled 514-step labels). Kept for inspection / ablation.

Sample schema:
  { "sample_id", "video_uid", "goal_text", "observed_steps":[label,...],
    "future_steps":[label x T], "horizon": T, "eval_split": "dev"|"test" }

Videos are re-split into dev/test at the VIDEO level (default 50/50, seeded).
A fixed candidate-label vocabulary is written so free-text predictions score
fairly (GT labels are always in-vocab).

Usage:
    python scripts/vpa/build_goalstep_vpa.py \
        --val-json data/Ego4D/v2/annotations/goalstep_val.json \
        --parsed-segments outputs/goalstep/taxonomy/goalstep_parsed_segments.csv \
        --horizons 3 4 --level step --label-mode action --output-dir outputs/goalstep/vpa
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpa_common import (  # noqa: E402
    dump_json, goal_text_of, iter_goalstep_videos, label_from_segment, load_json, normalize_label, video_steps,
)


def load_action_lookup(parsed_csv, level):
    """Map (video_uid, round(start,3)) -> '<verb_class> <noun_class>' for val
    segments at the chosen level. OTHER *and* long-tail-PRUNED segments map to
    None so they never enter a VPA sample (same exclusion the training index uses)."""
    lut = {}
    with open(parsed_csv) as f:
        for r in csv.DictReader(f):
            if r["split"] != "val" or r["level"] != level:
                continue
            key = (r["video_uid"], round(float(r["start_time"]), 3))
            dropped = int(r["is_other"]) or int(r.get("is_pruned", 0) or 0)
            lut[key] = None if dropped else f"{r['verb_class']} {r['noun_class']}"
    return lut


def make_label_fn(mode, label_field, action_lut):
    if mode == "step_category":
        return lambda uid, seg: label_from_segment(seg, label_field)
    def action_label(uid, seg):
        st = seg.get("start_time")
        if st is None:
            return None
        return action_lut.get((uid, round(float(st), 3)))
    return action_label


def build_samples(videos, split_of, level, essential_only, horizon, min_observed, label_fn, allowed=None):
    samples = []
    for v in videos:
        uid = v["video_uid"]
        steps = video_steps(v, level, essential_only)
        labels = [label_fn(uid, s) for s in steps]
        # Drop empty / OTHER / long-tail-pruned, and -- under the strict
        # train-scoped vocabulary -- any action the model could never predict
        # because it never occurs in train at this level. Same "restrict to
        # train-seen" rule the Z=1 training index applies.
        labels = [x for x in labels if x and (allowed is None or x in allowed)]
        goal = goal_text_of(v)
        for k in range(min_observed, len(labels) - horizon + 1):
            samples.append({
                "sample_id": f"{uid}_t{k}_T{horizon}",
                "video_uid": uid,
                "goal_text": goal,
                "observed_steps": labels[:k],
                "future_steps": labels[k:k + horizon],
                "horizon": horizon,
                "eval_split": split_of[uid],
            })
    return samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-json", default="data/Ego4D/v2/annotations/goalstep_val.json")
    p.add_argument("--parsed-segments", default="outputs/goalstep/taxonomy/goalstep_parsed_segments.csv",
                   help="Phase-2 parsed segments (verb_class/noun_class per step) for --label-mode action")
    p.add_argument("--output-dir", default="outputs/goalstep/vpa")
    p.add_argument("--horizons", type=int, nargs="+", default=[3, 4])
    p.add_argument("--level", choices=["step", "substep"], default="step")
    p.add_argument("--label-mode", choices=["action", "step_category"], default="action",
                   help="action = '<verb> <noun>' taxonomy class (training label space); step_category = raw label")
    p.add_argument("--label-field", choices=["step_category", "step_description"], default="step_category",
                   help="raw field used only when --label-mode step_category")
    p.add_argument("--essential-only", dest="essential_only", action="store_true", default=True)
    p.add_argument("--include-nonessential", dest="essential_only", action="store_false")
    p.add_argument("--vocab-scope", choices=["train", "val"], default="train",
                   help="source of the candidate action vocabulary. 'train' (default, strict) uses the "
                        "TRAIN-seen action space -- the labels a trained model can actually predict, and "
                        "it never reveals which actions occur in val. 'val' derives it from the eval set "
                        "itself (leaks the val label set; kept only for inspection).")
    p.add_argument("--min-observed", type=int, default=1)
    p.add_argument("--dev-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-videos", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    videos = list(iter_goalstep_videos(load_json(args.val_json)))
    if args.limit_videos:
        videos = videos[:args.limit_videos]
    print(f"[info] val videos used: {len(videos)}  label_mode={args.label_mode}")

    action_lut = load_action_lookup(args.parsed_segments, args.level) if args.label_mode == "action" else {}
    label_fn = make_label_fn(args.label_mode, args.label_field, action_lut)

    uids = sorted(v["video_uid"] for v in videos)
    rng = random.Random(args.seed)
    rng.shuffle(uids)
    n_dev = round(len(uids) * args.dev_frac)
    dev_uids = set(uids[:n_dev])
    split_of = {u: ("dev" if u in dev_uids else "test") for u in uids}

    # Candidate vocabulary. STRICT default: the train-seen action space, read
    # straight from the parsed segments -- so the prompt never discloses which
    # actions appear in val. (With --prune-on train upstream, every surviving val
    # action is train-seen, so GT stays in-vocab.)
    vocab = set()
    if args.label_mode == "action" and args.vocab_scope == "train":
        with open(args.parsed_segments) as f:
            for r in csv.DictReader(f):
                if r["split"] != "train" or r["level"] != args.level:
                    continue
                if int(r["is_other"]) or int(r.get("is_pruned", 0) or 0):
                    continue
                vocab.add(f"{r['verb_class']} {r['noun_class']}")
    else:
        for v in videos:
            for s in video_steps(v, args.level, args.essential_only):
                lab = label_fn(v["video_uid"], s)
                if lab:
                    vocab.add(lab)
    vocab = sorted(vocab)
    dump_json(out_dir / "candidate_vocab.json",
              {"level": args.level, "label_mode": args.label_mode, "label_field": args.label_field,
               "essential_only": args.essential_only, "vocab_scope": args.vocab_scope,
               "size": len(vocab), "labels": vocab})

    manifest = {"val_json": args.val_json, "level": args.level, "label_mode": args.label_mode,
                "label_field": args.label_field, "essential_only": args.essential_only,
                "dev_frac": args.dev_frac, "seed": args.seed, "min_observed": args.min_observed,
                "n_videos": len(videos), "n_dev_videos": len(dev_uids),
                "n_test_videos": len(uids) - len(dev_uids), "vocab_size": len(vocab), "horizons": {}}

    for T in args.horizons:
        samples = build_samples(videos, split_of, args.level, args.essential_only, T, args.min_observed,
                                label_fn, allowed=set(vocab) if args.vocab_scope == "train" else None)
        out_path = out_dir / f"goalstep_vpa_T{T}.json"
        dump_json(out_path, samples)
        by_split = collections.Counter(s["eval_split"] for s in samples)
        per_vid = collections.Counter(s["video_uid"] for s in samples)
        fut = collections.Counter(lab for s in samples for lab in s["future_steps"])
        manifest["horizons"][str(T)] = {"file": str(out_path), "n_samples": len(samples),
                                        "dev": by_split["dev"], "test": by_split["test"],
                                        "avg_samples_per_video": round(len(samples) / max(1, len(per_vid)), 2)}
        print(f"\n=== T={T} ===")
        print(f"  samples: {len(samples)}  (dev {by_split['dev']} / test {by_split['test']})")
        print(f"  videos producing samples: {len(per_vid)}  avg {len(samples)/max(1,len(per_vid)):.1f}/video")
        print(f"  candidate vocab size ({args.label_mode}): {len(vocab)}")
        print(f"  top future labels: {[l for l,_ in fut.most_common(6)]}")
        print(f"  wrote {out_path}")

    dump_json(out_dir / "vpa_manifest.json", manifest)
    print(f"\nwrote {out_dir/'candidate_vocab.json'} and {out_dir/'vpa_manifest.json'}")


if __name__ == "__main__":
    main()
