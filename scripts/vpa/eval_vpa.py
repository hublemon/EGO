"""Task 3 -- VPA evaluation: SR / mAcc / mIoU (+ bootstrap CIs), for T=3 and T=4.

Metrics (COIN/VPA definitions):
  SR   (Success Rate) : fraction of samples whose predicted T-step sequence
                        matches the ground truth EXACTLY, order included.
  mAcc (mean Accuracy): position-sensitive -- mean over all positions of
                        1[pred_i == gt_i] (micro-averaged across positions).
  mIoU (mean IoU)     : order-agnostic -- per-sample IoU of the predicted vs GT
                        step SETS, averaged over samples.

Prediction normalisation to the candidate vocabulary is logged: (1) exact match
after normalisation, (2) fuzzy nearest by string similarity, (3) unmatched.

Two entry points:
  * score a predictions file:  --pred preds.json --gt goalstep_vpa_T3.json ...
  * generate+score sanity baselines from train frequencies:
        --make-baselines --train-json ... --val-json ...

preds.json format: { "<sample_id>": ["label1", ..., "labelT"], ... }

Usage:
    python scripts/vpa/eval_vpa.py --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
        --vocab outputs/goalstep/vpa/candidate_vocab.json --pred preds_frontier_T3.json \
        --split test --output-dir outputs/goalstep/vpa/runs/frontier
"""

from __future__ import annotations

import argparse
import collections
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpa_common import dump_json, load_json, normalize_label  # noqa: E402

try:
    import numpy as np
except ImportError:  # bootstrap needs numpy; fall back to no-CI if absent
    np = None


# ----------------------------- normalisation -----------------------------
def map_to_vocab(label, vocab_set, vocab_list, fuzzy_cutoff=0.6):
    n = normalize_label(label)
    if n in vocab_set:
        return n, "exact"
    m = difflib.get_close_matches(n, vocab_list, n=1, cutoff=fuzzy_cutoff)
    if m:
        return m[0], "fuzzy"
    return n, "unmatched"


def fix_length(seq, T):
    seq = list(seq)[:T]
    seq += [""] * (T - len(seq))  # pad short predictions with a non-matching blank
    return seq


# ------------------------------- metrics --------------------------------
def per_sample_scores(gt_future, pred_future, T):
    """Return (success:0/1, correct_positions:int, iou:float) for one sample."""
    success = int(gt_future == pred_future)
    correct = sum(1 for a, b in zip(gt_future, pred_future) if a == b and a != "")
    gset, pset = set(x for x in gt_future if x), set(x for x in pred_future if x)
    union = gset | pset
    iou = (len(gset & pset) / len(union)) if union else 0.0
    return success, correct, iou


def aggregate(rows, T):
    """rows: list of (success, correct_positions, iou). Returns SR, mAcc, mIoU."""
    n = len(rows)
    if n == 0:
        return {"SR": 0.0, "mAcc": 0.0, "mIoU": 0.0, "n": 0}
    sr = sum(r[0] for r in rows) / n
    macc = sum(r[1] for r in rows) / (n * T)
    miou = sum(r[2] for r in rows) / n
    return {"SR": sr, "mAcc": macc, "mIoU": miou, "n": n}


def bootstrap_ci(rows, T, n_boot=1000, seed=0, alpha=0.05):
    if np is None or len(rows) == 0:
        return {}
    rng = np.random.default_rng(seed)
    arr = np.array(rows, dtype=float)  # [n,3] = success, correct, iou
    n = len(arr)
    srs, maccs, mious = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = arr[idx]
        srs.append(s[:, 0].mean())
        maccs.append(s[:, 1].sum() / (n * T))
        mious.append(s[:, 2].mean())
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    def ci(x):
        return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]
    return {"SR": ci(srs), "mAcc": ci(maccs), "mIoU": ci(mious)}


# ------------------------------ scoring ---------------------------------
def score(gt_samples, preds, vocab, split, T, n_boot, seed):
    vocab_set, vocab_list = set(vocab), list(vocab)
    rows, method_counts, missing = [], collections.Counter(), 0
    detail = []
    for s in gt_samples:
        if split != "all" and s["eval_split"] != split:
            continue
        sid = s["sample_id"]
        gt_future = fix_length(s["future_steps"], T)
        raw = preds.get(sid)
        if raw is None:
            missing += 1
            raw = []
        mapped = []
        for lab in fix_length(raw, T):
            m, how = map_to_vocab(lab, vocab_set, vocab_list)
            method_counts[how] += 1
            mapped.append(m if m in vocab_set else m)  # keep mapped/normalized string
        row = per_sample_scores(gt_future, mapped, T)
        rows.append(row)
        detail.append({"sample_id": sid, "gt": gt_future, "pred": mapped,
                       "success": row[0], "iou": round(row[2], 3)})
    metrics = aggregate(rows, T)
    metrics["ci95"] = bootstrap_ci(rows, T, n_boot=n_boot, seed=seed)
    metrics["norm_methods"] = dict(method_counts)
    metrics["missing_preds"] = missing
    return metrics, detail


# ---------------------- sanity (most-probable) baselines -----------------
def _train_action_lut(parsed_csv, level):
    import csv as _csv
    lut = {}
    with open(parsed_csv) as f:
        for r in _csv.DictReader(f):
            if r["split"] != "train" or r["level"] != level:
                continue
            dropped = int(r["is_other"]) or int(r.get("is_pruned", 0) or 0)
            lut[(r["video_uid"], round(float(r["start_time"]), 3))] = (
                None if dropped else f"{r['verb_class']} {r['noun_class']}")
    return lut


def train_label_freq(train_json, level, label_field, essential_only, label_mode, parsed_csv):
    from vpa_common import video_steps, label_from_segment  # local import
    data = load_json(train_json)
    lut = _train_action_lut(parsed_csv, level) if label_mode == "action" else {}
    def lab_of(uid, s):
        if label_mode == "action":
            st = s.get("start_time")
            return None if st is None else lut.get((uid, round(float(st), 3)))
        return label_from_segment(s, label_field)
    global_freq = collections.Counter()
    per_goal = collections.defaultdict(collections.Counter)
    for v in data["videos"]:
        gcat = v.get("goal_category", "")
        for s in video_steps(v, level, essential_only):
            lab = lab_of(v["video_uid"], s)
            if lab:
                global_freq[lab] += 1
                per_goal[gcat][lab] += 1
    return global_freq, per_goal


def make_baseline_preds(gt_samples, uid_to_goalcat, global_freq, per_goal, T, kind):
    top_global = [lab for lab, _ in global_freq.most_common()]
    preds = {}
    for s in gt_samples:
        if kind == "global":
            ranked = top_global
        else:  # goal-conditioned, fall back to global
            gcat = uid_to_goalcat.get(s["video_uid"], "")
            ranked = [lab for lab, _ in per_goal.get(gcat, collections.Counter()).most_common()] or top_global
        preds[s["sample_id"]] = ranked[:T]
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="goalstep_vpa_T{T}.json from build_goalstep_vpa.py")
    p.add_argument("--vocab", required=True, help="candidate_vocab.json")
    p.add_argument("--pred", default=None, help="predictions json {sample_id:[labels]}")
    p.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/goalstep/vpa/runs/eval")
    p.add_argument("--run-name", default="eval")
    # sanity baselines
    p.add_argument("--make-baselines", action="store_true",
                   help="generate+score Most-Probable and Most-Probable-w-Goal from train freq")
    p.add_argument("--train-json", default="data/Ego4D/v2/annotations/goalstep_train.json")
    p.add_argument("--val-json", default="data/Ego4D/v2/annotations/goalstep_val.json")
    p.add_argument("--parsed-segments", default="outputs/goalstep/taxonomy/goalstep_parsed_segments.csv")
    args = p.parse_args()

    gt_samples = load_json(args.gt)
    vocab = load_json(args.vocab)["labels"]
    meta = load_json(args.vocab)
    T = gt_samples[0]["horizon"] if gt_samples else 3
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.pred:
        preds = load_json(args.pred)
        metrics, detail = score(gt_samples, preds, vocab, args.split, T, args.n_boot, args.seed)
        results[args.run_name] = metrics
        dump_json(out_dir / f"{args.run_name}_detail_T{T}.json", detail)
        _print_metrics(f"{args.run_name} (T={T}, {args.split})", metrics)

    if args.make_baselines:
        gfreq, pgoal = train_label_freq(args.train_json, meta["level"], meta["label_field"],
                                        meta["essential_only"], meta.get("label_mode", "step_category"), args.parsed_segments)
        uid2cat = {v["video_uid"]: v.get("goal_category", "") for v in load_json(args.val_json)["videos"]}
        for kind, name in [("global", "most_probable"), ("goal", "most_probable_goal")]:
            preds = make_baseline_preds(gt_samples, uid2cat, gfreq, pgoal, T, kind)
            dump_json(out_dir / f"preds_{name}_T{T}.json", preds)
            metrics, _ = score(gt_samples, preds, vocab, args.split, T, args.n_boot, args.seed)
            results[name] = metrics
            _print_metrics(f"{name} (T={T}, {args.split})", metrics)

    dump_json(out_dir / f"metrics_T{T}_{args.split}.json",
              {"T": T, "split": args.split, "gt": args.gt, "results": results})
    print(f"\nwrote metrics to {out_dir}/metrics_T{T}_{args.split}.json")


def _print_metrics(title, m):
    ci = m.get("ci95", {})
    def fmt(k):
        v = m[k] * 100
        c = ci.get(k)
        return f"{v:5.1f}" + (f" [{c[0]*100:.1f},{c[1]*100:.1f}]" if c else "")
    print(f"\n{title}  (n={m['n']}, missing={m.get('missing_preds',0)})")
    print(f"  SR   = {fmt('SR')}")
    print(f"  mAcc = {fmt('mAcc')}")
    print(f"  mIoU = {fmt('mIoU')}")
    if m.get("norm_methods"):
        print(f"  norm: {m['norm_methods']}")


if __name__ == "__main__":
    main()
