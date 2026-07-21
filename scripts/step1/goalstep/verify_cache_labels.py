"""Verify that cached feature labels still agree with the index + action registry.

Why this exists: ``extract_features.py`` bakes ``verb_id``/``noun_id``/``action_id``
into every ``.pt``. If the taxonomy is re-pruned afterwards (this project went
verb 98 / noun 188 / action 390 -> 81 / 140 / 293), a stale cache trains on
silently wrong labels with no error -- see
``docs/experiments/2026-07-20_step1-goalstep-training-handoff.md`` §4-1.

Because a cache can also be assembled across several extraction runs (resume
skips anything already on disk), this reports old vs new files separately using
``--cutoff``, so a partially-stale cache cannot hide behind a clean majority.

Run this before training whenever the registry or index may have moved.

Usage:
    python scripts/step1/goalstep/verify_cache_labels.py --config configs/step1/goalstep/z1_jihun2.yaml
    python scripts/step1/goalstep/verify_cache_labels.py --config ... --full
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ego.common.config import load_config, require  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ego4d import z1_sample_id  # noqa: E402


def _load_registry(path: Path) -> tuple[dict, dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    verbs = {int(k): v for k, v in data["verb_classes"].items()}
    nouns = {int(k): v for k, v in data["noun_classes"].items()}
    actions = {}
    for key, action_id in data["action_classes"].items():
        v, n = key.split("|")
        actions[(int(v), int(n))] = action_id
    return verbs, nouns, actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--samples", type=int, default=2000, help="per split; ignored with --full")
    parser.add_argument("--full", action="store_true", help="check every cached sample (reads the whole cache)")
    parser.add_argument("--cutoff", default=None,
                        help="ISO time splitting 'old' from 'new' cache files, e.g. '2026-07-20T13:06'")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_config(args.config)
    index_dir = expand_path(require(config, "dataset.index_dir"))
    cache_dir = expand_path(require(config, "dataset.feature_cache_dir"))
    verbs, nouns, actions = _load_registry(index_dir / "action_registry.json")
    cutoff = datetime.fromisoformat(args.cutoff).timestamp() if args.cutoff else None

    print(f"registry: verb={len(verbs)} noun={len(nouns)} action={len(actions)}")
    print(f"cache:    {cache_dir}")

    failed = False
    for split in args.splits:
        index_path = index_dir / f"{split}.parquet"
        if not index_path.is_file():
            index_path = index_dir / f"{split}.csv"
        df = (pd.read_parquet(index_path) if index_path.suffix == ".parquet"
              else pd.read_csv(index_path)).reset_index(drop=True)

        positions = list(range(len(df)))
        if not args.full:
            random.Random(args.seed).shuffle(positions)

        # [checked, mismatched] per age bucket
        buckets = {"old": [0, 0], "new": [0, 0]}
        missing = 0
        examples: list[str] = []

        for position in positions:
            if not args.full and sum(b[0] for b in buckets.values()) >= args.samples:
                break
            row = df.iloc[position]
            path = cache_dir / split / f"{z1_sample_id(row['clip_uid'], position)}.pt"
            if not path.is_file():
                missing += 1
                continue

            record = torch.load(path, map_location="cpu")
            verb_raw, noun_raw = int(row["verb_label"]), int(row["noun_label"])
            expected = (verbs.get(verb_raw), nouns.get(noun_raw), actions.get((verb_raw, noun_raw)))
            got = (record["verb_id"], record["noun_id"], record["action_id"])

            age = "new" if cutoff is None or path.stat().st_mtime >= cutoff else "old"
            buckets[age][0] += 1
            if expected != got:
                buckets[age][1] += 1
                if len(examples) < 5:
                    examples.append(f"    {path.name} [{age}] expected={expected} got={got}")

        total, bad = sum(b[0] for b in buckets.values()), sum(b[1] for b in buckets.values())
        print(f"\n[{split}] index rows={len(df)}  checked={total}  not-yet-cached={missing}")
        for age, (checked, mismatched) in buckets.items():
            if checked:
                mark = "MISMATCH" if mismatched else "ok"
                print(f"    {age:3s} cache: {checked:6d} checked, {mismatched:5d} mismatched   {mark}")
        for line in examples:
            print(line)
        if bad:
            failed = True

    if failed:
        print("\nFAIL — cached labels disagree with the registry. Delete the cache and re-extract:")
        print(f"    rm -rf {cache_dir}")
        return 1
    print("\nPASS — every checked sample's cached labels match the index + registry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
