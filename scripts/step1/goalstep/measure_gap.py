"""Measure the train-vs-val generalization gap of a finished GoalStep probe run.

The per-epoch history only reports validation, so a run that memorizes its
training videos looks identical to one that simply cannot fit -- both show a
flat val curve. This loads a run's ``best.pt`` and reports train and val
accuracy side by side, which separates the two.

Usage:
    python scripts/step1/goalstep/measure_gap.py outputs/goalstep/sweep/s1_d1 [...]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ego.step1_action_anticipation.data.collator import anticipation_collate  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402

CACHE = "/mnt/nvme/migration/jihun/datasets/Ego4D/goalstep_feature_cache_jihun2"
HEADS = ("verb", "noun", "action")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    top1 = (logits.argmax(-1) == labels).float().mean().item() * 100
    top5 = (logits.topk(5, -1).indices == labels[:, None]).any(-1).float().mean().item() * 100
    return top1, top5


@torch.no_grad()
def _evaluate(model, cache_dir: str, sample_ids: list[str], heads: list[str], device) -> dict:
    loader = DataLoader(
        FeatureCacheDataset(sample_ids, cache_dir), batch_size=64, shuffle=False,
        collate_fn=anticipation_collate, num_workers=6,
    )
    logits = {h: [] for h in heads}
    labels = {h: [] for h in heads}
    for batch in loader:
        out = model(batch["video"].to(device))
        for h in heads:
            logits[h].append(out[h].float().cpu())
            labels[h].append(batch[f"{h}_id"])
    return {h: _accuracy(torch.cat(logits[h]), torch.cat(labels[h])) for h in heads}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A run must be scored on the cache it was trained on. f_fps uses the
    # l_obs=4.0 re-extraction, so a hardcoded cache silently mixes 3.5s features
    # into a 4.0s model and reports a meaningless number.
    id_cache: dict[str, tuple[list[str], list[str]]] = {}

    def ids_for(cache_dir: str) -> tuple[list[str], list[str]]:
        if cache_dir not in id_cache:
            tr = [Path(p).stem for p in glob.glob(f"{cache_dir}/train/*.pt")]
            random.Random(args.seed).shuffle(tr)
            va = [Path(p).stem for p in glob.glob(f"{cache_dir}/val/*.pt")]
            id_cache[cache_dir] = (tr[: args.train_samples], va)
        return id_cache[cache_dir]

    print(f"{'run':18s} {'heads':6s} {'ep':>3s} | {'TRAIN top1':>10s} {'top5':>7s} | "
          f"{'VAL top1':>9s} {'top5':>7s} | {'gap@5':>7s}")
    print("-" * 92)

    for run_dir in args.run_dirs:
        run = Path(run_dir)
        ckpt = torch.load(run / "best.pt", map_location="cpu")
        config = yaml.safe_load(open(run / "config_resolved.yaml", encoding="utf-8"))
        heads = config["training"].get("train_heads", list(HEADS))
        action_only = heads == ["action"]
        num_classes = ckpt["num_classes"]

        model = AnticipationHead(
            num_verb_classes=0 if action_only else num_classes["verb"],
            num_noun_classes=0 if action_only else num_classes["noun"],
            num_action_classes=num_classes["action"],
            embed_dim=1024,
            num_heads=config["model"]["classifier"].get("num_heads", 16),
            depth=config["model"]["classifier"].get("num_probe_blocks", 4),
            repository_dir=config["model"].get("repository_dir"),
        ).to(device).eval()
        model.load_state_dict(ckpt["model_state"])

        cache = str(Path(config["dataset"].get("feature_cache_dir", CACHE)).resolve())
        train_ids, val_ids = ids_for(cache)
        tr = _evaluate(model, f"{cache}/train", train_ids, heads, device)
        va = _evaluate(model, f"{cache}/val", val_ids, heads, device)
        gap = tr["action"][1] - va["action"][1]
        print(f"{run.name:18s} {'A' if action_only else 'VNA':6s} {ckpt['epoch']:3d} | "
              f"{tr['action'][0]:10.2f} {tr['action'][1]:7.2f} | "
              f"{va['action'][0]:9.2f} {va['action'][1]:7.2f} | {gap:7.2f}")

        (run / "gap.json").write_text(json.dumps(
            {"epoch": ckpt["epoch"], "heads": heads,
             "train": {h: {"top1": tr[h][0], "top5": tr[h][1]} for h in heads},
             "val": {h: {"top1": va[h][0], "top5": va[h][1]} for h in heads},
             "cache": cache, "n_train": len(train_ids), "n_val": len(val_ids),
             "action_top5_gap": gap}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
