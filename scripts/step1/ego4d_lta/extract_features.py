"""Cache frozen V-JEPA2 backbone features for an Ego4D LTA Z=1 index split.

Resamples each [obs_start_sec, obs_end_sec] window to a fixed frame count
(uniformly by default, or with a configured adaptive sampler), runs the frozen
encoder+predictor (the predictor receiving the same
"tau_a-seconds-ahead mask token" input it uses for EK100), and caches the
resulting token sequence per sample_id so
``train_lta_z1.py`` never has to touch raw video. Reuses
``ego.step1_action_anticipation.data.feature_cache.extract_and_cache_features``;
adaptive samples add audited probe-only time metadata to the otherwise shared
cache schema.

Usage:
    python scripts/step1/ego4d_lta/extract_features.py \
        --config configs/step1/ego4d_lta/full.yaml --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ego4d import Ego4DLTADataset  # noqa: E402
from ego.datasets.label_mapping import LabelMapping  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import extract_and_cache_features  # noqa: E402
from ego.step1_action_anticipation.data.transforms import build_transform  # noqa: E402
from ego.step1_action_anticipation.models import load_vjepa2_backbone  # noqa: E402


def _read_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _load_registry(path: Path) -> LabelMapping:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    verb_classes = {int(k): v for k, v in data["verb_classes"].items()}
    noun_classes = {int(k): v for k, v in data["noun_classes"].items()}
    action_classes = {}
    for key, action_id in data["action_classes"].items():
        v, n = key.split("|")
        action_classes[(int(v), int(n))] = action_id
    return LabelMapping(verb_classes=verb_classes, noun_classes=noun_classes, action_classes=action_classes)


def _find_index_file(index_dir: Path, split: str) -> Path:
    for ext in (".parquet", ".csv"):
        p = index_dir / f"{split}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No {split}.parquet or {split}.csv found under {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    # "val" exists for GoalStep (scripts/step1/goalstep), whose index is a plain
    # train/val pair rather than FHO's train/dev/heldout re-split.
    parser.add_argument("--split", choices=["train", "dev", "heldout", "val"], default="train")
    parser.add_argument("--cache-dir", default=None, help="Override dataset.feature_cache_dir from config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    index_dir = expand_path(require(config, "dataset.index_dir"))
    index_path = _find_index_file(index_dir, args.split)
    index_df = _read_index(index_path)
    step_log(1, "ExtractFeatures", f"{args.split}: {len(index_df)} samples from {index_path}")

    label_mapping = _load_registry(index_dir / "action_registry.json")

    tau_a = get(config, "dataset.tau_a", 1.0)
    frames_per_clip = require(config, "dataset.frames_per_clip")
    resolution = require(config, "dataset.resolution")
    video_root = expand_path(require(config, "dataset.video_root"))
    video_source = get(config, "dataset.video_source", "clips")
    sampling_cfg = get(config, "dataset.frame_sampling", {})
    repository_dir = get(config, "model.repository_dir")

    transform = build_transform(training=False, crop_size=resolution, repository_dir=repository_dir)
    dataset = Ego4DLTADataset(
        index_df=index_df,
        label_mapping=label_mapping,
        split=args.split,
        video_root=video_root,
        video_source=video_source,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        tau_a=tau_a,
        transform=transform,
        sampling_strategy=sampling_cfg.get("strategy", "uniform"),
        global_frames=int(sampling_cfg.get("global_frames", 24)),
        terminal_frames=int(sampling_cfg.get("terminal_frames", 8)),
        terminal_window_sec=float(sampling_cfg.get("terminal_window_sec", 2.0)),
    )

    backbone = load_vjepa2_backbone(
        frames_per_clip=frames_per_clip,
        frames_per_second=require(config, "dataset.frames_per_second"),
        resolution=resolution,
        checkpoint=expand_path(require(config, "model.checkpoint")),
        model_kwargs=require(config, "model.model_kwargs"),
        wrapper_kwargs=get(config, "model.wrapper_kwargs", {}),
        repository_dir=repository_dir,
        device=device,
    )

    cache_dir = expand_path(args.cache_dir or require(config, "dataset.feature_cache_dir")) / args.split
    stats = extract_and_cache_features(
        dataset,
        backbone,
        cache_dir,
        device,
        batch_size=get(config, "extraction.batch_size", 8),
        num_workers=get(config, "dataset.num_workers", 2),
    )
    step_log(1, "ExtractFeatures", f"{args.split} done: {stats}")


if __name__ == "__main__":
    main()
