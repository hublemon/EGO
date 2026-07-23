#!/usr/bin/env python3
"""Materialize a compact, reusable store for GoalStep history-context training.

This is *not* another video decode or V-JEPA extraction pass.  It reads the
already frozen ``action_end-1s / 8s`` feature cache and, for each annotated
segment, stores:

* 17 temporal tokens obtained by spatially averaging each 16x16 token grid
  from the existing ``[4352, 1024] == [17, 256, 1024]`` feature tensor;
* frozen visual logits from the best direct next-action probe; and
* frozen current-action recognition logits used by the Phase-0 transition
  gate and for audit.

Output is sharded and resumable.  No cached labels are copied into the
derived store, so history supervision cannot accidentally leak through it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ego.common.config import get, load_config, require  # noqa: E402
from ego.common.paths import expand_path  # noqa: E402
from ego.datasets.ego4d import z1_sample_id  # noqa: E402
from ego.step1_action_anticipation.data.collator import anticipation_collate  # noqa: E402
from ego.step1_action_anticipation.data.feature_cache import FeatureCacheDataset  # noqa: E402
from ego.step1_action_anticipation.models import AnticipationHead  # noqa: E402


HEADS = ("verb", "noun", "action")
SPATIAL_TOKENS = 16 * 16
EXPECTED_INPUT_TOKENS = 17 * SPATIAL_TOKENS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_index(index_dir: Path, split: str) -> tuple[pd.DataFrame, Path]:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = index_dir / f"{split}{suffix}"
        if path.is_file():
            return reader(path).reset_index(drop=True), path
    raise FileNotFoundError(f"No {split}.parquet or {split}.csv under {index_dir}")


def source_sample_ids(frame: pd.DataFrame) -> list[str]:
    return [z1_sample_id(str(row["clip_uid"]), i) for i, row in frame.iterrows()]


def build_head(config_path: Path, checkpoint_path: Path, embed_dim: int, device: torch.device):
    config = load_config(config_path)
    classifier = get(config, "model.classifier", {})
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint.get("num_classes") or {"verb": 81, "noun": 140, "action": 293}
    model = AnticipationHead(
        num_verb_classes=int(classes["verb"]),
        num_noun_classes=int(classes["noun"]),
        num_action_classes=int(classes["action"]),
        embed_dim=embed_dim,
        num_heads=int(classifier.get("num_heads", 16)),
        depth=int(classifier.get("num_probe_blocks", 4)),
        repository_dir=get(config, "model.repository_dir"),
        use_temporal_metadata=bool(classifier.get("use_temporal_metadata", False)),
        temporal_duration_scale_sec=float(classifier.get("temporal_duration_scale_sec", 32.0)),
    )
    if getattr(model, "use_temporal_metadata", False):
        raise ValueError(f"History source head must use the legacy uniform schema: {config_path}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, classes


def validate_existing_shard(
    path: Path,
    expected_ids: list[str],
    *,
    expected_provenance_fingerprint: str,
    embed_dim: int,
    num_classes: dict[str, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        record = torch.load(path, map_location="cpu")
        if record.get("sample_ids") != expected_ids:
            return False
        if record.get("provenance_fingerprint") != expected_provenance_fingerprint:
            return False
        summaries = record.get("summaries")
        if not (
            torch.is_tensor(summaries)
            and summaries.dtype == torch.float16
            and tuple(summaries.shape) == (len(expected_ids), 17, embed_dim)
            and torch.isfinite(summaries).all()
        ):
            return False
        for dictionary_name in ("visual_logits", "recognition_logits"):
            values = record.get(dictionary_name, {})
            if set(values) != set(HEADS):
                return False
            for head in HEADS:
                logits = values[head]
                if not (
                    torch.is_tensor(logits)
                    and logits.dtype == torch.float32
                    and tuple(logits.shape) == (len(expected_ids), int(num_classes[head]))
                    and torch.isfinite(logits).all()
                ):
                    return False
        return True
    except Exception:
        return False


def atomic_torch_save(record: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(record, temporary)
    os.replace(temporary, path)


def process_split(
    split: str,
    source_index_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    visual_model: torch.nn.Module,
    recognition_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    shard_size: int,
    provenance_base: dict,
    num_classes: dict[str, int],
) -> dict:
    frame, index_path = read_index(source_index_dir, split)
    sample_ids = source_sample_ids(frame)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate source sample ids in {index_path}")
    missing = [sid for sid in sample_ids if not (cache_dir / split / f"{sid}.pt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{split} endpoint cache is incomplete: {len(missing)} missing; first={missing[0]}"
        )

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict] = []
    for shard_index, start in enumerate(range(0, len(sample_ids), shard_size)):
        stop = min(start + shard_size, len(sample_ids))
        shard_ids = sample_ids[start:stop]
        shard_path = split_dir / f"shard_{shard_index:05d}.pt"
        cache_stats = []
        for sample_id in shard_ids:
            cache_path = cache_dir / split / f"{sample_id}.pt"
            stat = cache_path.stat()
            cache_stats.append([sample_id, stat.st_size, stat.st_mtime_ns])
        provenance_fingerprint = fingerprint(
            {
                **provenance_base,
                "split": split,
                "source_index_sha256": sha256(index_path),
                "source_cache_file_stats": cache_stats,
            }
        )
        if validate_existing_shard(
            shard_path,
            shard_ids,
            expected_provenance_fingerprint=provenance_fingerprint,
            embed_dim=int(provenance_base["embed_dim"]),
            num_classes=num_classes,
        ):
            print(f"[{split}] reuse {shard_path.name} rows={len(shard_ids)}", flush=True)
            shard_records.append({
                "path": str(shard_path.relative_to(output_dir)),
                "start": start,
                "stop": stop,
                "rows": len(shard_ids),
                "provenance_fingerprint": provenance_fingerprint,
                "sha256": sha256(shard_path),
            })
            continue

        dataset = FeatureCacheDataset(shard_ids, cache_dir / split)
        if len(dataset) != len(shard_ids):
            raise RuntimeError(
                f"{split}/{shard_path.name}: requested={len(shard_ids)} cached={len(dataset)}"
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=anticipation_collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        stored_ids: list[str] = []
        summaries: list[torch.Tensor] = []
        visual_logits = {head: [] for head in HEADS}
        recognition_logits = {head: [] for head in HEADS}
        with torch.inference_mode():
            for batch in loader:
                features = batch["video"].to(device, non_blocking=True)
                if features.ndim != 3 or features.shape[1] != EXPECTED_INPUT_TOKENS:
                    raise ValueError(
                        "Expected cached V-JEPA tokens [B,4352,D] for 17x16x16 compression; "
                        f"got {tuple(features.shape)}"
                    )
                summary = features.reshape(
                    features.shape[0], 17, SPATIAL_TOKENS, features.shape[-1]
                ).mean(dim=2)
                visual = visual_model(features)
                recognition = recognition_model(features)

                stored_ids.extend(batch["sample_id"])
                summaries.append(summary.cpu().half())
                for head in HEADS:
                    visual_logits[head].append(visual[head].float().cpu())
                    recognition_logits[head].append(recognition[head].float().cpu())

        if stored_ids != shard_ids:
            raise RuntimeError(f"Sample-order drift while writing {shard_path}")
        record = {
            "schema_version": 1,
            "provenance_fingerprint": provenance_fingerprint,
            "compression": "reshape [4352,D] -> [17,256,D], spatial mean over 256",
            "sample_ids": stored_ids,
            "source_positions": [start, stop],
            "summaries": torch.cat(summaries),
            "visual_logits": {head: torch.cat(parts) for head, parts in visual_logits.items()},
            "recognition_logits": {
                head: torch.cat(parts) for head, parts in recognition_logits.items()
            },
        }
        atomic_torch_save(record, shard_path)
        shard_records.append({
            "path": str(shard_path.relative_to(output_dir)),
            "start": start,
            "stop": stop,
            "rows": len(shard_ids),
            "provenance_fingerprint": provenance_fingerprint,
            "sha256": sha256(shard_path),
        })
        print(f"[{split}] wrote {shard_path.name} rows={len(shard_ids)}", flush=True)

    return {
        "rows": len(sample_ids),
        "source_index": str(index_path),
        "source_index_sha256": sha256(index_path),
        "provenance_base_fingerprint": fingerprint(provenance_base),
        "shards": shard_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-index-dir",
        default="src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8",
    )
    parser.add_argument(
        "--cache-dir", default="../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna"
    )
    parser.add_argument(
        "--output-dir", default="../datasets/Ego4D/goalstep_history_context_store"
    )
    parser.add_argument(
        "--visual-config",
        default="configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10.yaml",
    )
    parser.add_argument(
        "--visual-checkpoint",
        default="outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10/best.pt",
    )
    parser.add_argument(
        "--recognition-config", default="configs/step1/goalstep/z1_end_m1_lobs8_vna.yaml"
    )
    parser.add_argument(
        "--recognition-checkpoint",
        default="outputs/goalstep/runs/z1_end_m1_lobs8_vna/best.pt",
    )
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size < 1 or args.shard_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size/shard-size must be positive and num-workers non-negative")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_index_dir = expand_path(args.source_index_dir)
    cache_dir = expand_path(args.cache_dir)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_cache = next((cache_dir / "val").glob("*.pt"))
    first_record = torch.load(first_cache, map_location="cpu")
    feature_shape = tuple(first_record["features"].shape)
    if len(feature_shape) != 2 or feature_shape[0] != EXPECTED_INPUT_TOKENS:
        raise ValueError(f"Unsupported endpoint cache feature shape: {feature_shape}")
    embed_dim = int(feature_shape[1])

    visual_config = expand_path(args.visual_config)
    visual_checkpoint = expand_path(args.visual_checkpoint)
    recognition_config = expand_path(args.recognition_config)
    recognition_checkpoint = expand_path(args.recognition_checkpoint)
    visual_model, visual_classes = build_head(
        visual_config, visual_checkpoint, embed_dim, device
    )
    recognition_model, recognition_classes = build_head(
        recognition_config, recognition_checkpoint, embed_dim, device
    )
    if visual_classes != recognition_classes:
        raise ValueError(
            f"Visual/recognition taxonomy mismatch: {visual_classes} != {recognition_classes}"
        )

    provenance_base = {
        "schema_version": 1,
        "source_cache_dir": str(cache_dir),
        "feature_shape": list(feature_shape),
        "embed_dim": embed_dim,
        "compression": "17 temporal slices x spatial mean(256 tokens)",
        "visual_config": str(visual_config),
        "visual_config_sha256": sha256(visual_config),
        "visual_checkpoint": str(visual_checkpoint),
        "visual_checkpoint_sha256": sha256(visual_checkpoint),
        "recognition_config": str(recognition_config),
        "recognition_config_sha256": sha256(recognition_config),
        "recognition_checkpoint": str(recognition_checkpoint),
        "recognition_checkpoint_sha256": sha256(recognition_checkpoint),
        "num_classes": visual_classes,
    }

    splits = ("train", "val") if args.split == "all" else (args.split,)
    manifest_path = output_dir / "manifest.json"
    existing_manifest = {}
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_base_fingerprint = fingerprint(provenance_base)
    reusable_splits = existing_manifest.get("splits", {})
    if existing_manifest.get("provenance_base_fingerprint") != provenance_base_fingerprint:
        # Never publish a manifest mixing splits produced from different
        # checkpoints/configs/cache schemas. Old shard files may remain on
        # disk, but they are unreferenced and fail per-shard fingerprint
        # validation if encountered by a later resumable run.
        reusable_splits = {}
    manifest = {
        **existing_manifest,
        "schema_version": 1,
        "kind": "goalstep_history_context_derived_store",
        "backbone_reextraction": False,
        "source_cache_dir": str(cache_dir),
        "feature_shape": list(feature_shape),
        "summary_shape": [17, embed_dim],
        "compression": "17 temporal slices x spatial mean(256 tokens)",
        "visual_config": str(visual_config),
        "visual_config_sha256": provenance_base["visual_config_sha256"],
        "visual_checkpoint": str(visual_checkpoint),
        "visual_checkpoint_sha256": provenance_base["visual_checkpoint_sha256"],
        "recognition_config": str(recognition_config),
        "recognition_config_sha256": provenance_base["recognition_config_sha256"],
        "recognition_checkpoint": str(recognition_checkpoint),
        "recognition_checkpoint_sha256": provenance_base["recognition_checkpoint_sha256"],
        "provenance_base_fingerprint": provenance_base_fingerprint,
        "num_classes": visual_classes,
        "splits": reusable_splits,
    }
    for split in splits:
        manifest["splits"][split] = process_split(
            split,
            source_index_dir,
            cache_dir,
            output_dir,
            visual_model,
            recognition_model,
            device,
            args.batch_size,
            args.num_workers,
            args.shard_size,
            provenance_base,
            {head: int(visual_classes[head]) for head in HEADS},
        )
        manifest_temporary = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
        manifest_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(manifest_temporary, manifest_path)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
