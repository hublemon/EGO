"""Feature cache: precompute and reuse frozen-backbone tokens across epochs.

For a dataset the size of Assembly101 (hundreds of GB of video), decoding and
running the backbone forward pass every epoch is prohibitively slow. This
mirrors the validated prototype's two-stage design
(``scripts/extract_features_a101.py`` + ``scripts/train_probe_a101.py``):
extract once, train the (much cheaper) classifier heads from cache afterward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ego.common.io import ensure_dir
from ego.common.logging import step_log
from ego.step1_action_anticipation.data.collator import anticipation_collate


def extract_and_cache_features(
    dataset: Dataset,
    backbone: torch.nn.Module,
    cache_dir: str | Path,
    device: str | torch.device,
    batch_size: int = 8,
    num_workers: int = 2,
) -> dict[str, int]:
    """Run the frozen backbone over every sample in ``dataset`` and cache its output tokens.

    Skips samples whose cache file already exists, so extraction is resumable
    across process restarts. Returns ``{"saved", "skipped", "total"}``.
    """
    cache_dir = ensure_dir(cache_dir)

    # Resume before decode: Ego4D's dataset can expose a sample_id through
    # get_sample_metadata() without opening the video. Filtering here avoids
    # decoding every already-cached clip after an interrupted extraction.
    pending_indices: list[int] | None = None
    get_metadata = getattr(dataset, "get_sample_metadata", None)
    if callable(get_metadata):
        pending_indices = [
            index for index in range(len(dataset))
            if not (cache_dir / f"{get_metadata(index).sample_id}.pt").is_file()
        ]
    loader_dataset = Subset(dataset, pending_indices) if pending_indices is not None else dataset
    loader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=anticipation_collate,
    )

    saved = 0
    skipped = len(dataset) - len(loader_dataset)
    for batch_idx, batch in enumerate(loader):
        sample_ids = batch["sample_id"]
        pending = [i for i, sid in enumerate(sample_ids) if not (cache_dir / f"{sid}.pt").exists()]
        if not pending:
            skipped += len(sample_ids)
            continue

        clips = batch["video"].to(device)
        ant_times = batch["anticipation_time_sec"].to(device)
        with torch.no_grad():
            features = backbone(clips, ant_times)

        for i in pending:
            record = {
                "features": features[i].detach().cpu().half(),
                "verb_id": int(batch["verb_id"][i]),
                "noun_id": int(batch["noun_id"][i]),
                "action_id": int(batch["action_id"][i]),
                "anticipation_time_sec": float(batch["anticipation_time_sec"][i]),
                "sample_id": sample_ids[i],
            }
            for metadata_key in (
                "observation_duration_sec",
                "observed_action_duration_sec",
                "frame_time_positions",
                "frame_terminal_mask",
                "annotation_level_id",
            ):
                if metadata_key in batch:
                    value = batch[metadata_key][i]
                    record[metadata_key] = value.detach().cpu() if torch.is_tensor(value) else value
            torch.save(record, cache_dir / f"{sample_ids[i]}.pt")
            saved += 1
        skipped += len(sample_ids) - len(pending)

        if (batch_idx + 1) % 50 == 0:
            step_log(1, "FeatureCache", f"batch {batch_idx + 1}: saved={saved} skipped={skipped}")

    step_log(1, "FeatureCache", f"Done: saved={saved} skipped={skipped} total={saved + skipped}")
    return {"saved": saved, "skipped": skipped, "total": saved + skipped}


class FeatureCacheDataset(Dataset):
    """Reads cached backbone features by sample_id instead of decoding video.

    Samples whose cache file is missing are silently dropped (they simply
    haven't been extracted yet); callers should check ``len()`` against the
    expected sample count and re-run :func:`extract_and_cache_features` first.
    """

    def __init__(
        self,
        sample_ids: list[str],
        cache_dir: str | Path,
        label_overrides: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.sample_ids = [sid for sid in sample_ids if (self.cache_dir / f"{sid}.pt").is_file()]
        self.label_overrides = label_overrides or {}

    @classmethod
    def from_cache_dir(cls, cache_dir: str | Path) -> "FeatureCacheDataset":
        """Discover every cached sample under ``cache_dir`` (one ``.pt`` file per sample)."""
        cache_dir = Path(cache_dir)
        sample_ids = [p.stem for p in sorted(cache_dir.glob("*.pt"))]
        return cls(sample_ids, cache_dir)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = torch.load(self.cache_dir / f"{self.sample_ids[index]}.pt", map_location="cpu")
        result = {
            "video": record["features"].float(),
            "verb_id": record["verb_id"],
            "noun_id": record["noun_id"],
            "action_id": record["action_id"],
            "anticipation_time_sec": record["anticipation_time_sec"],
            "sample_id": record["sample_id"],
        }
        for metadata_key in (
            "observation_duration_sec",
            "observed_action_duration_sec",
            "frame_time_positions",
            "frame_terminal_mask",
            "annotation_level_id",
        ):
            if metadata_key in record:
                result[metadata_key] = record[metadata_key]
        # Some protocols reuse an identical frozen visual observation while
        # changing only its supervised target (for example, A2.end-1s -> A3).
        # Keep the large feature cache immutable and apply the audited label
        # overlay at read time instead of duplicating hundreds of GB of tokens.
        override = self.label_overrides.get(record["sample_id"])
        if override is not None:
            result.update(override)
        return result
