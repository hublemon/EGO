"""Batch collation for Step 1 action-anticipation samples."""

from __future__ import annotations

from typing import Any

import torch


def anticipation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors, keep ids as LongTensors, and keep strings as plain lists.

    Dataset ``__getitem__`` implementations (``EK100Dataset``,
    ``Assembly101Dataset``) return one dict per sample with a fixed key set;
    this assembles a batch dict of the same keys.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if key == "video":
            out[key] = torch.stack(values)
        elif isinstance(values[0], str):
            out[key] = values
        elif isinstance(values[0], bool):
            out[key] = torch.tensor(values, dtype=torch.bool)
        elif isinstance(values[0], int):
            out[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], float):
            out[key] = torch.tensor(values, dtype=torch.float32)
        else:
            out[key] = torch.utils.data.default_collate(values)
    return out
