"""Base dataset adapter scaffold.

Common interface every egocentric action-anticipation dataset adapter
(``ek100.py``, ``assembly101.py``, and eventually ``ego4d.py``) must follow,
so Step 1 training/inference code is dataset-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ego.contracts.observation import Observation
from ego.datasets.label_mapping import LabelMapping


class EgoActionAnticipationDataset(Dataset, ABC):
    """Map-style dataset: one item == one (observation clip, future action) sample.

    ``__getitem__`` must return a dict with at least:
        video: FloatTensor[C, T, H, W] -- the transformed observation clip
        verb_id / noun_id / action_id: int -- unified (dense) label ids
        anticipation_time_sec: float
        sample_id: str
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @abstractmethod
    def get_label_mapping(self) -> LabelMapping:
        """Return the :class:`LabelMapping` this dataset's labels are encoded with."""

    @abstractmethod
    def get_sample_metadata(self, index: int) -> Observation:
        """Return the timing/identity metadata for ``index`` without decoding video."""
