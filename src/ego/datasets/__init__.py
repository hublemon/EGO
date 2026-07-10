"""Dataset adapters for EGO."""

from ego.datasets.assembly101 import Assembly101Dataset, build_assembly101_manifest
from ego.datasets.base import EgoActionAnticipationDataset
from ego.datasets.ek100 import EK100Dataset, build_ek100_manifest
from ego.datasets.label_mapping import LabelMapping, build_label_mapping

__all__ = [
    "EgoActionAnticipationDataset",
    "EK100Dataset",
    "build_ek100_manifest",
    "Assembly101Dataset",
    "build_assembly101_manifest",
    "LabelMapping",
    "build_label_mapping",
]
