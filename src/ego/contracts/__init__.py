"""Inter-stage data contract definitions for EGO."""

from ego.contracts.action import ActionLabel
from ego.contracts.candidates import ActionCandidate, StepOneCandidateRecord
from ego.contracts.observation import Observation

__all__ = [
    "ActionLabel",
    "ActionCandidate",
    "StepOneCandidateRecord",
    "Observation",
]
