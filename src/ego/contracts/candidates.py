"""Step 1 candidate distribution contract scaffold.

Records produced here are the Step 1 -> Step 2 hand-off artifact and must stay
compatible with ``schemas/step1_candidates.schema.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ActionCandidate:
    """One ranked (verb, noun) candidate with raw logit and probability.

    Matches the ``candidates[]`` item schema in
    ``schemas/step1_candidates.schema.json`` (``rank``/``verb``/``noun``/
    ``probability`` required; the rest optional but always populated by EGO).
    """

    rank: int
    verb: str | None
    noun: str | None
    probability: float
    verb_id: int | None = None
    noun_id: int | None = None
    action_id: int | None = None
    logit: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepOneCandidateRecord:
    """One Step 1 inference sample: observation window + ranked action candidates.

    ``to_dict()`` emits ``candidates`` (the ranked verb+noun action-pair list,
    the field required by the schema) alongside ``verb_candidates`` /
    ``noun_candidates`` (independent per-head top-K) and ``gt`` so Step 2 reward
    shaping and offline evaluation both have what they need. The schema allows
    additional properties, so this superset stays schema-valid.
    """

    sample_id: str
    dataset: str
    split: str
    video_id: str
    observation_start_sec: float
    observation_end_sec: float
    target_start_sec: float
    anticipation_time_sec: float
    entropy: float
    action_candidates: list[ActionCandidate]
    verb_candidates: list[ActionCandidate] = field(default_factory=list)
    noun_candidates: list[ActionCandidate] = field(default_factory=list)
    gt: dict | None = None
    checkpoint: str | None = None
    config_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "video_id": self.video_id,
            "observation_start_sec": self.observation_start_sec,
            "observation_end_sec": self.observation_end_sec,
            "target_start_sec": self.target_start_sec,
            "anticipation_time_sec": self.anticipation_time_sec,
            "entropy": self.entropy,
            "candidates": [c.to_dict() for c in self.action_candidates],
            "verb_candidates": [c.to_dict() for c in self.verb_candidates],
            "noun_candidates": [c.to_dict() for c in self.noun_candidates],
            "gt": self.gt,
            "checkpoint": self.checkpoint,
            "config_path": self.config_path,
        }
