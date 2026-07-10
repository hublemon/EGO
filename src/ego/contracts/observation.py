"""Observation contract scaffold for egocentric video inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Observation:
    """A single observation window sampled from an egocentric video.

    Represents "what the model looked at" for one action-anticipation sample:
    the clip lies entirely within ``[observation_start_sec, observation_end_sec]``
    and the anticipated action begins at ``target_start_sec``, which must be
    ``>= observation_end_sec`` so no target-action frames leak into the input.
    """

    sample_id: str
    dataset: str
    split: str
    video_id: str
    observation_start_sec: float
    observation_end_sec: float
    target_start_sec: float
    anticipation_time_sec: float
    frames_per_clip: int
    frames_per_second: int

    def __post_init__(self) -> None:
        if self.observation_end_sec < self.observation_start_sec:
            raise ValueError(
                f"{self.sample_id}: observation_end_sec "
                f"({self.observation_end_sec}) < observation_start_sec "
                f"({self.observation_start_sec})"
            )
        if self.target_start_sec < self.observation_end_sec:
            raise ValueError(
                f"{self.sample_id}: target_start_sec ({self.target_start_sec}) "
                f"is before observation_end_sec ({self.observation_end_sec}); "
                "target-action frames would leak into the observation window."
            )

    def to_dict(self) -> dict:
        return asdict(self)
