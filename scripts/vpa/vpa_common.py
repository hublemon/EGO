"""Shared helpers for the GoalStep VPA (Visual Planning for Assistance) pipeline.

VPA (Patel et al., ICCV 2023): given the observation history + an explicit goal,
predict the next T high-level steps. Here we port it to Ego4D GoalStep (cooking).
This module holds only pieces every script needs: label normalisation, JSON IO,
and the GoalStep annotation reader. NO video is required — the port is
text-conditioned (history = goal text + observed step labels).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def normalize_label(text: str) -> str:
    """Canonicalise a step label so equal steps compare equal regardless of
    surface noise: unicode NFKC, lowercase, collapse whitespace, strip edge
    punctuation. Applied identically to GT labels, the candidate vocabulary,
    and model predictions so scoring is fair."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;:\t\n\r")
    return s


def label_from_segment(seg: dict, field: str) -> str:
    """Pick the raw label field ('step_category' or 'step_description') then
    normalise. step_category is the controlled 514-step taxonomy label."""
    return normalize_label(seg.get(field, ""))


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def iter_goalstep_videos(goalstep_json: dict):
    """Yield each video dict from a goalstep_{train,val}.json. In GoalStep each
    video carries exactly one goal (video-level goal_category/goal_description)
    and `segments` are its time-ordered steps (each step may hold substeps)."""
    for v in goalstep_json.get("videos", []):
        yield v


def video_steps(video: dict, level: str, essential_only: bool):
    """Return the chosen-level segments of a video, time-sorted, optionally
    restricted to is_relevant=='essential'.

    level='step'    -> video['segments']
    level='substep' -> flattened video['segments'][*]['segments']
    """
    if level == "step":
        segs = list(video.get("segments", []) or [])
    elif level == "substep":
        segs = [sub for st in video.get("segments", []) or [] for sub in (st.get("segments", []) or [])]
    else:
        raise ValueError(f"level must be 'step' or 'substep', got {level!r}")
    if essential_only:
        segs = [s for s in segs if s.get("is_relevant") == "essential"]
    segs = [s for s in segs if s.get("start_time") is not None]
    segs.sort(key=lambda s: (float(s["start_time"]), float(s.get("end_time", s["start_time"]))))
    return segs


def goal_text_of(video: dict) -> str:
    """Human-readable goal string given to the planner (goal_description,
    falling back to a de-prefixed goal_category)."""
    desc = (video.get("goal_description") or "").strip()
    if desc:
        return desc
    cat = (video.get("goal_category") or "").strip()
    return cat.split(":", 1)[-1].replace("_", " ").strip().lower()
