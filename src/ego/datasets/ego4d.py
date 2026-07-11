"""Ego4D dataset adapter: LTA (Long-Term Anticipation) Z=1 next-action form.

Reduces the Ego4D LTA benchmark (predict the next Z actions) to the same
Z=1 "anticipate the single next action from an observation window" shape
that ``ego.datasets.ek100``/``assembly101`` already use, so the existing
Step 1 architecture (frozen V-JEPA2 + attentive probe + verb/noun/action
heads) can be reused unchanged -- only the data loader and output taxonomy
differ. See ``scripts/step1/ego4d_lta/`` for the CLI tools built on this
module and ``scripts/step1/ego4d_lta/PILOT.md`` for the validation procedure.

Schema note: the exact field names below follow the publicly documented
Ego4D FHO LTA annotation format (``fho_lta_{train,val}.json``: a flat list of
per-action records sharing a ``clip_uid``, each with clip-relative and
clip-parent-relative timestamps; ``fho_lta_taxonomy.json``: ``{"verbs": [...],
"nouns": [...]}`` dense-index lists; scenario tags come from the separate
``ego4d.json`` video metadata catalog, joined here by ``video_uid``). This
has **not been run against the real files yet** (see
``develop_report/`` for why) -- ``_first_present`` resolves several
candidate key names per field and raises a clear, catalogued error instead
of a bare ``KeyError`` if the real schema differs, so a first real run should
fail loudly and locally-fixably rather than silently mis-parse.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ego.common.exceptions import EgoDatasetError
from ego.contracts.observation import Observation
from ego.datasets.base import EgoActionAnticipationDataset
from ego.datasets.label_mapping import LabelMapping, build_label_mapping
from ego.datasets.video_sampling import sample_uniform_frame_indices

DATASET_NAME = "Ego4D-LTA-Z1"


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #


@dataclass
class LTATaxonomy:
    """Dense verb/noun vocabularies as defined by ``fho_lta_taxonomy.json``."""

    verbs: list[str]
    nouns: list[str]

    @property
    def num_verbs(self) -> int:
        return len(self.verbs)

    @property
    def num_nouns(self) -> int:
        return len(self.nouns)

    def verb_text(self, verb_label: int) -> str:
        return self.verbs[verb_label]

    def noun_text(self, noun_label: int) -> str:
        return self.nouns[noun_label]


def load_lta_taxonomy(path: str | Path) -> LTATaxonomy:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "verbs" not in data or "nouns" not in data:
        raise EgoDatasetError(
            f"{path}: expected top-level 'verbs' and 'nouns' lists, found keys {list(data.keys())}"
        )
    taxonomy = LTATaxonomy(verbs=list(data["verbs"]), nouns=list(data["nouns"]))
    return taxonomy


# --------------------------------------------------------------------------- #
# Annotation parsing (defensive: tries several known field-name variants)
# --------------------------------------------------------------------------- #

_FIELD_CANDIDATES: dict[str, list[str]] = {
    "clip_uid": ["clip_uid"],
    "video_uid": ["video_uid"],
    "clip_parent_start_sec": ["clip_parent_start_sec", "parent_start_sec"],
    "action_clip_start_sec": ["action_clip_start_sec", "clip_start_sec", "start_sec"],
    "action_clip_end_sec": ["action_clip_end_sec", "clip_end_sec", "end_sec"],
    "action_idx": ["action_idx", "action_index"],
    "verb_label": ["verb_label", "verb_id"],
    "noun_label": ["noun_label", "noun_id"],
    "verb": ["verb"],
    "noun": ["noun"],
}


def _first_present(record: dict, field_name: str) -> Any:
    for candidate in _FIELD_CANDIDATES[field_name]:
        if candidate in record:
            return record[candidate]
    raise EgoDatasetError(
        f"None of {_FIELD_CANDIDATES[field_name]} (looking for '{field_name}') found in annotation "
        f"record. Available keys: {sorted(record.keys())}. The Ego4D LTA JSON schema may have "
        f"changed -- add the real key name to _FIELD_CANDIDATES['{field_name}'] in "
        f"src/ego/datasets/ego4d.py."
    )


def parse_lta_annotations(path: str | Path) -> pd.DataFrame:
    """Flatten ``fho_lta_{train,val}.json`` into one row per action.

    Output columns: ``clip_uid, video_uid, action_idx, clip_parent_start_sec,
    action_clip_start_sec, action_clip_end_sec, verb_label, noun_label,
    verb_text, noun_text``. ``action_clip_*_sec`` are clip-relative (relative
    to the clip file's own start, i.e. ``clip_parent_start_sec`` in the
    parent video) -- see the module docstring on ``video_source``.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("clips", data if isinstance(data, list) else None)
    if records is None:
        raise EgoDatasetError(
            f"{path}: expected a top-level 'clips' list or a bare list of action records, "
            f"found keys {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    rows = []
    for r in records:
        rows.append(
            {
                "clip_uid": _first_present(r, "clip_uid"),
                "video_uid": _first_present(r, "video_uid"),
                "action_idx": int(_first_present(r, "action_idx")),
                "clip_parent_start_sec": float(r.get("clip_parent_start_sec", 0.0) or 0.0),
                "action_clip_start_sec": float(_first_present(r, "action_clip_start_sec")),
                "action_clip_end_sec": float(_first_present(r, "action_clip_end_sec")),
                "verb_label": int(_first_present(r, "verb_label")),
                "noun_label": int(_first_present(r, "noun_label")),
                "verb_text": r.get("verb"),
                "noun_text": r.get("noun"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise EgoDatasetError(f"{path}: parsed zero action records.")
    return df.sort_values(["clip_uid", "action_idx"]).reset_index(drop=True)


def load_video_scenarios(ego4d_json_path: str | Path) -> dict[str, str]:
    """Map ``video_uid -> scenario`` from the Ego4D metadata catalog (``ego4d.json``).

    A video can carry multiple scenario tags; this keeps the first one for a
    single-valued join column (sufficient for the stratified-sampling and
    per-scenario breakdown use cases here -- see ``docs``/``PILOT.md`` for
    the caveat).
    """
    with open(ego4d_json_path, encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", data if isinstance(data, list) else None)
    if videos is None:
        raise EgoDatasetError(f"{ego4d_json_path}: expected a top-level 'videos' list.")

    mapping = {}
    for v in videos:
        video_uid = v.get("video_uid")
        scenarios = v.get("scenarios") or v.get("scenario")
        if video_uid is None:
            continue
        if isinstance(scenarios, list):
            mapping[video_uid] = scenarios[0] if scenarios else "unknown"
        elif scenarios:
            mapping[video_uid] = scenarios
        else:
            mapping[video_uid] = "unknown"
    return mapping


# --------------------------------------------------------------------------- #
# Z=1 index construction
# --------------------------------------------------------------------------- #


@dataclass
class Z1IndexStats:
    total_actions: int = 0
    kept: int = 0
    truncated: int = 0
    excluded_min_obs: int = 0
    excluded_first_action: int = 0
    boundary_policy: str = "truncate"

    def to_dict(self) -> dict:
        return {
            "total_actions": self.total_actions,
            "kept": self.kept,
            "truncated": self.truncated,
            "excluded_min_obs": self.excluded_min_obs,
            "excluded_first_action": self.excluded_first_action,
            "boundary_policy": self.boundary_policy,
        }


def build_z1_index(
    actions_df: pd.DataFrame,
    tau_a: float = 1.0,
    l_obs: float = 3.5,
    min_obs_sec: float = 0.5,
    boundary_policy: str = "truncate",
    scenario_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, Z1IndexStats]:
    """Convert per-action rows into Z=1 anticipation samples.

    For each action (except the first in a clip, which has no preceding
    observation), the target is that action's (verb, noun); the observation
    window is::

        obs_end_sec   = action_start_sec - tau_a
        obs_start_sec = obs_end_sec - l_obs

    clamped to the clip start (``0.0``, clip-relative). ``boundary_policy``:
    - ``"truncate"``: clamp ``obs_start_sec`` to ``0.0`` and keep the sample
      (marking ``boundary_flag=True``) as long as the resulting window is
      still >= ``min_obs_sec`` long.
    - ``"exclude"``: drop any sample whose window would need truncation at all.

    Returns the index DataFrame (columns: ``video_uid, clip_uid,
    obs_start_sec, obs_end_sec, verb_label, noun_label, scenario,
    boundary_flag``) and a :class:`Z1IndexStats` recording how many samples
    were kept/truncated/excluded and why (printed by the CLI, per spec).
    """
    if boundary_policy not in ("truncate", "exclude"):
        raise ValueError(f"boundary_policy must be 'truncate' or 'exclude', got {boundary_policy!r}")

    stats = Z1IndexStats(boundary_policy=boundary_policy)
    rows = []

    for clip_uid, clip_df in actions_df.groupby("clip_uid", sort=False):
        clip_df = clip_df.sort_values("action_idx")
        video_uid = clip_df["video_uid"].iloc[0]
        scenario = (scenario_map or {}).get(video_uid, "unknown")

        for _, action in clip_df.iterrows():
            stats.total_actions += 1
            if action["action_idx"] == clip_df["action_idx"].min():
                # No preceding observation is available for the first action in a clip.
                stats.excluded_first_action += 1
                continue

            action_start_sec = action["action_clip_start_sec"]
            obs_end_sec = action_start_sec - tau_a
            obs_start_sec = obs_end_sec - l_obs

            boundary_flag = obs_start_sec < 0.0
            if boundary_flag:
                if boundary_policy == "exclude":
                    stats.excluded_min_obs += 1
                    continue
                obs_start_sec = 0.0

            if obs_end_sec - obs_start_sec < min_obs_sec:
                stats.excluded_min_obs += 1
                continue

            rows.append(
                {
                    "video_uid": video_uid,
                    "clip_uid": clip_uid,
                    "obs_start_sec": obs_start_sec,
                    "obs_end_sec": obs_end_sec,
                    "verb_label": int(action["verb_label"]),
                    "noun_label": int(action["noun_label"]),
                    "scenario": scenario,
                    "boundary_flag": bool(boundary_flag),
                }
            )
            stats.kept += 1
            if boundary_flag:
                stats.truncated += 1

    index_df = pd.DataFrame(
        rows,
        columns=[
            "video_uid", "clip_uid", "obs_start_sec", "obs_end_sec",
            "verb_label", "noun_label", "scenario", "boundary_flag",
        ],
    )
    return index_df, stats


def z1_sample_id(clip_uid: str, row_position: int) -> str:
    """Single source of truth for sample_id, so it's reconstructible from an index
    file alone (e.g. for joining ``scenario`` back onto cached-feature training
    batches) without re-instantiating :class:`Ego4DLTADataset`."""
    return f"{clip_uid}_{row_position}"


def index_scenario_lookup(index_df: pd.DataFrame) -> dict[str, str]:
    """Map ``sample_id -> scenario`` for every row in ``index_df``, in its current row order.

    Requires ``index_df`` to be in the same row order it was in when
    ``Ego4DLTADataset``/``extract_features.py`` generated sample ids from it
    (true as long as the same parquet/csv file is read both times, since
    pandas preserves on-disk row order).
    """
    df = index_df.reset_index(drop=True)
    return {z1_sample_id(row["clip_uid"], i): row["scenario"] for i, row in df.iterrows()}


def split_dev_heldout(
    val_df: pd.DataFrame, dev_fraction: float = 0.8, seed: int = 42, group_col: str = "clip_uid"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split val into internal dev/heldout by clip (not by action) to avoid leakage.

    Deterministic for a given ``seed``: the same clip always lands in the
    same split across runs.
    """
    clip_uids = sorted(val_df[group_col].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(clip_uids)
    n_dev = round(len(clip_uids) * dev_fraction)
    dev_clips = set(clip_uids[:n_dev])
    dev_df = val_df[val_df[group_col].isin(dev_clips)].reset_index(drop=True)
    heldout_df = val_df[~val_df[group_col].isin(dev_clips)].reset_index(drop=True)
    return dev_df, heldout_df


def register_action_labels(index_df: pd.DataFrame) -> LabelMapping:
    """Fit the dense verb/noun/action label mapping on the (train) index's observed pairs.

    Reuses :func:`ego.datasets.label_mapping.build_label_mapping` unchanged --
    "only (verb, noun) combinations seen in train are registered" is exactly
    what it already does.
    """
    pairs = list(zip(index_df["verb_label"].astype(int), index_df["noun_label"].astype(int)))
    return build_label_mapping(pairs)


# --------------------------------------------------------------------------- #
# Video path resolution
# --------------------------------------------------------------------------- #


def resolve_clip_video_path(
    video_root: str | Path, video_source: str, video_uid: str, clip_uid: str
) -> Path:
    """Resolve the video file backing a sample.

    ``video_source="clips"`` (default, matches the boundary-truncation policy
    above): ``video_root/clip_uid.mp4``, a pre-trimmed Ego4D clip.
    ``video_source="full_scale"``: ``video_root/video_uid.mp4``, the full
    parent video -- in this mode timestamps must be video-relative, i.e. the
    caller should add ``clip_parent_start_sec`` before building the index.
    """
    base = Path(video_root)
    if video_source == "clips":
        return base / f"{clip_uid}.mp4"
    if video_source == "full_scale":
        return base / f"{video_uid}.mp4"
    raise ValueError(f"video_source must be 'clips' or 'full_scale', got {video_source!r}")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class Ego4DLTADataset(EgoActionAnticipationDataset):
    """One item == one Z=1 Ego4D LTA sample (fixed observation window, see :func:`build_z1_index`)."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        label_mapping: LabelMapping,
        split: str,
        video_root: str | Path,
        video_source: str,
        frames_per_clip: int,
        resolution: int,
        tau_a: float,
        transform: Any | None = None,
    ) -> None:
        if len(index_df) == 0:
            raise EgoDatasetError(f"Ego4DLTADataset[{split}] built from an empty index.")
        self._rows = index_df.reset_index(drop=True)
        self._label_mapping = label_mapping
        self.split = split
        self.video_root = video_root
        self.video_source = video_source
        self.frames_per_clip = frames_per_clip
        self.resolution = resolution
        self.tau_a = tau_a
        self.transform = transform
        self._fps_cache: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def get_label_mapping(self) -> LabelMapping:
        return self._label_mapping

    def _sample_id(self, row: pd.Series, index: int) -> str:
        return z1_sample_id(row["clip_uid"], index)

    def _video_path(self, row: pd.Series) -> Path:
        return resolve_clip_video_path(self.video_root, self.video_source, row["video_uid"], row["clip_uid"])

    def _video_fps(self, video_path: Path) -> float:
        key = str(video_path)
        if key not in self._fps_cache:
            from decord import VideoReader, cpu

            vr = VideoReader(key, num_threads=1, ctx=cpu(0))
            self._fps_cache[key] = float(vr.get_avg_fps())
        return self._fps_cache[key]

    def get_sample_metadata(self, index: int) -> Observation:
        row = self._rows.iloc[index]
        return Observation(
            sample_id=self._sample_id(row, index),
            dataset=DATASET_NAME,
            split=self.split,
            video_id=str(row["clip_uid"]),
            observation_start_sec=float(row["obs_start_sec"]),
            observation_end_sec=float(row["obs_end_sec"]),
            target_start_sec=float(row["obs_end_sec"]) + self.tau_a,
            anticipation_time_sec=self.tau_a,
            frames_per_clip=self.frames_per_clip,
            frames_per_second=round(self.frames_per_clip / max(row["obs_end_sec"] - row["obs_start_sec"], 1e-6)),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        from decord import VideoReader, cpu

        row = self._rows.iloc[index]
        video_path = self._video_path(row)
        vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        vfps = self._fps_cache.setdefault(str(video_path), float(vr.get_avg_fps()))

        frame_indices = sample_uniform_frame_indices(
            start_sec=row["obs_start_sec"],
            end_sec=row["obs_end_sec"],
            video_fps=vfps,
            num_frames=self.frames_per_clip,
        )
        frame_indices = frame_indices.clip(0, len(vr) - 1)
        buffer = vr.get_batch(frame_indices.tolist()).asnumpy()
        video = self.transform(buffer) if self.transform is not None else buffer

        verb_raw = int(row["verb_label"])
        noun_raw = int(row["noun_label"])

        return {
            "video": video,
            "verb_id": self._label_mapping.encode_verb(verb_raw),
            "noun_id": self._label_mapping.encode_noun(noun_raw),
            "action_id": self._label_mapping.encode_action(verb_raw, noun_raw),
            "verb_id_raw": verb_raw,
            "noun_id_raw": noun_raw,
            "anticipation_time_sec": self.tau_a,
            "observation_start_sec": float(row["obs_start_sec"]),
            "observation_end_sec": float(row["obs_end_sec"]),
            "target_start_sec": float(row["obs_end_sec"]) + self.tau_a,
            "sample_id": self._sample_id(row, index),
            "video_id": str(row["clip_uid"]),
            "scenario": str(row["scenario"]),
        }
