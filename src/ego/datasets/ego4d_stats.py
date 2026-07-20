"""Class-distribution and imbalance statistics for the Ego4D LTA Z=1 index.

Pure computation only (no plotting, no file I/O) -- ``scripts/step1/ego4d_lta/
analyze_lta_stats.py`` calls these and handles saving json/png, matching the
metrics.py convention used elsewhere in Step 1.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def class_frequency(df: pd.DataFrame, column: str) -> Counter:
    return Counter(df[column].tolist())


def head_mid_tail_bands(freq: dict[Any, int], head_frac: float = 0.2, tail_frac: float = 0.5) -> dict[Any, str]:
    """Assign each class to a 'head' / 'mid' / 'tail' band by frequency rank.

    ``head`` = the most frequent ``head_frac`` fraction of classes, ``tail`` =
    the least frequent ``tail_frac`` fraction, ``mid`` = everything between.
    """
    if not 0 < head_frac < 1 or not 0 < tail_frac < 1 or head_frac + tail_frac > 1:
        raise ValueError("head_frac and tail_frac must be in (0, 1) and sum to <= 1")

    ordered = sorted(freq.items(), key=lambda kv: -kv[1])
    n = len(ordered)
    n_head = max(1, round(n * head_frac))
    n_tail = max(1, round(n * tail_frac))

    bands: dict[Any, str] = {}
    for i, (cls, _) in enumerate(ordered):
        if i < n_head:
            bands[cls] = "head"
        elif i >= n - n_tail:
            bands[cls] = "tail"
        else:
            bands[cls] = "mid"
    return bands


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient of a frequency distribution: 0 = perfectly even, -> 1 = maximally imbalanced."""
    if not values:
        return float("nan")
    v = sorted(values)
    n = len(v)
    total = sum(v)
    if total == 0:
        return 0.0
    cum_sum = sum(i * x for i, x in enumerate(v, start=1))
    return (2 * cum_sum) / (n * total) - (n + 1) / n


def imbalance_ratio(values: list[float]) -> float:
    """max(count) / min(count) across classes with at least one observed sample."""
    nonzero = [v for v in values if v > 0]
    if not nonzero:
        return float("nan")
    return max(nonzero) / min(nonzero)


def verb_noun_cooccurrence(
    df: pd.DataFrame, verb_col: str = "verb_label", noun_col: str = "noun_label"
) -> pd.DataFrame:
    """Verb x noun co-occurrence count matrix (rows=verb, cols=noun)."""
    return pd.crosstab(df[verb_col], df[noun_col])


def scenario_distribution(df: pd.DataFrame, scenario_col: str = "scenario") -> Counter:
    return Counter(df[scenario_col].tolist())


def build_pilot_taxonomy(
    df: pd.DataFrame,
    top_verb: int,
    top_noun: int,
    mode: str = "exclude",
    verb_col: str = "verb_label",
    noun_col: str = "noun_label",
) -> tuple[pd.DataFrame, dict]:
    """Restrict the index to the ``top_verb``/``top_noun`` most frequent classes.

    ``mode="exclude"`` (recommended): drop rows whose verb or noun falls
    outside the kept sets. ``mode="other"``: remap excluded verb/noun ids to
    a synthetic "other" bucket (``-1``) instead of dropping the row, so
    sample count is preserved at the cost of a catch-all class.

    Returns ``(filtered_df, info)`` where ``info`` records the kept raw ids
    and row counts before/after -- callers (the CLI) are responsible for
    surfacing the "pilot taxonomy results are not comparable to full
    taxonomy results" warning, per the Step 1 -> Ego4D LTA spec.
    """
    if mode not in ("exclude", "other"):
        raise ValueError(f"mode must be 'exclude' or 'other', got {mode!r}")

    verb_freq = class_frequency(df, verb_col)
    noun_freq = class_frequency(df, noun_col)
    kept_verbs = {v for v, _ in sorted(verb_freq.items(), key=lambda kv: -kv[1])[:top_verb]}
    kept_nouns = {n for n, _ in sorted(noun_freq.items(), key=lambda kv: -kv[1])[:top_noun]}

    info = {
        "mode": mode,
        "top_verb": top_verb,
        "top_noun": top_noun,
        "kept_verb_ids": sorted(kept_verbs),
        "kept_noun_ids": sorted(kept_nouns),
        "rows_before": len(df),
    }

    if mode == "exclude":
        filtered = df[df[verb_col].isin(kept_verbs) & df[noun_col].isin(kept_nouns)].reset_index(drop=True)
    else:
        filtered = df.copy()
        filtered[verb_col] = filtered[verb_col].where(filtered[verb_col].isin(kept_verbs), -1)
        filtered[noun_col] = filtered[noun_col].where(filtered[noun_col].isin(kept_nouns), -1)

    info["rows_after"] = len(filtered)
    return filtered, info
