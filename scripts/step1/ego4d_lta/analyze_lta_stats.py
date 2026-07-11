"""Class-distribution / imbalance report for a built Ego4D LTA Z=1 index.

Reads the ``train`` index written by ``build_lta_z1_index.py`` and reports,
per verb/noun/action: class frequency, head/mid/tail bands, Gini
coefficient, and max/min imbalance ratio -- plus verb-noun co-occurrence and
scenario distribution. Optionally builds a "pilot taxonomy" (top-N verb/noun
classes only) for fast iteration -- see the WARNING this prints and
``PILOT.md``: pilot-taxonomy metrics are not comparable to full-taxonomy
results, because the class count and difficulty differ.

Usage:
    python scripts/step1/ego4d_lta/analyze_lta_stats.py \
        --index outputs/ego4d_lta/index/train.parquet \
        --output-dir outputs/ego4d_lta/stats

    python scripts/step1/ego4d_lta/analyze_lta_stats.py \
        --index outputs/ego4d_lta/index/train.parquet \
        --output-dir outputs/ego4d_lta/stats_pilot \
        --top-verb 80 --top-noun 150 --pilot-mode exclude
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pandas as pd  # noqa: E402

from ego.common.io import ensure_dir, write_json  # noqa: E402
from ego.common.logging import step_log  # noqa: E402
from ego.datasets.ego4d_stats import (  # noqa: E402
    build_pilot_taxonomy,
    class_frequency,
    gini_coefficient,
    head_mid_tail_bands,
    imbalance_ratio,
    scenario_distribution,
    verb_noun_cooccurrence,
)


def _read_index(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _class_report(df: pd.DataFrame, column: str) -> dict:
    freq = class_frequency(df, column)
    bands = head_mid_tail_bands(dict(freq))
    band_counts = {"head": 0, "mid": 0, "tail": 0}
    for band in bands.values():
        band_counts[band] += 1
    values = list(freq.values())
    return {
        "num_classes": len(freq),
        "num_samples": int(sum(values)),
        "band_class_counts": band_counts,
        "gini": gini_coefficient(values),
        "imbalance_ratio_max_over_min": imbalance_ratio(values),
        "top_10": freq.most_common(10),
        "bottom_10": freq.most_common()[:-11:-1],
    }


def _save_bar_png(freq: dict, title: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        step_log(1, "AnalyzeLTAStats", f"matplotlib not installed; skipping {path}")
        return False

    ordered = sorted(freq.items(), key=lambda kv: -kv[1])
    labels = [str(k) for k, _ in ordered]
    counts = [v for _, v in ordered]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.08), 4))
    ax.bar(range(len(counts)), counts)
    ax.set_title(title)
    ax.set_xlabel("class (sorted by frequency)")
    ax.set_ylabel("count")
    if len(labels) <= 40:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def run(args: argparse.Namespace) -> dict:
    output_dir = ensure_dir(args.output_dir)
    df = _read_index(args.index)
    step_log(1, "AnalyzeLTAStats", f"Loaded {len(df)} samples from {args.index}")

    pilot_info = None
    if args.top_verb is not None or args.top_noun is not None:
        top_verb = args.top_verb or df["verb_label"].nunique()
        top_noun = args.top_noun or df["noun_label"].nunique()
        df, pilot_info = build_pilot_taxonomy(df, top_verb=top_verb, top_noun=top_noun, mode=args.pilot_mode)
        step_log(
            1, "AnalyzeLTAStats",
            "*** PILOT TAXONOMY ACTIVE: these stats/metrics are NOT comparable to full-taxonomy "
            f"results (top_verb={top_verb}, top_noun={top_noun}, mode={args.pilot_mode}, "
            f"rows {pilot_info['rows_before']} -> {pilot_info['rows_after']}) ***",
        )

    report = {
        "num_samples": len(df),
        "pilot_taxonomy": pilot_info,
        "verb": _class_report(df, "verb_label"),
        "noun": _class_report(df, "noun_label"),
        "action": _class_report(df, "action_label") if "action_label" in df.columns else None,
        "scenario_distribution": dict(scenario_distribution(df)),
        "boundary_flag_rate": float(df["boundary_flag"].mean()) if "boundary_flag" in df.columns else None,
    }
    write_json(output_dir / "lta_stats.json", report)
    step_log(1, "AnalyzeLTAStats", f"Wrote {output_dir / 'lta_stats.json'}")

    cooc = verb_noun_cooccurrence(df)
    cooc.to_csv(output_dir / "verb_noun_cooccurrence.csv")
    step_log(1, "AnalyzeLTAStats", f"Wrote {output_dir / 'verb_noun_cooccurrence.csv'} (shape={cooc.shape})")

    for col, title in (("verb_label", "Verb frequency"), ("noun_label", "Noun frequency")):
        _save_bar_png(dict(class_frequency(df, col)), title, output_dir / f"{col}_frequency.png")
    _save_bar_png(dict(scenario_distribution(df)), "Scenario distribution", output_dir / "scenario_distribution.png")

    step_log(
        1, "AnalyzeLTAStats",
        f"Verb: {report['verb']['num_classes']} classes, Gini={report['verb']['gini']:.3f}, "
        f"imbalance={report['verb']['imbalance_ratio_max_over_min']:.1f}x",
    )
    step_log(
        1, "AnalyzeLTAStats",
        f"Noun: {report['noun']['num_classes']} classes, Gini={report['noun']['gini']:.3f}, "
        f"imbalance={report['noun']['imbalance_ratio_max_over_min']:.1f}x",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="Path to a Z=1 index parquet/csv (usually train)")
    parser.add_argument("--output-dir", default="outputs/ego4d_lta/stats")
    parser.add_argument("--top-verb", type=int, default=None, help="Pilot taxonomy: keep only top-N verb classes")
    parser.add_argument("--top-noun", type=int, default=None, help="Pilot taxonomy: keep only top-N noun classes")
    parser.add_argument("--pilot-mode", choices=["exclude", "other"], default="exclude")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
