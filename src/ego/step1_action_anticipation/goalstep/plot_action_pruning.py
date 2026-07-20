"""Long-tail pruning trade-off for GoalStep action classes.

Question it answers: if we DROP every action class whose total support is <= k,
how many classes do we lose, and how much of the segment mass do we lose?

Both quantities are percentages, so they share ONE y-axis (never a dual axis):
  * classes retained (%)   -- how much of the label space survives
  * segment coverage (%)   -- how much annotated data survives

Also writes the underlying table as CSV so the threshold decision is auditable.

Usage:
    python src/ego/step1_action_anticipation/goalstep/plot_action_pruning.py \
        --input src/ego/step1_action_anticipation/goalstep/taxonomy/action_classes.csv \
        --output src/ego/step1_action_anticipation/goalstep/taxonomy/action_pruning_tradeoff.png --kmax 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

SURFACE = "#fcfcfb"
SERIES_SEG = "#2a78d6"     # slot 1 blue  -- segment coverage
SERIES_CLS = "#008300"     # slot 2 green -- classes retained
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, GRID = "#0b0b0b", "#52514e", "#78776f", "#e6e5e1"
MARKS = [5, 10, 20, 30]     # candidate thresholds to call out


def pick_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for c in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK JP", "DejaVu Sans"):
        if c in names:
            return c
    return "DejaVu Sans"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="src/ego/step1_action_anticipation/goalstep/taxonomy/action_classes.csv")
    p.add_argument("--output", default="src/ego/step1_action_anticipation/goalstep/taxonomy/figures/action_pruning_tradeoff.png")
    p.add_argument("--csv-out", default="src/ego/step1_action_anticipation/goalstep/taxonomy/action_pruning_table.csv")
    p.add_argument("--kmax", type=int, default=60)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    tot = [int(r["total_count"]) for r in rows]
    N, S = len(tot), sum(tot)

    ks = list(range(0, args.kmax + 1))
    keep_cls, keep_seg, table = [], [], []
    for k in ks:
        kept = [t for t in tot if t > k]
        keep_cls.append(100 * len(kept) / N)
        keep_seg.append(100 * sum(kept) / S)
        table.append({"k": k, "dropped_classes": N - len(kept), "kept_classes": len(kept),
                      "lost_segments": S - sum(kept), "kept_segments": sum(kept),
                      "lost_pct": round(100 * (S - sum(kept)) / S, 3),
                      "coverage_pct": round(100 * sum(kept) / S, 3)})
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)

    plt.rcParams.update({"font.family": pick_font(), "axes.unicode_minus": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "text.color": TEXT_PRIMARY, "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.745, bottom=0.13)

    ax.plot(ks, keep_seg, lw=2, color=SERIES_SEG, label="남는 세그먼트 (커버리지 %)", zorder=3)
    ax.plot(ks, keep_cls, lw=2, color=SERIES_CLS, label="남는 클래스 (%)", zorder=3)
    for k in MARKS:
        if k > args.kmax:
            continue
        ax.plot([k], [keep_seg[k]], "o", ms=8, color=SERIES_SEG, mec=SURFACE, mew=2, zorder=4)
        ax.plot([k], [keep_cls[k]], "o", ms=8, color=SERIES_CLS, mec=SURFACE, mew=2, zorder=4)
        ax.axvline(k, color=GRID, lw=0.8, zorder=1)
        ax.text(k, 103, f"k={k}", ha="center", fontsize=9, color=TEXT_SECONDARY)
        ax.text(k + 0.6, keep_seg[k] - 4.5, f"{keep_seg[k]:.1f}%", fontsize=8.8, color=TEXT_MUTED)
        ax.text(k + 0.6, keep_cls[k] + 2.0, f"{keep_cls[k]:.0f}%", fontsize=8.8, color=TEXT_MUTED)

    ax.set_ylim(0, 108)
    ax.set_xlim(0, args.kmax)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], fontsize=9.5)
    ax.set_xlabel("임계값 k — 총 세그먼트 수가 k 이하인 action 클래스를 제거", fontsize=10, labelpad=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=9.5)
    ax.legend(frameon=False, fontsize=10, loc="lower left", handlelength=1.6)

    fig.suptitle("action 클래스 롱테일 가지치기 트레이드오프", x=0.085, y=0.965,
                 ha="left", va="top", fontsize=15.5, fontweight="bold")
    fig.text(0.085, 0.862,
             f"전체 {N}개 클래스 · {S:,} 세그먼트 — 클래스는 크게 줄어도 세그먼트 손실은 매우 작다\n"
             f"(k=10이면 클래스 {table[10]['dropped_classes']}개를 버리고 세그먼트는 {table[10]['lost_pct']:.2f}%만 손실)",
             ha="left", va="top", fontsize=10.2, color=TEXT_SECONDARY, linespacing=1.55)

    fig.savefig(args.output, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.output}\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
