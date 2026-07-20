"""Plot the GoalStep action-class distribution from action_classes.csv.

Two bar panels (matplotlib):
  (1) Top-N action classes by support, horizontal STACKED bars (train + val).
      Horizontal because action labels are long; stacked because train/val are
      parts of each class's total.
  (2) Support histogram over ALL action classes -- how many classes fall in each
      frequency bucket (the long-tail view), single-hue columns.

Design follows the project's dataviz rules: validated 2-hue categorical palette
(blue/green; passes CVD + normal-vision + contrast gates in light mode), thin
marks with a surface gap between stacked segments, hairline recessive gridlines,
a legend for the two series, and values at the bar tips.

Usage:
    python src/ego/step1_action_anticipation/goalstep/plot_action_classes.py \
        --input src/ego/step1_action_anticipation/goalstep/taxonomy/action_classes.csv \
        --output src/ego/step1_action_anticipation/goalstep/taxonomy/action_classes_distribution.png --top 25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

# Validated categorical palette (light surface) -- see dataviz palette reference.
SURFACE = "#fcfcfb"
SERIES_TRAIN = "#2a78d6"   # slot 1 blue
SERIES_VAL = "#008300"     # slot 2 green
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#78776f"
GRID = "#e6e5e1"

BUCKETS = [("1", 1, 1), ("2–5", 2, 5), ("6–20", 6, 20),
           ("21–100", 21, 100), ("101–500", 101, 500), ("501+", 501, 10**9)]


def pick_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK JP", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["train"], r["val"] = int(r["train_count"]), int(r["val_count"])
            r["total"] = int(r["total_count"])
            rows.append(r)
    rows.sort(key=lambda r: -r["total"])
    return rows


def plot(rows, out_path, top_n):
    font = pick_font()
    plt.rcParams.update({
        "font.family": font, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_SECONDARY,
    })
    top = rows[:top_n]
    total_segments = sum(r["total"] for r in rows)

    fig = plt.figure(figsize=(11.5, 12.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.45, 1], hspace=0.30,
                          left=0.24, right=0.965, top=0.875, bottom=0.075)

    # ---------- Panel 1: top-N stacked horizontal bars ----------
    ax = fig.add_subplot(gs[0])
    labels = [r["action_label"] for r in top][::-1]
    tr = [r["train"] for r in top][::-1]
    va = [r["val"] for r in top][::-1]
    tot = [r["total"] for r in top][::-1]
    y = range(len(top))
    gap = max(tot) * 0.004  # 2px-equivalent surface gap between stacked segments

    ax.barh(y, tr, height=0.62, color=SERIES_TRAIN, label="train", zorder=3)
    ax.barh(y, va, height=0.62, left=[t + gap for t in tr], color=SERIES_VAL,
            label="val", zorder=3)
    for i, t in enumerate(tot):
        ax.text(t + max(tot) * 0.012, i, f"{t:,}", va="center", ha="left",
                fontsize=8.6, color=TEXT_MUTED)

    ax.set_yticks(list(y), labels, fontsize=9.4)
    ax.set_xlim(0, max(tot) * 1.10)
    ax.set_xlabel("세그먼트 수 (train + val)", fontsize=9.5, labelpad=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", ncol=2,
              handlelength=1.0, handleheight=1.0, borderpad=0.2)
    ax.set_title(f"상위 {top_n}개 action 클래스 (지원 세그먼트 수)",
                 fontsize=12.5, fontweight="semibold", pad=16, loc="left")

    # ---------- Panel 2: support-bucket histogram over all classes ----------
    ax2 = fig.add_subplot(gs[1])
    names = [b[0] for b in BUCKETS]
    counts = [sum(1 for r in rows if lo <= r["total"] <= hi) for _, lo, hi in BUCKETS]
    ax2.bar(names, counts, width=0.56, color=SERIES_TRAIN, zorder=3)
    for i, c in enumerate(counts):
        ax2.text(i, c + max(counts) * 0.03, str(c), ha="center", va="bottom",
                 fontsize=9.6, fontweight="semibold", color=TEXT_PRIMARY)
    ax2.set_ylim(0, max(counts) * 1.16)
    ax2.set_xlabel("클래스당 총 세그먼트 수 (지원 구간)", fontsize=9.5, labelpad=8)
    ax2.set_ylabel("action 클래스 수", fontsize=9.5, labelpad=10)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(axis="both", length=0, labelsize=9.4)
    ax2.set_title(f"전체 {len(rows)}개 action 클래스의 지원 분포 (롱테일)",
                  fontsize=12.5, fontweight="semibold", pad=16, loc="left")

    top10 = sum(r["total"] for r in rows[:10])
    med = sorted(r["total"] for r in rows)[len(rows) // 2]
    fig.suptitle("Ego4D GoalStep — action 클래스 (verb × noun) 분포", x=0.055,
                 y=0.968, ha="left", fontsize=15.5, fontweight="bold")
    fig.text(0.055, 0.930,
             f"{len(rows)}개 클래스 · {total_segments:,} 세그먼트 · "
             f"상위 10개 점유율 {100*top10/total_segments:.1f}% · 중앙값 지원 {med} · "
             f"1회 등장 {sum(1 for r in rows if r['total']==1)}개",
             ha="left", fontsize=10, color=TEXT_SECONDARY)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"wrote {out_path}")
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="src/ego/step1_action_anticipation/goalstep/taxonomy/action_classes.csv")
    p.add_argument("--output", default="src/ego/step1_action_anticipation/goalstep/taxonomy/figures/action_classes_distribution.png")
    p.add_argument("--top", type=int, default=25, help="how many top classes in panel 1")
    args = p.parse_args()
    rows = load(args.input)
    counts = plot(rows, args.output, args.top)
    print(f"classes={len(rows)}  bucket counts={dict(zip([b[0] for b in BUCKETS], counts))}")


if __name__ == "__main__":
    main()
