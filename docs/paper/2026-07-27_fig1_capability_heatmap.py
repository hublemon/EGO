# Generates the TikZ body for the capability-profile heatmap (Figure 1).
# Diverging colormap on Delta-vs-Base, clamped to +/-20 pp, with colors
# precomputed in sRGB so the rendered fill is exactly what the caption claims.
import sys

ORANGE = (213, 94, 0)      # profileorange: lower than Base
BLUE = (0, 73, 160)        # measuredblue:  higher than Base
NEUTRAL = (245, 245, 245)  # Delta = 0
SPAN = 20.0                # pp, saturation point
MAXSAT = 0.85              # keep the darkest cell short of full ink
WHITE_TEXT_AT = 0.45       # saturation above which cell text flips to white

GROUPS = [
    ("Grounded selection", [
        (r"SelAcc\\@Top-$K$",          [21.0, 21.9, 28.8, 28.8]),
        (r"$G_1$: retain\\WM-correct", [29.7, 26.5, 37.4, 42.5]),
        (r"$G_2$: correct\\WM-wrong",  [18.7, 20.8, 27.8, 25.6]),
    ]),
    ("Trajectory continuity", [
        (r"Continuation\\recall",      [28.4, 32.9, 43.5, 48.4]),
        (r"Continuation\\precision",   [72.7, 71.3, 71.1, 68.2]),
    ]),
    ("Evidence-conditioned selection", [
        (r"Evidence\\mentioned",       [31.8, 24.1, 38.4, 44.7]),
        (r"Evidence\\absent",          [18.4, 21.0, 26.1, 26.4]),
    ]),
]
ROWS = ["Base", "GT-only", "Cand.-CE", r"\method{}"]

W, H, GAP = 2.05, 0.78, 0.18
X0, TOP = 2.60, 4.42


def rgb(delta):
    t = min(abs(delta) / SPAN, 1.0) * MAXSAT
    end = BLUE if delta > 0 else ORANGE
    return tuple(round(n + t * (e - n)) for n, e in zip(NEUTRAL, end)), t


centers, spans, x = [], [], X0
for _, metrics in GROUPS:
    lo = x
    for _ in metrics:
        centers.append(x + W / 2)
        x += W
    spans.append((lo, x))
    x += GAP
right = spans[-1][1]

flat = [m for _, ms in GROUPS for m in ms]
ys = [TOP - H / 2 - i * H for i in range(len(ROWS))]

out = []
A = out.append
A(r"\begin{tikzpicture}[x=1cm,y=1cm,font=\small]")
A(r"  % Metric groups: booktabs-style spanning rules over their columns.")
for (label, _), (lo, hi) in zip(GROUPS, spans):
    A(r"  \draw[black!40,line width=0.5pt] (%.2f,5.46) -- (%.2f,5.46);" % (lo + 0.02, hi - 0.02))
    A(r"  \node[anchor=south,font=\small\bfseries,text=black!80] at (%.2f,5.54) {%s};"
      % ((lo + hi) / 2, label))
A("")
A(r"  % Metric column headers.")
for (label, _), cx in zip(flat, centers):
    A(r"  \node[anchor=south,align=center,font=\footnotesize,text=black!75] at (%.2f,4.56) {%s};"
      % (cx, label))
A("")
A(r"  % Training conditions (rows); the final model is set in bold.")
for lab, cy in zip(ROWS, ys):
    bold = r"\bfseries" if lab == r"\method{}" else ""
    A(r"  \node[anchor=east,font=\small%s] at (%.2f,%.2f) {%s};" % (bold, X0 - 0.16, cy, lab))
A("")
A(r"  % Cells: raw percentage, signed change from Base, and the Delta-encoded fill.")
for (label, vals), cx in zip(flat, centers):
    base = vals[0]
    for lab, v, cy in zip(ROWS, vals, ys):
        d = v - base
        (r_, g_, b_), t = rgb(d)
        txt = "white" if t >= WHITE_TEXT_AT else "black!88"
        body = "%.1f" % v
        if lab != "Base":
            body += r"\,{\scriptsize %+.1f}" % d
        A(r"  \filldraw[fill={rgb,255:red,%d;green,%d;blue,%d},draw=white,line width=1pt]"
          % (r_, g_, b_))
        A(r"    (%.2f,%.2f) rectangle (%.2f,%.2f);"
          % (cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2))
        A(r"  \node[font=\small,text=%s] at (%.2f,%.2f) {%s};" % (txt, cx, cy, body))
A("")
A(r"  % Continuous colorbar for the shared +/-20 pp scale.")
BX1, BY, BH, NSEG = right, 0.62, 0.26, 48
BX0 = BX1 - 3.60
seg = (BX1 - BX0) / NSEG
for i in range(NSEG):
    d = -SPAN + 2 * SPAN * (i + 0.5) / NSEG
    (r_, g_, b_), _ = rgb(d)
    A(r"  \fill[fill={rgb,255:red,%d;green,%d;blue,%d}] (%.3f,%.2f) rectangle (%.3f,%.2f);"
      % (r_, g_, b_, BX0 + i * seg, BY, BX0 + (i + 1) * seg, BY + BH))
A(r"  \draw[black!30,line width=0.4pt] (%.2f,%.2f) rectangle (%.2f,%.2f);"
  % (BX0, BY, BX1, BY + BH))
for frac, lab in [(0.0, "$-20$"), (0.5, "$0$"), (1.0, "$+20$")]:
    A(r"  \node[anchor=north,font=\scriptsize,text=black!70] at (%.2f,%.2f) {%s};"
      % (BX0 + frac * (BX1 - BX0), BY - 0.04, lab))
A(r"  \node[anchor=east,font=\footnotesize,text=black!70] at (%.2f,%.2f) {$\Delta$ vs.\ Base (pp)};"
  % (BX0 - 0.22, BY + BH / 2))
A(r"\end{tikzpicture}")

print("\n".join(out))
print("grid width %.2f cm (textwidth 17.78 cm); rows y %.2f..%.2f"
      % (right - X0, TOP - len(ROWS) * H, TOP), file=sys.stderr)
