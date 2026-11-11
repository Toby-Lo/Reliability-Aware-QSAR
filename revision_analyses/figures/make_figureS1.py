"""Supplementary Figure S1: sensitivity of the multi-objective ranking to its weights.

Sequential magnitude uses a single hue, light to dark, per the visualization rules; the
two-series panel carries a legend and direct labels.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import ANA, ROOT

OUT = os.path.join(ROOT, "Revision_R1", "figures")
CAT = ["#0072B2", "#D55E00"]
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d8d8d8"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "legend.frameon": False, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.alpha": 0.7,
})


def _place_letters(fig):
    """Panel letters sit outside each axes, placed once the layout is final."""
    fig.canvas.draw()
    for ax in fig.axes:
        letter = getattr(ax, "_panel_letter", None)
        if not letter:
            continue
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(max(bb.x0 - 0.012, 0.002), min(bb.y1 + 0.004, 0.992), f"({letter})",
                 fontsize=11, fontweight="bold", ha="left", va="bottom")


def panel(ax, letter, title=None):
    ax._panel_letter = letter
    if title:
        ax.set_title(title, pad=5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


W = pd.read_csv(os.path.join(ANA, "R12_weight_simplex_full.csv"))
TOPN = [10, 25, 50, 100, 250, 500]

fig, axes = plt.subplots(1, 2, figsize=(5.77, 3.20))

# (A) Top-100 overlap across the whole simplex
ax = axes[0]
step = 0.05
wp = np.round(np.arange(0, 1.0001, step), 2)
wa = np.round(np.arange(0, 1.0001, step), 2)
M = np.full((len(wa), len(wp)), np.nan)
for _, r in W.iterrows():
    i = int(round(r.wA / step)); j = int(round(r.wP / step))
    if 0 <= i < len(wa) and 0 <= j < len(wp):
        M[i, j] = r.overlap_top100 * 100
im = ax.imshow(M, origin="lower", cmap="Blues", vmin=20, vmax=100,
               extent=[-step/2, 1 + step/2, -step/2, 1 + step/2], aspect="auto")
ax.scatter([0.5], [0.3], s=110, facecolor="none", edgecolor="#B8340A", linewidth=2.0, zorder=5)
ax.annotate("published\n0.5 / 0.2 / 0.3", (0.5, 0.3), xytext=(0.70, 0.66),
            fontsize=7.5, color="#B8340A", ha="left",
            arrowprops=dict(arrowstyle="-", color="#B8340A", lw=0.9))
import matplotlib.patches as mpatches
ax.add_patch(mpatches.Rectangle((0.35, 0.15), 0.30, 0.30, fill=False,
                                edgecolor="#B8340A", linewidth=1.0, linestyle="--", zorder=4))
ax.set_xlabel("Weight on calibrated probability, $w_P$")
ax.set_ylabel("Weight on ADMET desirability, $w_A$")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Top-100 overlap with published ranking (%)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
panel(ax, "A", "Whole weight simplex")

# (B) overlap by candidate-set size, all schemes vs the plausible neighborhood
ax = axes[1]
nb = W[(W.wP.between(0.35, 0.65)) & (W.wR.between(0.05, 0.35)) & (W.wA.between(0.15, 0.45))]
x = np.arange(len(TOPN))
for df, lab, c in [(W, f"all {len(W)} schemes", CAT[0]),
                   (nb, f"within ±0.15 of published ({len(nb)})", CAT[1])]:
    med = [df[f"overlap_top{n}"].median() * 100 for n in TOPN]
    lo = [df[f"overlap_top{n}"].min() * 100 for n in TOPN]
    ax.plot(x, med, "o-", color=c, lw=2, ms=5, label=lab)
    ax.fill_between(x, lo, med, color=c, alpha=0.16, lw=0)
ax.set_xticks(x); ax.set_xticklabels([f"Top {n}" for n in TOPN])
ax.set_ylabel("Overlap with published ranking (%)")
ax.set_ylim(0, 103)
ax.text(0.02, 0.04, "shaded band spans the median down to the worst scheme",
        transform=ax.transAxes, fontsize=7.5, color=MUTED)
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
panel(ax, "B", "Stability of the candidate sets")

fig.tight_layout()
_place_letters(fig)
p = os.path.join(OUT, "FigureS1_weight_sensitivity.png")
fig.savefig(p)
fig.savefig(os.path.splitext(p)[0] + ".svg")
plt.close(fig)
print("wrote FigureS1_weight_sensitivity.png and .svg")
print(f"neighborhood schemes: {len(nb)}; Top-100 min {nb.overlap_top100.min():.2f}, "
      f"median {nb.overlap_top100.median():.2f}")
