# -*- coding: utf-8 -*-
"""Rebuild Supplementary Figure S6 (Y-scrambling) at final page size.

Same four panels and the same numbers as the published version, redrawn from the
archived permutation output so the axis type is legible at column width. The published
version renders its numerals at about 7.5 pt on the page; the figures the reviewers
accepted sit at 8.5-9.8 pt.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import MODELS, FIG, REP_SEED, REF_MODEL

SRC = os.path.join(MODELS, f"split_seed_{REP_SEED}", "robustness", REF_MODEL)
PERM = pd.read_csv(os.path.join(SRC, "y_scrambling_permutations.csv"))
SCORES = pd.read_csv(os.path.join(SRC, "true_vs_scrambled_scores.csv"))
SUM = json.load(open(os.path.join(SRC, "y_scrambling_summary.json")))

PAGE_W = 5.77
CAT = ["#0072B2", "#D55E00", "#009E73", "#9158A6"]
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


def panel(ax, letter, title=None):
    ax._panel_letter = letter
    if title:
        ax.set_title(title, pad=5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def place_letters(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        letter = getattr(ax, "_panel_letter", None)
        if not letter:
            continue
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(max(bb.x0 - 0.012, 0.002), min(bb.y1 + 0.004, 0.992), f"({letter})",
                 fontsize=11, fontweight="bold", ha="left", va="bottom")


n_perm = int(SUM["n_permutations"])
true_auc, true_mcc = SUM["actual_roc_auc"], SUM["actual_mcc"]
sc_auc_m, sc_auc_s = SUM["perm_roc_auc_mean"], SUM["perm_roc_auc_std"]
sc_mcc_m, sc_mcc_s = SUM["perm_mcc_mean"], SUM["perm_mcc_std"]

fig, axx = plt.subplots(2, 2, figsize=(PAGE_W, 4.55))

ax = axx[0, 0]
ax.hist(PERM.roc_auc, bins=24, color=CAT[0], edgecolor="white", linewidth=0.5)
ax.axvline(true_auc, color=CAT[1], ls="--", lw=1.6)
ax.annotate(f"true label\n{true_auc:.3f}", (true_auc, ax.get_ylim()[1] * 0.92),
            textcoords="offset points", xytext=(-6, 0), ha="right", va="top",
            fontsize=8, color=CAT[1])
ax.set_xlabel("ROC-AUC")
ax.set_ylabel("Permutations")
panel(ax, "A", f"ROC-AUC under {n_perm} scrambles")

ax = axx[0, 1]
ax.hist(PERM.mcc, bins=24, color=CAT[2], edgecolor="white", linewidth=0.5)
ax.axvline(true_mcc, color=CAT[1], ls="--", lw=1.6)
ax.annotate(f"true label\n{true_mcc:.3f}", (true_mcc, ax.get_ylim()[1] * 0.92),
            textcoords="offset points", xytext=(-6, 0), ha="right", va="top",
            fontsize=8, color=CAT[1])
ax.set_xlabel("MCC")
ax.set_ylabel("Permutations")
panel(ax, "B", f"MCC under {n_perm} scrambles")

ax = axx[1, 0]
vals = [true_auc, sc_auc_m, true_mcc, sc_mcc_m]
errs = [0.0, sc_auc_s, 0.0, sc_mcc_s]
cols = [CAT[0], CAT[0], CAT[2], CAT[2]]
alph = [1.0, 0.45, 1.0, 0.45]
x = np.arange(4)
for xi, v, e, c, a in zip(x, vals, errs, cols, alph):
    ax.bar(xi, v, width=0.62, color=c, alpha=a, edgecolor="white", linewidth=0.8,
           yerr=(e if e else None), capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
    ax.text(xi, v + e + 0.045, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
LABELS_C = ["ROC-AUC\ntrue", "ROC-AUC\nscram.", "MCC\ntrue", "MCC\nscram."]
ax.set_xticklabels(LABELS_C, fontsize=7.5)
ax.set_ylabel("Metric value")
ax.set_ylim(0, 1.12)
panel(ax, "C", "True against scrambled")

ax = axx[1, 1]
ax.scatter(SCORES.true_score, SCORES.scrambled_mean_score, s=9, color=CAT[3],
           alpha=0.55, lw=0)
ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=CAT[1])
ax.annotate("identity", (0.62, 0.62), textcoords="offset points", xytext=(4, -12),
            fontsize=8, color=CAT[1])
ax.set_xlabel("True-label probability")
ax.set_ylabel("Mean scrambled probability")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
panel(ax, "D", "Probability structure is lost")

fig.tight_layout()
place_letters(fig)
p = os.path.join(FIG, "FigureS6_y_scrambling.png")
fig.savefig(p)
fig.savefig(os.path.splitext(p)[0] + ".svg")
plt.close(fig)
print("wrote FigureS6_y_scrambling.png and .svg")
print(f"  n_perm {n_perm} | true ROC-AUC {true_auc:.3f} MCC {true_mcc:.3f} "
      f"| scrambled {sc_auc_m:.3f}±{sc_auc_s:.3f}, {sc_mcc_m:.3f}±{sc_mcc_s:.3f}")
print(f"  scrambled mean probability: {SCORES.scrambled_mean_score.mean():.3f} "
      f"(sd {SCORES.scrambled_mean_score.std():.3f})")
