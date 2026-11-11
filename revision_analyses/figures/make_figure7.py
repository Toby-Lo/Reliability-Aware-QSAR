# -*- coding: utf-8 -*-
"""Rebuild main-text Figure 7 (time-resolved PCA projections) at final page size.

The published panel used axis numerals that render at about 3.7 pt once the image is
placed at column width, against 8.5-9.8 pt for the figures the reviewers accepted. The
figure is redrawn here from the same per-system PCA_projection.dat files, at the width
it is actually printed at, with shared axis labels and fewer ticks so the numerals have
room. Component labels read PC1/PC2 to match the caption and the Results text.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import PROJ, FIG

SYSD = os.path.join(PROJ, "mds", "Results")
PAGE_W = 5.77
INK, MUTED = "#1a1a1a", "#555555"

# panel order as published: reference first, then the eight candidates
ORDER = [("a1a0m", "A1A0M"),
         ("18198196", "Z20D-18198196"),
         ("18198198", "Z20D-18198198"),
         ("18198200", "Z20D-18198200"),
         ("36354596", "Z20D-36354596"),
         ("58175203", "Z20D-58175203"),
         ("219210673", "Z20D-219210673"),
         ("1857797734", "Z20D-1857797734"),
         ("1857797811", "Z20D-1857797811")]
LETTERS = "ABCDEFGHI"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 9.5, "axes.titlesize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.grid": False,
})

TOTAL_NS = 200.0

fig, axes = plt.subplots(3, 3, figsize=(PAGE_W, 5.35))
sc = None
missing = []
for k, ((sysname, label), letter) in enumerate(zip(ORDER, LETTERS)):
    ax = axes.ravel()[k]
    path = os.path.join(SYSD, f"{sysname}-Analysis", "PCA", "projection.txt")
    if not os.path.exists(path):
        missing.append(sysname)
        ax.set_axis_off()
        continue
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                     names=["frame", "pc1", "pc2", "pc3"])
    t = (df.frame.to_numpy() / df.frame.max()) * TOTAL_NS
    sc = ax.scatter(df.pc1, df.pc2, c=t, cmap="plasma_r", s=0.8, lw=0,
                    vmin=0, vmax=TOTAL_NS, rasterized=True)
    ax.text(0.03, 0.97, f"({letter}) {label}", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.82))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(length=2.5, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.supxlabel("PC1", fontsize=9.5, y=0.045)
fig.supylabel("PC2", fontsize=9.5, x=0.035)
fig.tight_layout(rect=(0.03, 0.04, 0.90, 1.0))

if sc is not None:
    cax = fig.add_axes([0.925, 0.12, 0.022, 0.76])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Simulation time (ns)", fontsize=9)
    cb.ax.tick_params(labelsize=7.5, length=2.5)
    cb.set_ticks([0, 50, 100, 150, 200])

p = os.path.join(FIG, "Figure7_pca_projections.png")
fig.savefig(p)
fig.savefig(os.path.splitext(p)[0] + ".svg")
plt.close(fig)
print("wrote Figure7_pca_projections.png and .svg")
if missing:
    print("MISSING PCA data for:", missing)
