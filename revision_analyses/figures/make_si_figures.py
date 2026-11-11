"""Generate the new supplementary figures for the revision.

Categorical palette is the Okabe-Ito-derived set validated with the dataviz
validator (all six checks pass, worst adjacent CVD delta-E 11.0): hues are assigned in
fixed order and never cycled.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import ANA, ROOT

OUT = os.path.join(ROOT, "Revision_R1", "figures")
os.makedirs(OUT, exist_ok=True)

CAT = ["#0072B2", "#D55E00", "#009E73", "#9158A6"]   # fixed order, validated
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d8d8d8"

# Figures are drawn at the width they are placed on the page (5.77 in), so every
# point size below is the size the reader actually sees.
PAGE_W = 5.77
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


def save(fig, name):
    """Write both a raster and a vector copy of every figure."""
    fig.tight_layout()
    _place_letters(fig)
    p = os.path.join(OUT, name)
    fig.savefig(p)
    svg = os.path.splitext(p)[0] + ".svg"
    fig.savefig(svg)
    plt.close(fig)
    print("wrote", name, "and", os.path.basename(svg))


# ---------------------------------------------------------------- Figure S8
dec = pd.read_csv(os.path.join(ANA, "R04_ad_decile_profile.csv"))
swp = pd.read_csv(os.path.join(ANA, "R04_ad_threshold_sweep.csv"))

fig = plt.figure(figsize=(PAGE_W, 4.45))
_gs = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(_gs[0, 0]), fig.add_subplot(_gs[0, 1]),
        fig.add_subplot(_gs[1, :])]
ax = axes[0]
ax.errorbar(dec.decile, dec.logloss_mean,
            yerr=[dec.logloss_mean - dec.ci_low, dec.ci_high - dec.logloss_mean],
            fmt="o-", color=CAT[0], lw=2, ms=5, capsize=3, elinewidth=1)
ax.set_xlabel("Applicability-domain decile")
ax.set_ylabel("Mean per-compound log loss")
ax.set_xticks(range(1, 11))
panel(ax, "A", "Error falls with domain support")

ax = axes[1]
ax.plot(dec.decile, dec.accuracy, "s-", color=CAT[2], lw=2, ms=5)
ax.set_xlabel("Applicability-domain decile")
ax.set_ylabel("Classification accuracy")
ax.set_xticks(range(1, 11))
ax.set_ylim(0.70, 1.02)
panel(ax, "B", "Accuracy rises with domain support")

ax = axes[2]
ax.plot(swp.mean_coverage * 100, swp.relative_reduction * 100, "o-",
        color=CAT[1], lw=2, ms=5)
ax.fill_between(swp.mean_coverage * 100, swp.red_ci_low * 100, swp.red_ci_high * 100,
                color=CAT[1], alpha=0.18, lw=0)
op = swp[np.isclose(swp.ad_threshold, 0.30)]
if len(op):
    ax.scatter(op.mean_coverage * 100, op.relative_reduction * 100, s=70,
               facecolor="white", edgecolor=CAT[1], zorder=5, linewidth=1.8)
    ax.annotate("operating gate\n$S_{AD}$ = 0.30",
                (float(op.mean_coverage.iloc[0]) * 100, float(op.relative_reduction.iloc[0]) * 100),
                textcoords="offset points", xytext=(-6, 18), ha="right", fontsize=8, color=MUTED)
ax.set_xlabel("External compounds retained (%)")
ax.set_ylabel("Reduction in mean log loss (%)")
panel(ax, "C", "Coverage against error reduction")
fig.tight_layout()
save(fig, "FigureS8_ad_error_stratification.png")

# ---------------------------------------------------------------- Figure S9
lc = pd.read_csv(os.path.join(ANA, "R08_learning_curve_raw.csv"))
agg = lc.groupby("frac").agg(n=("n_train", "mean"), mcc=("mcc", "mean"), mcc_sd=("mcc", "std"),
                             roc=("roc_auc", "mean"), roc_sd=("roc_auc", "std")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 3.05))
for ax, col, sd, lab, c, letter in [
        (axes[0], "mcc", "mcc_sd", "External MCC", CAT[0], "A"),
        (axes[1], "roc", "roc_sd", "External ROC-AUC", CAT[2], "B")]:
    ax.plot(agg.n, agg[col], "o-", color=c, lw=2, ms=5)
    ax.fill_between(agg.n, agg[col] - agg[sd], agg[col] + agg[sd], color=c, alpha=0.16, lw=0)
    ax.set_xlabel("Training compounds")
    ax.set_ylabel(lab)
    panel(ax, letter)
axes[0].axhline(agg[agg.frac == 1.0].mcc.iloc[0], color=MUTED, ls=":", lw=1)
axes[0].text(140, agg[agg.frac == 1.0].mcc.iloc[0] + 0.004, "full development set",
             fontsize=7.5, color=MUTED)
fig.tight_layout()
save(fig, "FigureS9_learning_curve.png")

# ---------------------------------------------------------------- Figure S13
abl = pd.read_csv(os.path.join(ANA, "R05_ablation_per_partition.csv"))
sh_full = pd.read_csv(os.path.join(ANA, "R05b_shap_full.csv")).head(12)
sh_free = pd.read_csv(os.path.join(ANA, "R05b_shap_size_free.csv")).head(12)
order = ["Full descriptor panel (published)", "Size proxies removed", "Fingerprints only"]
short = ["Full panel\n(20 descriptors)", "Size proxies\nremoved (13)", "Fingerprints\nonly"]

fig = plt.figure(figsize=(PAGE_W, 4.70))
_gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3])
axes = [fig.add_subplot(_gs[0, :]), fig.add_subplot(_gs[1, 0]),
        fig.add_subplot(_gs[1, 1])]
ax = axes[0]
for i, cfg in enumerate(order):
    v = abl[abl.config == cfg].mcc.values
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.08, .08, len(v)),
               v, s=18, color=CAT[i], alpha=0.65, zorder=3, edgecolor="none")
    ax.plot([i - .22, i + .22], [v.mean()] * 2, color=CAT[i], lw=2.5, zorder=4)
    ax.text(i, v.mean() + 0.055, f"{v.mean():.3f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(short)
ax.set_ylabel("External MCC")
panel(ax, "A", "Performance without size descriptors")

for ax, tab, lab, letter, c in [(axes[1], sh_full, "Full descriptor panel", "B", CAT[0]),
                                (axes[2], sh_free, "Size proxies removed", "C", CAT[2])]:
    y = np.arange(len(tab))[::-1]
    ax.barh(y, tab.mean_abs_shap, height=0.62, color=c, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(tab.feature, fontsize=7.5)
    ax.set_xlabel("Mean |SHAP|")
    ax.grid(axis="y", visible=False)
    panel(ax, letter, lab)
fig.tight_layout()
save(fig, "FigureS13_size_ablation_shap.png")

# ---------------------------------------------------------------- Figure S21
conv = pd.read_csv(os.path.join(ROOT, "Revision_R1", "tables", "R09_md_convergence.csv"))
blk = pd.read_csv(os.path.join(ANA, "R09_block_averages.csv"))
conv = conv.sort_values("ess_ca", ascending=False)
fig = plt.figure(figsize=(PAGE_W, 4.55))
_gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])
axes = [fig.add_subplot(_gs[0, :]), fig.add_subplot(_gs[1, 0]),
        fig.add_subplot(_gs[1, 1])]

ax = axes[0]
# nine systems exceed the categorical palette, so the set is drawn as a neutral
# background and only the reference and the two extremes carry hue and a direct label
hi = {"A1A0M": CAT[0], "Z20D-1857797811": CAT[2], "Z20D-36354596": CAT[1]}
for s in conv.system:
    g = blk[blk.system == s].sort_values("block")
    x, y = g.block * 25 - 12.5, g.ca_rmsd_mean
    if s in hi:
        ax.plot(x, y, "-o", ms=4, lw=2, color=hi[s], zorder=4)
        ax.annotate(s.replace("Z20D-", ""), (x.iloc[-1], y.iloc[-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=7,
                    color=hi[s], va="center")
    else:
        ax.plot(x, y, "-", lw=1, color="#b9b9b9", zorder=2)
ax.set_xlim(0, 240)
ax.set_xlabel("Trajectory block midpoint (ns)")
ax.set_ylabel(r"Block-averaged C$\alpha$ RMSD (${\AA}$)")
panel(ax, "A", "Block averages over 25 ns windows")
ax.legend(handles=[Line2D([], [], color="#b9b9b9", lw=1, label="other complexes")],
          loc="upper left", fontsize=7.5)

ax = axes[1]
y = np.arange(len(conv))[::-1]
ax.barh(y, conv.ess_ca, height=0.6, color=CAT[0], edgecolor="white", linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels(conv.system, fontsize=7.5)
ax.set_xlabel("Effective independent samples")
ax.grid(axis="y", visible=False)
panel(ax, "B", "Sampling over 15–200 ns")

ax = axes[2]
ax.barh(y, conv.cos_pc1, height=0.6, color=CAT[1], edgecolor="white", linewidth=0.6)
ax.axvline(0.5, color=MUTED, ls=":", lw=1)
ax.text(0.52, len(conv) - 0.6, "diffusion-dominated", fontsize=7, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(conv.system, fontsize=7)
ax.set_xlabel("Cosine content of PC1")
ax.set_xlim(0, 1)
ax.grid(axis="y", visible=False)
panel(ax, "C", "Character of the leading mode")
fig.tight_layout()
save(fig, "FigureS21_md_convergence.png")

print("\nsupplementary figures complete")
