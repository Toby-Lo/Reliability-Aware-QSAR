"""Supplementary Figure S14: ligand-based and docking enrichment against matched decoys."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import ANA, TAB, ROOT

OUT = os.path.join(ROOT, "Revision_R1", "figures")
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


S = pd.read_csv(os.path.join(ANA, "R11_scores_for_curves.csv"))
B = pd.read_csv(os.path.join(TAB, "R11_decoy_benchmark_summary.csv"), index_col=0)
dockp = os.path.join(ANA, "R13_summary.json")
DOCK = json.load(open(dockp)) if os.path.exists(dockp) else None
dsc = os.path.join(ROOT, "Revision_R1", "docking_enrichment", "docking_scores.csv")
D = pd.read_csv(dsc) if os.path.exists(dsc) else None

if DOCK:
    fig, _ax = plt.subplots(2, 2, figsize=(5.77, 5.00))
    axes = _ax.ravel()
else:
    fig, axes = plt.subplots(1, 2, figsize=(5.77, 3.05))

# (A) ligand-based ROC, one faint curve per partition plus the pooled curve
ax = axes[0]
for seed, g in S.groupby("split_seed"):
    fpr, tpr, _ = roc_curve(g.y, g.score)
    ax.plot(fpr, tpr, lw=0.9, color="#b9b9b9", zorder=2)
fpr, tpr, _ = roc_curve(S.y, S.score)
ax.plot(fpr, tpr, lw=2.2, color=CAT[0], zorder=4,
        label=f"pooled (AUC {roc_auc_score(S.y, S.score):.3f})")
ax.plot([0, 1], [0, 1], ls=":", lw=1, color=MUTED, zorder=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right")
panel(ax, "A", "Ligand-based discrimination")

# (B) enrichment metrics
ax = axes[1]
keys = [("ef1", "EF 1%"), ("ef5", "EF 5%"), ("ef10", "EF 10%")]
vals = [B.loc[k, "mean"] for k, _ in keys]
errs = [B.loc[k, "sd"] for k, _ in keys]
x = np.arange(len(keys))
ax.bar(x, vals, yerr=errs, width=0.55, color=CAT[0], capsize=4,
       edgecolor="white", linewidth=0.8, error_kw=dict(ecolor=MUTED, lw=1))
for xi, v, e in zip(x, vals, errs):
    ax.text(xi, v + e + max(vals) * 0.05, f"{v:.1f}", ha="center", fontsize=8.5)
ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.18)
ax.axhline(1.0, color=MUTED, ls=":", lw=1)
# the dotted line marks random expectation; it is explained in the caption rather than
# labeled in the panel, which would collide with the first bar
ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.18)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys])
ax.set_ylabel("Enrichment factor")
panel(ax, "B", "Enrichment, ligand-based")

if DOCK and D is not None:
    ok = D.dropna(subset=["vina_score"])
    body = ok[ok.name != "reference_A1A0M"]
    y = body.label.values.astype(int)
    s = -body.vina_score.values
    ax = axes[2]
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, lw=2.2, color=CAT[1], label=f"AUC {roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], ls=":", lw=1, color=MUTED)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    panel(ax, "C", "Docking enrichment (9CVD)")

    ax = axes[3]
    bins = np.linspace(min(body.vina_score), max(body.vina_score), 28)
    ax.hist(body.vina_score[y == 0], bins=bins, density=True, color="#b9b9b9",
            label=f"decoys (n = {int((y==0).sum())})")
    ax.hist(body.vina_score[y == 1], bins=bins, density=True, color=CAT[1], alpha=0.75,
            label=f"actives (n = {int(y.sum())})")
    ax.set_xlabel("Vina score (kcal/mol)"); ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    rf = ok[ok.name == "reference_A1A0M"]
    if len(rf):
        ax.axvline(rf.vina_score.iloc[0], color=INK, ls="--", lw=1.2)
        ax.annotate("A1A0M", (rf.vina_score.iloc[0], ax.get_ylim()[1] * 0.55),
                    textcoords="offset points", xytext=(5, 0), fontsize=7.5, color=INK)
    panel(ax, "D", "Score distributions")

fig.tight_layout()
_place_letters(fig)
p = os.path.join(OUT, "FigureS14_enrichment.png")
fig.savefig(p)
fig.savefig(os.path.splitext(p)[0] + ".svg")
plt.close(fig)
print("wrote FigureS14_enrichment.png and .svg  (docking panels:", DOCK is not None, ")")
