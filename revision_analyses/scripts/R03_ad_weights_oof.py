"""R1 comment 2 / R2 comment 3: choose the AD weights w1-w4 without touching the external set.

For every scaffold partition the AD terms are recomputed out-of-fold inside the
development set (each CV fold is scored against the remaining folds only). The
(w1, w3) grid is then searched against development out-of-fold error, and the
selected configuration is applied once to the untouched external set.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
from R_ad import *

GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PUB_W1, PUB_W3 = 0.9, 0.05

def oof_terms_for_seed(sd, seed):
    """AD terms for every development compound, computed against out-of-fold training data."""
    dev = load_split(sd, "dev_train")
    m = fp_mask(sd).astype(bool)
    fp = dev["fp"][:, m]; desc = dev["desc"]
    folds = []
    for p in sorted(glob.glob(os.path.join(sd, "predictions", "cv_predictions_fold_*.csv"))):
        f = pd.read_csv(p)
        folds.append(f[f.model == REF_MODEL])
    oof = pd.concat(folds, ignore_index=True)
    oof = oof.drop_duplicates(subset=["id"], keep="first")
    pos = {i: k for k, i in enumerate(dev["ids"])}
    oof = oof[oof.id.isin(pos)].copy()
    oof["row"] = oof.id.map(pos)
    tan = np.full(len(dev["ids"]), np.nan); cos = np.full_like(tan, np.nan)
    dens = np.full_like(tan, np.nan); lev = np.full_like(tan, np.nan)
    for fold, g in oof.groupby("fold"):
        q = g.row.values
        tr = np.setdiff1d(np.arange(len(dev["ids"])), q)
        t = ad_terms(fp[tr], desc[tr], fp[q], desc[q])
        tan[q] = t["tanimoto"]; cos[q] = t["cosine"]; dens[q] = t["density"]; lev[q] = t["leverage"]
    ok = ~np.isnan(tan)
    err = np.full_like(tan, np.nan)
    err[oof.row.values] = per_sample_logloss(oof.y_true.values, oof.y_prob.values)
    return dict(tanimoto=tan[ok], cosine=cos[ok], density=dens[ok], leverage=lev[ok],
                error=err[ok], n=int(ok.sum()))


def grid_search(terms, err):
    rows = []
    for w1 in GRID:
        sim = w1 * terms["tanimoto"] + (1 - w1) * terms["cosine"]
        for w3 in GRID:
            s = np.clip(w3 * sim + (1 - w3) * terms["density"], 0, 1)
            if np.std(s) < 1e-9: continue
            rows.append(dict(w1=w1, w2=round(1 - w1, 2), w3=w3, w4=round(1 - w3, 2),
                             pearson=pearsonr(s, err)[0], spearman=spearmanr(s, err)[0]))
    return pd.DataFrame(rows)

all_grids, sel_rows = [], []
for seed, sd in sorted(seed_dirs().items()):
    T = oof_terms_for_seed(sd, seed)
    g = grid_search(T, T["error"])
    g["split_seed"] = seed
    all_grids.append(g)
    best = g.loc[g.spearman.idxmin()]
    # external application of the OOF-selected weights (test set used once, for reporting)
    dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
    m = fp_mask(sd).astype(bool)
    te = ad_terms(dev["fp"][:, m], dev["desc"], ext["fp"][:, m], ext["desc"])
    pr = pd.read_csv(os.path.join(sd, "predictions", "external_test_predictions.csv"))
    pr = pr[pr.model == REF_MODEL].set_index("id").reindex(ext["ids"])
    e_ext = per_sample_logloss(pr.y_true.values, pr.y_prob.values)
    s_sel = ad_score(te, best.w1, best.w3)
    s_pub = ad_score(te, PUB_W1, PUB_W3)
    sel_rows.append(dict(split_seed=seed, n_oof=T["n"],
        oof_w1=best.w1, oof_w2=best.w2, oof_w3=best.w3, oof_w4=best.w4,
        oof_spearman=best.spearman, oof_pearson=best.pearson,
        ext_spearman_oofsel=spearmanr(s_sel, e_ext)[0], ext_pearson_oofsel=pearsonr(s_sel, e_ext)[0],
        ext_spearman_pub=spearmanr(s_pub, e_ext)[0], ext_pearson_pub=pearsonr(s_pub, e_ext)[0],
        rank_corr_between_scores=spearmanr(s_sel, s_pub)[0]))
    print(f"seed {seed:>6}: OOF-selected w1={best.w1:.2f} w3={best.w3:.2f} "
          f"(OOF rho={best.spearman:+.3f}) -> external rho={sel_rows[-1]['ext_spearman_oofsel']:+.3f} "
          f"| published config external rho={sel_rows[-1]['ext_spearman_pub']:+.3f}")

G = pd.concat(all_grids, ignore_index=True)
G.to_csv(os.path.join(ANA, "R03_ad_weight_grid_all_seeds.csv"), index=False)
S = pd.DataFrame(sel_rows)
S.to_csv(os.path.join(ANA, "R03_ad_weight_selection_per_seed.csv"), index=False)

# pooled grid: mean OOF Spearman over the 10 partitions
pooled = G.groupby(["w1","w2","w3","w4"], as_index=False).agg(
    oof_spearman_mean=("spearman","mean"), oof_spearman_sd=("spearman","std"),
    oof_pearson_mean=("pearson","mean"))
pooled = pooled.sort_values("oof_spearman_mean")
pooled.to_csv(os.path.join(ANA, "R03_ad_weight_grid_pooled.csv"), index=False)
best_pooled = pooled.iloc[0]
print(f"\nPOOLED best over 10 partitions: w1={best_pooled.w1} w2={best_pooled.w2} "
      f"w3={best_pooled.w3} w4={best_pooled.w4}  mean OOF rho={best_pooled.oof_spearman_mean:+.4f}")
pub_row = pooled[(pooled.w1 == PUB_W1) & (pooled.w3 == PUB_W3)].iloc[0]
print(f"PUBLISHED config (0.9/0.1/0.05/0.95): mean OOF rho={pub_row.oof_spearman_mean:+.4f} "
      f"(rank {int((pooled.oof_spearman_mean < pub_row.oof_spearman_mean).sum())+1} of {len(pooled)})")
within = pooled[pooled.oof_spearman_mean <= best_pooled.oof_spearman_mean + 0.02]
print(f"configurations within 0.02 of the optimum: {len(within)} of {len(pooled)} "
      f"(w3 range {within.w3.min()}-{within.w3.max()}, w1 range {within.w1.min()}-{within.w1.max()})")

print("\nExternal confirmation across 10 partitions (OOF-selected weights):")
print(f"  Spearman rho = {S.ext_spearman_oofsel.mean():+.3f} +/- {S.ext_spearman_oofsel.std(ddof=1):.3f}")
print(f"  Pearson  r   = {S.ext_pearson_oofsel.mean():+.3f} +/- {S.ext_pearson_oofsel.std(ddof=1):.3f}")
print("External with the published a priori weights:")
print(f"  Spearman rho = {S.ext_spearman_pub.mean():+.3f} +/- {S.ext_spearman_pub.std(ddof=1):.3f}")
print(f"  Pearson  r   = {S.ext_pearson_pub.mean():+.3f} +/- {S.ext_pearson_pub.std(ddof=1):.3f}")
print(f"  rank agreement between OOF-selected and published S_AD: "
      f"rho = {S.rank_corr_between_scores.mean():.4f} +/- {S.rank_corr_between_scores.std(ddof=1):.4f}")

json.dump(dict(pooled_best=dict(w1=float(best_pooled.w1), w2=float(best_pooled.w2),
                                w3=float(best_pooled.w3), w4=float(best_pooled.w4),
                                mean_oof_spearman=float(best_pooled.oof_spearman_mean)),
               published=dict(w1=PUB_W1, w2=round(1-PUB_W1,2), w3=PUB_W3, w4=round(1-PUB_W3,2),
                              mean_oof_spearman=float(pub_row.oof_spearman_mean)),
               n_within_0p02=int(len(within)), n_grid=int(len(pooled)),
               ext_spearman_oofsel_mean=float(S.ext_spearman_oofsel.mean()),
               ext_spearman_oofsel_sd=float(S.ext_spearman_oofsel.std(ddof=1)),
               ext_spearman_pub_mean=float(S.ext_spearman_pub.mean()),
               ext_spearman_pub_sd=float(S.ext_spearman_pub.std(ddof=1)),
               ext_pearson_pub_mean=float(S.ext_pearson_pub.mean()),
               ext_pearson_pub_sd=float(S.ext_pearson_pub.std(ddof=1)),
               score_rank_agreement_mean=float(S.rank_corr_between_scores.mean())),
          open(os.path.join(ANA, "R03_summary.json"), "w"), indent=2)
print("\nDONE R03")
