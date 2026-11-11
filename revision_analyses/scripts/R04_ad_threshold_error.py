"""R2 comment 5: how much does the applicability-domain threshold actually reduce error?

Across all 10 external scaffold-split partitions: AD-decile error profile, a sweep of
S_AD thresholds with coverage/error trade-off, bootstrap CIs, and formal tests
(Mann-Whitney on in- vs out-of-domain error, Jonckheere-style trend via Spearman
permutation, and paired comparison of retained-set error against the full set).
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
from R_ad import *

W1, W3 = 0.9, 0.05
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

recs = []
for seed, sd in sorted(seed_dirs().items()):
    dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
    m = fp_mask(sd).astype(bool)
    t = ad_terms(dev["fp"][:, m], dev["desc"], ext["fp"][:, m], ext["desc"])
    pr = pd.read_csv(os.path.join(sd, "predictions", "external_test_predictions.csv"))
    pr = pr[pr.model == REF_MODEL].set_index("id").reindex(ext["ids"])
    s = ad_score(t, W1, W3)
    ll = per_sample_logloss(pr.y_true.values, pr.y_prob.values)
    recs.append(pd.DataFrame(dict(split_seed=seed, id=ext["ids"], ad=s,
                                  logloss=ll, brier=(pr.y_prob.values - pr.y_true.values) ** 2,
                                  correct=(pr.y_pred.values == pr.y_true.values).astype(int),
                                  y_true=pr.y_true.values, y_prob=pr.y_prob.values,
                                  tanimoto=t["tanimoto"], cosine=t["cosine"],
                                  density=t["density"], leverage=t["leverage"])))
D = pd.concat(recs, ignore_index=True)
D.to_csv(os.path.join(ANA, "R04_external_ad_error_pooled.csv"), index=False)
print(f"pooled external predictions with AD scores: n={len(D)} over {D.split_seed.nunique()} partitions")

# ---- per-partition correlations ----
cor = D.groupby("split_seed").apply(
    lambda g: pd.Series(dict(spearman=spearmanr(g.ad, g.logloss)[0],
                             pearson=pearsonr(g.ad, g.logloss)[0],
                             spearman_brier=spearmanr(g.ad, g.brier)[0])), include_groups=False)
cor.to_csv(os.path.join(ANA, "R04_per_partition_correlations.csv"))
print("\nAD vs log-loss association per partition:")
print(f"  Spearman rho = {cor.spearman.mean():+.3f} +/- {cor.spearman.std(ddof=1):.3f} "
      f"(all 10 negative: {bool((cor.spearman<0).all())})")
print(f"  Pearson  r   = {cor.pearson.mean():+.3f} +/- {cor.pearson.std(ddof=1):.3f}")
lo, hi = boot_ci(cor.spearman.values, np.mean, n=20000, seed=3)
print(f"  mean Spearman 95% CI over partitions: [{lo:+.3f}, {hi:+.3f}]")

# ---- decile profile ----
D["decile"] = D.groupby("split_seed").ad.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1)
dec = D.groupby("decile").agg(n=("logloss","size"), ad_mean=("ad","mean"),
                             logloss_mean=("logloss","mean"), brier_mean=("brier","mean"),
                             accuracy=("correct","mean"))
ci = D.groupby("decile").logloss.apply(lambda x: pd.Series(boot_ci(x.values, np.mean, n=5000, seed=11),
                                                           index=["ci_low","ci_high"])).unstack()
dec = dec.join(ci)
dec.to_csv(os.path.join(ANA, "R04_ad_decile_profile.csv"))
print("\nAD-decile error profile (pooled over 10 partitions):")
print(dec.round(4).to_string())
rho_dec, p_dec = spearmanr(dec.index.values, dec.logloss_mean.values)
print(f"  monotone trend across deciles: Spearman rho = {rho_dec:+.3f}, p = {p_dec:.4g}")

# ---- threshold sweep ----
rows = []
for thr in THRESHOLDS:
    sub = []
    for seed, g in D.groupby("split_seed"):
        keep = g.ad >= thr
        if keep.sum() < 5 or (~keep).sum() < 1: 
            sub.append(dict(split_seed=seed, coverage=keep.mean(), ll_in=np.nan, ll_out=np.nan,
                            acc_in=np.nan, brier_in=np.nan, red=np.nan)); continue
        sub.append(dict(split_seed=seed, coverage=keep.mean(),
                        ll_in=g.logloss[keep].mean(), ll_out=g.logloss[~keep].mean(),
                        acc_in=g.correct[keep].mean(), brier_in=g.brier[keep].mean(),
                        red=1 - g.logloss[keep].mean() / g.logloss.mean()))
    s = pd.DataFrame(sub)
    u, pu = mannwhitneyu(D.logloss[D.ad >= thr], D.logloss[D.ad < thr], alternative="less")
    lo_r, hi_r = boot_ci(s.red.dropna().values, np.mean, n=20000, seed=5)
    rows.append(dict(ad_threshold=thr, mean_coverage=s.coverage.mean(),
                     logloss_retained=s.ll_in.mean(), logloss_excluded=s.ll_out.mean(),
                     logloss_all=D.logloss.mean(),
                     relative_reduction=s.red.mean(), red_ci_low=lo_r, red_ci_high=hi_r,
                     accuracy_retained=s.acc_in.mean(), brier_retained=s.brier_in.mean(),
                     p_mannwhitney=pu, n_partitions=int(s.red.notna().sum())))
T = pd.DataFrame(rows)
T["p_mannwhitney_holm"] = holm(T.p_mannwhitney.values)
T.to_csv(os.path.join(ANA, "R04_ad_threshold_sweep.csv"), index=False)
print("\nS_AD threshold sweep (pooled external compounds, 10 partitions):")
print(T[["ad_threshold","mean_coverage","logloss_retained","logloss_excluded","relative_reduction",
         "red_ci_low","red_ci_high","accuracy_retained","p_mannwhitney_holm"]].round(4).to_string(index=False))

# ---- the operating threshold actually used in screening (0.30) ----
thr = 0.30
keep = D.ad >= thr
print(f"\nOperating threshold S_AD >= {thr}: coverage {keep.mean():.3f}, "
      f"log-loss {D.logloss[keep].mean():.4f} vs {D.logloss[~keep].mean():.4f} excluded "
      f"({100*(1-D.logloss[keep].mean()/D.logloss.mean()):.1f}% reduction vs all), "
      f"accuracy {D.correct[keep].mean():.3f} vs {D.correct[~keep].mean():.3f}")

# ---- which AD component carries the signal ----
comp = {}
for c in ["ad","tanimoto","cosine","density"]:
    per = D.groupby("split_seed").apply(lambda g: spearmanr(g[c], g.logloss)[0], include_groups=False)
    comp[c] = (per.mean(), per.std(ddof=1))
print("\nAD component vs log-loss (mean Spearman over partitions):")
for k,(mu,sd_) in comp.items(): print(f"  {k:<10} {mu:+.3f} +/- {sd_:.3f}")

json.dump(dict(n_pooled=int(len(D)),
               spearman_mean=float(cor.spearman.mean()), spearman_sd=float(cor.spearman.std(ddof=1)),
               spearman_ci=[lo,hi], pearson_mean=float(cor.pearson.mean()),
               decile_trend_rho=float(rho_dec), decile_trend_p=float(p_dec),
               decile_logloss_first=float(dec.logloss_mean.iloc[0]),
               decile_logloss_last=float(dec.logloss_mean.iloc[-1]),
               operating=dict(threshold=thr, coverage=float(keep.mean()),
                              logloss_in=float(D.logloss[keep].mean()),
                              logloss_out=float(D.logloss[~keep].mean()),
                              acc_in=float(D.correct[keep].mean()),
                              acc_out=float(D.correct[~keep].mean())),
               components={k: [float(v[0]), float(v[1])] for k,v in comp.items()}),
          open(os.path.join(ANA, "R04_summary.json"), "w"), indent=2)
print("\nDONE R04")
