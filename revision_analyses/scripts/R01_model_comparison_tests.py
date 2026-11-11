"""R1 comment 3: are the between-model differences in external MCC statistically meaningful?

Paired tests across the 10 common scaffold-disjoint partitions (same splits for every
model, so the partitions are the matched unit), plus McNemar on pooled external
predictions, plus threshold-stability diagnostics.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

res = os.path.join(MODELS, "results")

# ---------- 1. per-partition MCC, external and CV ----------
def read_seed_matrix(path, metric="mcc"):
    df = pd.read_csv(path, index_col=0)
    df = df[~df.index.isin(["MEAN", "STD"])]
    cols = {m: f"{m}_{metric}" for m in MODEL_NAMES}
    out = pd.DataFrame({m: df[c] for m, c in cols.items() if c in df.columns})
    return out

ext = read_seed_matrix(os.path.join(res, "seed_metrics_summary.csv"))
cv  = read_seed_matrix(os.path.join(res, "seed_cv_metrics_summary.csv"))
ext.to_csv(os.path.join(ANA, "R01_external_mcc_per_partition.csv"))
cv.to_csv(os.path.join(ANA, "R01_cv_mcc_per_partition.csv"))
print(f"partitions: external n={len(ext)}, cv n={len(cv)}")

def paired_block(mat, label):
    rows = []
    ref = mat[REF_MODEL].values
    for m in MODEL_NAMES:
        if m == REF_MODEL: continue
        other = mat[m].values
        d = ref - other
        t, pt = stats.ttest_rel(ref, other)
        try:
            w, pw = stats.wilcoxon(ref, other, zero_method="wilcox")
        except Exception:
            w, pw = np.nan, np.nan
        lo, hi = boot_ci(d, np.mean, n=20000, seed=7)
        sd = d.std(ddof=1)
        rows.append(dict(comparison=f"{REF_MODEL} vs {m}", n_partitions=len(d),
                         mean_ref=ref.mean(), mean_other=other.mean(),
                         mean_diff=d.mean(), diff_ci_low=lo, diff_ci_high=hi,
                         cohen_dz=d.mean()/sd if sd > 0 else np.nan,
                         t_stat=t, p_ttest=pt, W=w, p_wilcoxon=pw))
    df = pd.DataFrame(rows)
    df["p_ttest_holm"] = holm(df["p_ttest"].values)
    df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"].values)
    df.insert(0, "evaluation", label)
    return df

tests = pd.concat([paired_block(ext, "External scaffold-split"),
                   paired_block(cv, "Nested scaffold-grouped CV")], ignore_index=True)
tests.to_csv(os.path.join(ANA, "R01_paired_model_tests.csv"), index=False)
print(tests[["evaluation","comparison","mean_diff","diff_ci_low","diff_ci_high","p_ttest_holm","p_wilcoxon_holm"]].to_string(index=False))

# ---------- 2. McNemar on pooled external predictions ----------
pred = pd.read_csv(os.path.join(res, "all_seed_external_predictions.csv"))
print("\npred shape", pred.shape, "models", sorted(pred.model.unique()), "seeds", sorted(pred.seed.unique()))
mc_rows = []
ref_p = pred[pred.model == REF_MODEL].set_index(["seed", "id"])
for m in MODEL_NAMES:
    if m == REF_MODEL: continue
    oth = pred[pred.model == m].set_index(["seed", "id"])
    common = ref_p.index.intersection(oth.index)
    a = ref_p.loc[common]; b = oth.loc[common]
    assert (a.y_true.values == b.y_true.values).all()
    ok_a = (a.y_pred.values == a.y_true.values)
    ok_b = (b.y_pred.values == b.y_true.values)
    n01 = int(np.sum(~ok_a & ok_b))   # ref wrong, other right
    n10 = int(np.sum(ok_a & ~ok_b))   # ref right, other wrong
    nsum = n01 + n10
    # exact binomial McNemar
    p = stats.binomtest(n10, nsum, 0.5).pvalue if nsum > 0 else 1.0
    mc_rows.append(dict(comparison=f"{REF_MODEL} vs {m}", n_paired=len(common),
                        acc_ref=ok_a.mean(), acc_other=ok_b.mean(),
                        ref_only_correct=n10, other_only_correct=n01,
                        p_mcnemar_exact=p))
mc = pd.DataFrame(mc_rows)
mc["p_mcnemar_holm"] = holm(mc["p_mcnemar_exact"].values)
mc.to_csv(os.path.join(ANA, "R01_mcnemar_pooled.csv"), index=False)
print("\n", mc.to_string(index=False))

# ---------- 3. threshold stability and sensitivity ----------
thr_rows = []
for seed, sd in seed_dirs().items():
    p = os.path.join(sd, "results", "threshold_selection_summary.csv")
    if not os.path.exists(p): continue
    t = pd.read_csv(p)
    t["split_seed"] = seed
    thr_rows.append(t)
thr = pd.concat(thr_rows, ignore_index=True)
thr_sum = (thr.groupby("model")
             .agg(n=("selected_threshold","size"),
                  thr_mean=("selected_threshold","mean"),
                  thr_sd=("selected_threshold","std"),
                  thr_min=("selected_threshold","min"),
                  thr_max=("selected_threshold","max"),
                  oof_mcc_mean=("selected_mcc","mean"))
             .reindex(MODEL_NAMES))
thr_sum.to_csv(os.path.join(ANA, "R01_threshold_stability.csv"))
print("\nthreshold stability\n", thr_sum.to_string())

# MCC sensitivity: how flat is the external MCC vs threshold curve for SVC?
curves = []
for seed, sd in seed_dirs().items():
    p = os.path.join(sd, "results", "threshold_curves_data.csv")
    if not os.path.exists(p): continue
    c = pd.read_csv(p)
    curves.append(c)
cur = pd.concat(curves, ignore_index=True)
cur.to_csv(os.path.join(ANA, "R01_threshold_curves_all.csv"), index=False)
sens_rows = []
for (seed, ds), g in cur[cur.model == REF_MODEL].groupby(["split_seed", "dataset"]):
    g = g.sort_values("Threshold")
    best = g.MCC.max(); thr_star = g.loc[g.MCC.idxmax(), "Threshold"]
    within = g[g.MCC >= best - 0.02]
    sens_rows.append(dict(split_seed=seed, dataset=ds, best_mcc=best, best_threshold=thr_star,
                          plateau_low=within.Threshold.min(), plateau_high=within.Threshold.max(),
                          plateau_width=within.Threshold.max() - within.Threshold.min()))
sens = pd.DataFrame(sens_rows)
sens.to_csv(os.path.join(ANA, "R01_svc_threshold_plateau.csv"), index=False)
print("\nSVC threshold plateau (MCC within 0.02 of max)\n",
      sens.groupby("dataset")[["plateau_low","plateau_high","plateau_width","best_threshold"]].mean().to_string())

json.dump(dict(n_partitions=int(len(ext)),
               external_mcc_mean={m: float(ext[m].mean()) for m in ext.columns},
               external_mcc_sd={m: float(ext[m].std(ddof=1)) for m in ext.columns},
               cv_mcc_mean={m: float(cv[m].mean()) for m in cv.columns},
               cv_mcc_sd={m: float(cv[m].std(ddof=1)) for m in cv.columns}),
          open(os.path.join(ANA, "R01_summary.json"), "w"), indent=2)
print("\nDONE R01")
