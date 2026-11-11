"""R1 comment 4: are the top SHAP features just proxies for molecular size?

Three descriptor configurations are retrained with the published SVC hyperparameters on
all 10 scaffold-disjoint partitions: the full 20-descriptor panel, the panel with every
size/bulk proxy removed, and fingerprints only. Collinearity among the size descriptors
is quantified, and SHAP is recomputed for the size-free model.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

CONFIGS = {
    "Full descriptor panel (published)": None,
    "Size proxies removed":             SIZE_DESC,
    "Fingerprints only":               DESC_NAMES,
}

def metrics(y, p, thr):
    yp = (p >= thr).astype(int)
    return dict(mcc=matthews_corrcoef(y, yp), f1=f1_score(y, yp, zero_division=0),
                accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp, zero_division=0),
                roc_auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                pr_auc=average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan)

rows = []
for seed, sd in sorted(seed_dirs().items()):
    dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
    m = fp_mask(sd).astype(bool)
    fp_d, fp_e = dev["fp"][:, m], ext["fp"][:, m]
    _, _, thr, cfg = load_svc(sd, seed)
    bp = cfg["best_hyperparameters"]["best_params"]
    for label, drop in CONFIGS.items():
        keep = [i for i, n in enumerate(DESC_NAMES) if drop is None or n not in drop]
        sc = StandardScaler().fit(dev["desc"][:, keep]) if keep else None
        Xd = np.hstack([fp_d, sc.transform(dev["desc"][:, keep])]) if keep else fp_d
        Xe = np.hstack([fp_e, sc.transform(ext["desc"][:, keep])]) if keep else fp_e
        clf = SVC(C=bp.get("C", 1.0), gamma=bp.get("gamma", "scale"),
                  kernel=bp.get("kernel", "rbf"), probability=True, random_state=seed)
        clf.fit(Xd, dev["y"])
        p = clf.predict_proba(Xe)[:, 1]
        r = metrics(ext["y"], p, thr)
        r.update(split_seed=seed, config=label, n_descriptors=len(keep), n_features=Xd.shape[1])
        rows.append(r)
    print(f"seed {seed:>6} done")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(ANA, "R05_ablation_per_partition.csv"), index=False)
summ = R.groupby("config").agg(n_descriptors=("n_descriptors","first"), n_features=("n_features","first"),
                               mcc_mean=("mcc","mean"), mcc_sd=("mcc","std"),
                               f1_mean=("f1","mean"), f1_sd=("f1","std"),
                               acc_mean=("accuracy","mean"), acc_sd=("accuracy","std"),
                               roc_mean=("roc_auc","mean"), roc_sd=("roc_auc","std"),
                               pr_mean=("pr_auc","mean"), pr_sd=("pr_auc","std")).reindex(CONFIGS.keys())
summ.to_csv(os.path.join(TAB, "R05_ablation_summary.csv"))
print("\nExternal performance by descriptor configuration (mean +/- SD over 10 partitions):")
print(summ.round(3).to_string())

base = "Full descriptor panel (published)"
piv = R.pivot(index="split_seed", columns="config", values="mcc")
tests = []
for label in CONFIGS:
    if label == base: continue
    d = piv[base].values - piv[label].values
    t, pt = ttest_rel(piv[base].values, piv[label].values)
    try: w, pw = wilcoxon(piv[base].values, piv[label].values)
    except Exception: w, pw = np.nan, np.nan
    lo, hi = boot_ci(d, np.mean, n=20000, seed=13)
    tests.append(dict(comparison=f"{base} vs {label}", mean_mcc_drop=d.mean(),
                      ci_low=lo, ci_high=hi, p_ttest=pt, p_wilcoxon=pw))
TT = pd.DataFrame(tests)
TT.to_csv(os.path.join(ANA, "R05_ablation_tests.csv"), index=False)
print("\nPaired comparison of external MCC against the published panel:")
print(TT.round(4).to_string(index=False))

# ---- collinearity among the size descriptors ----
sd0 = seed_dirs()[REP_SEED]
dev0 = load_split(sd0, "dev_train")
dfd = pd.DataFrame(dev0["desc"], columns=DESC_NAMES)
cm = dfd[SIZE_DESC].corr(method="pearson")
cm.to_csv(os.path.join(ANA, "R05_size_descriptor_correlation.csv"))
print("\nPearson correlation among size/bulk descriptors (development set, seed 12345):")
print(cm.round(3).to_string())
off = cm.values[np.triu_indices_from(cm.values, 1)]
print(f"  off-diagonal |r|: median {np.median(np.abs(off)):.3f}, max {np.abs(off).max():.3f}, "
      f"fraction >0.9: {np.mean(np.abs(off) > 0.9):.2f}")

# correlation of each size descriptor with the activity label
lab = pd.DataFrame({c: [np.corrcoef(dfd[c], dev0["y"])[0,1]] for c in DESC_NAMES}).T
lab.columns = ["pearson_r_with_label"]
lab.sort_values("pearson_r_with_label", key=np.abs, ascending=False).to_csv(
    os.path.join(ANA, "R05_descriptor_label_correlation.csv"))
print("\nTop descriptor-label correlations:")
print(lab.reindex(lab.pearson_r_with_label.abs().sort_values(ascending=False).index).head(8).round(3).to_string())

json.dump(dict(summary=summ.round(4).to_dict(orient="index"),
               tests=TT.round(5).to_dict(orient="records"),
               size_desc=SIZE_DESC,
               size_collinearity_median_abs_r=float(np.median(np.abs(off))),
               size_collinearity_max_abs_r=float(np.abs(off).max())),
          open(os.path.join(ANA, "R05_summary.json"), "w"), indent=2)
print("\nDONE R05")
