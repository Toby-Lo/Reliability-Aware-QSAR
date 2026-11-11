"""R1 comment 4, interpretability half: SHAP attribution after removing size proxies."""
import os, sys, json
import numpy as np, pandas as pd
import shap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

sd = seed_dirs()[REP_SEED]
dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
m = fp_mask(sd).astype(bool)
fp_idx = np.flatnonzero(m)
_, _, thr, cfg = load_svc(sd, REP_SEED)
bp = cfg["best_hyperparameters"]["best_params"]

for tag, drop in [("full", []), ("size_free", SIZE_DESC)]:
    keep = [i for i, n in enumerate(DESC_NAMES) if n not in drop]
    names = [f"fp_{i}" for i in fp_idx] + [DESC_NAMES[i] for i in keep]
    sc = StandardScaler().fit(dev["desc"][:, keep])
    Xd = np.hstack([dev["fp"][:, m], sc.transform(dev["desc"][:, keep])])
    Xe = np.hstack([ext["fp"][:, m], sc.transform(ext["desc"][:, keep])])
    clf = SVC(C=bp.get("C",1.0), gamma=bp.get("gamma","scale"),
              kernel=bp.get("kernel","rbf"), probability=True, random_state=REP_SEED).fit(Xd, dev["y"])
    bg = shap.kmeans(Xd, 25)
    expl = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:, 1], bg)
    sv = expl.shap_values(Xe, nsamples=200, silent=True)
    imp = pd.DataFrame(dict(feature=names, mean_abs_shap=np.abs(sv).mean(0),
                            mean_shap=sv.mean(0)))
    imp["kind"] = ["fingerprint" if f.startswith("fp_") else "descriptor" for f in imp.feature]
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(ANA, f"R05b_shap_{tag}.csv"), index=False)
    tot = imp.mean_abs_shap.sum()
    print(f"\n=== {tag} model ({Xd.shape[1]} features) ===")
    print(imp.head(15).round(5).to_string(index=False))
    print(f"  descriptor share of total |SHAP|: "
          f"{100*imp[imp.kind=='descriptor'].mean_abs_shap.sum()/tot:.1f}%")
    print(f"  fingerprint share:                "
          f"{100*imp[imp.kind=='fingerprint'].mean_abs_shap.sum()/tot:.1f}%")
    if tag == "full":
        s = imp[imp.feature.isin(SIZE_DESC)].mean_abs_shap.sum()
        print(f"  size-proxy share of total |SHAP|: {100*s/tot:.1f}%")
print("\nDONE R05b")
