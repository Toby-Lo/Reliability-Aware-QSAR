"""R2 comment 4: is a 530-compound training set informative enough to rank an ultra-large library?

Scaffold-aware learning curves: whole Bemis-Murcko scaffolds are removed from the
development set to reach each target size, the published SVC is retrained, and external
scaffold-split performance is measured. A plateau indicates the available data are near
their information limit for this representation; a rising curve quantifies how much is
still to be gained from more data.
"""
import os, sys, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
RDLogger.DisableLog("rdApp.*")

FRACS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 5

def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return smi
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return smi

rows = []
for seed, sd in sorted(seed_dirs().items()):
    dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
    m = fp_mask(sd).astype(bool)
    fp_d, fp_e = dev["fp"][:, m], ext["fp"][:, m]
    _, _, thr, cfg = load_svc(sd, seed)
    bp = cfg["best_hyperparameters"]["best_params"]
    scaf = np.array([scaffold_of(s) for s in dev["smiles"]])
    groups = pd.Series(scaf).groupby(scaf).groups
    gkeys = list(groups.keys())
    for frac in FRACS:
        target = int(round(frac * len(dev["y"])))
        for rep in range(1 if frac == 1.0 else N_REPEATS):
            rng = np.random.default_rng(1000 * rep + seed)
            order = rng.permutation(len(gkeys))
            idx = []
            for gi in order:
                idx.extend(list(groups[gkeys[gi]]))
                if len(idx) >= target: break
            idx = np.array(sorted(idx))
            y = dev["y"][idx]
            if len(np.unique(y)) < 2: continue
            sc = StandardScaler().fit(dev["desc"][idx])
            Xd = np.hstack([fp_d[idx], sc.transform(dev["desc"][idx])])
            Xe = np.hstack([fp_e, sc.transform(ext["desc"])])
            clf = SVC(C=bp.get("C",1.0), gamma=bp.get("gamma","scale"),
                      kernel=bp.get("kernel","rbf"), probability=True, random_state=seed)
            clf.fit(Xd, y)
            p = clf.predict_proba(Xe)[:, 1]
            rows.append(dict(split_seed=seed, frac=frac, rep=rep, n_train=len(idx),
                             n_scaffolds=len(set(scaf[idx])),
                             mcc=matthews_corrcoef(ext["y"], (p >= thr).astype(int)),
                             f1=f1_score(ext["y"], (p >= thr).astype(int), zero_division=0),
                             roc_auc=roc_auc_score(ext["y"], p),
                             pr_auc=average_precision_score(ext["y"], p)))
    print(f"seed {seed:>6} done")

L = pd.DataFrame(rows)
L.to_csv(os.path.join(ANA, "R08_learning_curve_raw.csv"), index=False)
agg = L.groupby("frac").agg(n_train_mean=("n_train","mean"), n_scaffolds_mean=("n_scaffolds","mean"),
                            mcc_mean=("mcc","mean"), mcc_sd=("mcc","std"),
                            roc_mean=("roc_auc","mean"), roc_sd=("roc_auc","std"),
                            pr_mean=("pr_auc","mean"), pr_sd=("pr_auc","std"), n=("mcc","size"))
agg.to_csv(os.path.join(TAB, "R08_learning_curve_summary.csv"))
print("\nScaffold-aware learning curve (external scaffold-split performance):")
print(agg.round(3).to_string())

full = agg.loc[1.0]
for f in [0.5, 0.6, 0.7, 0.8, 0.9]:
    r = agg.loc[f]
    print(f"  at {int(r.n_train_mean)} training compounds ({f:.0%}): MCC {r.mcc_mean:.3f} "
          f"= {100*r.mcc_mean/full.mcc_mean:.1f}% of the full-data MCC")
gain = full.mcc_mean - agg.loc[0.8].mcc_mean
print(f"\n  MCC gain over the final 20% of the data: {gain:+.4f}")
print(f"  MCC gain from 20% to 100%: {full.mcc_mean - agg.loc[0.2].mcc_mean:+.4f}")

json.dump(dict(curve=agg.round(5).to_dict(orient="index"),
               full_mcc=float(full.mcc_mean),
               gain_last_20pct=float(gain),
               gain_20_to_100=float(full.mcc_mean - agg.loc[0.2].mcc_mean)),
          open(os.path.join(ANA, "R08_summary.json"), "w"), indent=2)
print("\nDONE R08")
