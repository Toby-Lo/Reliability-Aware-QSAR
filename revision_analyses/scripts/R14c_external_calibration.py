"""Apply the stored calibrators once to the untouched external sets.

Selection between Platt scaling and isotonic regression was made on development
out-of-fold predictions. Here the fitted calibrators are applied to the external
scaffold-split sets so the gain can be reported on genuinely held-out data as well.
"""
import os, sys, json
import numpy as np, pandas as pd
import joblib
from sklearn.metrics import brier_score_loss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == 0 else (p > edges[i]) & (p <= edges[i + 1])
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return out


rows = []
for seed, sd in sorted(seed_dirs().items()):
    dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
    m = fp_mask(sd).astype(bool)
    model, scaler, thr, _ = load_svc(sd, seed)
    Xe = np.hstack([ext["fp"][:, m], scaler.transform(ext["desc"])])
    y = ext["y"].astype(int)
    p_raw = model.predict_proba(Xe)[:, 1]
    rec = dict(split_seed=seed, n=len(y), brier_raw=brier_score_loss(y, p_raw),
               ece_raw=ece(y, p_raw))
    for meth in ("isotonic", "sigmoid"):
        f = os.path.join(sd, "calibration", REF_MODEL, f"method_{meth}", "calibrated_model.joblib")
        if not os.path.exists(f):
            continue
        cal = joblib.load(f)
        q = cal.predict_proba(Xe)[:, 1]
        rec[f"brier_{meth}"] = brier_score_loss(y, q)
        rec[f"ece_{meth}"] = ece(y, q)
    rows.append(rec)
    print(f"seed {seed:>6}: raw Brier {rec['brier_raw']:.4f} | "
          f"isotonic {rec.get('brier_isotonic', float('nan')):.4f} | "
          f"Platt {rec.get('brier_sigmoid', float('nan')):.4f}")

E = pd.DataFrame(rows)
E.to_csv(os.path.join(ANA, "R14c_external_calibration.csv"), index=False)
cols = [c for c in E.columns if c.startswith(("brier", "ece"))]
print("\nExternal scaffold-split sets, mean +/- SD over 10 partitions:")
for c in cols:
    print(f"  {c:<18} {E[c].mean():.4f} +/- {E[c].std(ddof=1):.4f}")
imp = E.brier_raw - E.brier_isotonic
lo, hi = boot_ci(imp.values, np.mean, n=20000, seed=17)
print(f"\nisotonic Brier improvement on external data: {imp.mean():+.4f} "
      f"(95% CI [{lo:+.4f}, {hi:+.4f}]); improved in "
      f"{int((imp > 0).sum())}/{len(imp)} partitions")
impp = E.brier_raw - E.brier_sigmoid
print(f"Platt Brier improvement on external data:    {impp.mean():+.4f}; improved in "
      f"{int((impp > 0).sum())}/{len(impp)} partitions")

json.dump(dict(external={c: [float(E[c].mean()), float(E[c].std(ddof=1))] for c in cols},
               isotonic_improvement=float(imp.mean()), isotonic_ci=[lo, hi],
               isotonic_n_improved=int((imp > 0).sum()),
               platt_improvement=float(impp.mean()),
               platt_n_improved=int((impp > 0).sum())),
          open(os.path.join(ANA, "R14c_summary.json"), "w"), indent=2)
print("\nDONE R14c")
