"""Summarize the retrospective docking enrichment from whatever has been docked.

Decoys are docked in random order after the actives, so a partially completed run is a
uniform random sample of the matched decoy pool and the enrichment metrics remain valid.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import ROOT, ANA, TAB, ef_at, bedroc

OUTD = os.path.join(ROOT, "Revision_R1", "docking_enrichment")
full = os.path.join(OUTD, "docking_scores.csv")
part = os.path.join(OUTD, "docking_scores_partial.csv")
src = full if os.path.exists(full) else part
if not os.path.exists(src):
    sys.exit("no docking scores yet")
print("source:", os.path.basename(src))

D = pd.read_csv(src).dropna(subset=["vina_score"])
ref = D[D.name == "reference_A1A0M"]
body = D[D.name != "reference_A1A0M"]
y = body.label.values.astype(int)
s = -body.vina_score.values            # a more negative Vina score is better
n_a, n_d = int(y.sum()), int((y == 0).sum())
print(f"scored: {len(D)} ligands  ({n_a} actives, {n_d} decoys)")
if n_a < 20 or n_d < 100:
    sys.exit(f"not enough yet for a stable estimate (need >=20 actives and >=100 decoys)")

out = dict(n_actives=n_a, n_decoys=n_d, complete=os.path.exists(full),
           roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
           ef1=float(ef_at(y, s, 0.01)), ef2=float(ef_at(y, s, 0.02)),
           ef5=float(ef_at(y, s, 0.05)), ef10=float(ef_at(y, s, 0.10)),
           bedroc20=float(bedroc(y, s, 20.0)), bedroc8=float(bedroc(y, s, 8.0)),
           active_score_mean=float(body.vina_score[y == 1].mean()),
           active_score_sd=float(body.vina_score[y == 1].std()),
           decoy_score_mean=float(body.vina_score[y == 0].mean()),
           decoy_score_sd=float(body.vina_score[y == 0].std()),
           reference_score=float(ref.vina_score.iloc[0]) if len(ref) else None)
out["p_mannwhitney"] = float(mannwhitneyu(body.vina_score[y == 1],
                                          body.vina_score[y == 0], alternative="less")[1])

rng = np.random.default_rng(3)
bs = {k: [] for k in ("roc_auc", "ef1", "bedroc20")}
for _ in range(2000):
    idx = rng.choice(len(y), len(y), replace=True)
    if y[idx].sum() < 5:
        continue
    bs["roc_auc"].append(roc_auc_score(y[idx], s[idx]))
    bs["ef1"].append(ef_at(y[idx], s[idx], 0.01))
    bs["bedroc20"].append(bedroc(y[idx], s[idx], 20.0))
for k, v in bs.items():
    out[k + "_ci"] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

pd.DataFrame([out]).to_csv(os.path.join(TAB, "R13_docking_enrichment.csv"), index=False)
json.dump(out, open(os.path.join(ANA, "R13_summary.json"), "w"), indent=2)
print("\nRetrospective docking enrichment against 9CVD:")
for k in ("n_actives", "n_decoys", "complete", "roc_auc", "roc_auc_ci", "ef1", "ef1_ci",
          "ef2", "ef5", "ef10", "bedroc20", "bedroc20_ci", "active_score_mean",
          "decoy_score_mean", "reference_score", "p_mannwhitney"):
    print(f"  {k:<18} {out[k]}")
print("\nDONE R13c")
