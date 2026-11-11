"""R1 comment 1, step 2: can the model separate true actives from property-matched
putative inactives drawn from the screening library itself?

The training labels contrast potent against weakly potent compounds from the same
medicinal-chemistry series, so this test asks the different question that matters for
prospective screening: actives versus library compounds matched on size, lipophilicity
and hydrogen-bonding but topologically unrelated. Scored with the per-partition SVC, so
every active is evaluated only by a model that never saw its scaffold.
"""
import os, sys, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
from R_ad import ad_terms, ad_score

ZF = os.path.join(PROJ, "data", "database", "zinc_features.parquet")
AD_GATE = 0.30

dec = pd.read_csv(os.path.join(ANA, "R11_decoy_set.csv"))
cov = pd.read_csv(os.path.join(ANA, "R11_decoy_coverage.csv"))
matched_actives = set(cov[cov.selected > 0].active)
print(f"decoys: {len(dec)}; actives with matched decoys: {len(matched_actives)}")

# ---- pull the published features for the decoys ----
want = set(dec.zinc_id.astype(int))
mcols = [f"morgan_{i}" for i in range(2048)]
pf = pq.ParquetFile(ZF)
parts = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["zinc_id"]).to_pandas()
    hit = t.zinc_id.isin(want).values
    if not hit.any(): continue
    full = pf.read_row_group(rg, columns=["zinc_id"] + mcols + DESC_NAMES).to_pandas()
    parts.append(full[hit])
    if rg % 20 == 0: print(f"  row group {rg+1}/{pf.metadata.num_row_groups}", flush=True)
DF = pd.concat(parts, ignore_index=True).drop_duplicates("zinc_id")
print(f"decoy features retrieved: {len(DF)} of {len(want)}")
dec_fp_full = DF[mcols].values.astype(np.float32)
dec_desc = DF[DESC_NAMES].values.astype(np.float32)
dec_ids = DF.zinc_id.values

# full 2048-bit fp needed only for AD Tanimoto; reuse the masked bits (that is the AD space)
rows, curves = [], []
for seed, sdir in sorted(seed_dirs().items()):
    dev = load_split(sdir, "dev_train"); ext = load_split(sdir, "external_test")
    m = fp_mask(sdir).astype(bool)
    model, scaler, thr, _ = load_svc(sdir, seed)
    dec_fp_masked = dec_fp_full[:, m]

    # actives: external-test actives that have matched decoys
    keep = np.array([(y == 1) and (i in matched_actives) for i, y in zip(ext["ids"], ext["y"])])
    if keep.sum() < 10:
        print(f"seed {seed}: only {keep.sum()} usable actives, skipped"); continue
    a_fp = ext["fp"][:, m][keep]; a_desc = ext["desc"][keep]

    X = np.vstack([np.hstack([a_fp, scaler.transform(a_desc)]),
                   np.hstack([dec_fp_masked, scaler.transform(dec_desc)])])
    y = np.r_[np.ones(keep.sum(), int), np.zeros(len(dec_ids), int)]
    p = model.predict_proba(X)[:, 1]

    r = dict(split_seed=seed, n_actives=int(keep.sum()), n_decoys=len(dec_ids),
             roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
             ef1=ef_at(y, p, 0.01), ef5=ef_at(y, p, 0.05), ef10=ef_at(y, p, 0.10),
             bedroc20=bedroc(y, p, 20.0), bedroc8=bedroc(y, p, 8.0),
             hit_rate_at_thr=float((p[y == 1] >= thr).mean()),
             decoy_flag_rate_at_thr=float((p[y == 0] >= thr).mean()))

    # with the applicability-domain gate applied, as in the screening workflow
    t_a = ad_terms(dev["fp"][:, m], dev["desc"], a_fp, a_desc)
    t_d = ad_terms(dev["fp"][:, m], dev["desc"], dec_fp_masked, dec_desc)
    s_all = np.r_[ad_score(t_a, 0.9, 0.05), ad_score(t_d, 0.9, 0.05)]
    g = s_all >= AD_GATE
    if g.sum() > 20 and y[g].sum() >= 5:
        r.update(ad_gate_coverage=float(g.mean()),
                 ad_gate_active_retention=float(g[y == 1].mean()),
                 ad_gate_decoy_retention=float(g[y == 0].mean()),
                 roc_auc_ad=roc_auc_score(y[g], p[g]),
                 ef1_ad=ef_at(y[g], p[g], 0.01), bedroc20_ad=bedroc(y[g], p[g], 20.0))
    rows.append(r)
    print(f"seed {seed:>6}: actives {r['n_actives']:>3}  ROC-AUC {r['roc_auc']:.3f}  "
          f"EF1% {r['ef1']:.1f}  BEDROC20 {r['bedroc20']:.3f}  "
          f"decoys flagged active {r['decoy_flag_rate_at_thr']:.3f}")

B = pd.DataFrame(rows)
B.to_csv(os.path.join(ANA, "R11_decoy_benchmark_per_seed.csv"), index=False)
cols = ["roc_auc","pr_auc","ef1","ef5","ef10","bedroc20","bedroc8",
        "hit_rate_at_thr","decoy_flag_rate_at_thr","roc_auc_ad","ef1_ad","bedroc20_ad",
        "ad_gate_active_retention","ad_gate_decoy_retention"]
S = B[[c for c in cols if c in B.columns]].agg(["mean","std"]).T
S.columns = ["mean","sd"]
S.to_csv(os.path.join(TAB, "R11_decoy_benchmark_summary.csv"))
print("\nRetrospective discrimination of actives from property-matched library decoys")
print(f"(mean +/- SD over {len(B)} scaffold-disjoint partitions; "
      f"{int(B.n_actives.mean())} actives vs {len(dec_ids)} decoys per partition):")
print(S.round(4).to_string())

json.dump(dict(n_partitions=int(len(B)), n_decoys=int(len(dec_ids)),
               mean_actives=float(B.n_actives.mean()),
               summary={k: [float(v["mean"]), float(v["sd"])] for k, v in S.iterrows()}),
          open(os.path.join(ANA, "R11c_summary.json"), "w"), indent=2)
print("\nDONE R11c")
