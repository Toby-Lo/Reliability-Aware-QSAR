"""R1 comment 1, step 1: assemble a property-matched putative-inactive (decoy) set.

For every curated active, candidate decoys are drawn from the 7.88 M drug-like
ZINC20-derived library inside tight physicochemical windows (DUD-E style) and are
required to be topologically dissimilar to every active (full 2048-bit Morgan Tanimoto
< 0.35), so the benchmark tests active/inactive discrimination rather than property
discrimination. Decoys may be shared between actives; the final set is deduplicated.
"""
import os, sys, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors as De
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
RDLogger.DisableLog("rdApp.*")

ZF = os.path.join(PROJ, "data", "database", "zinc_features.parquet")
N_DECOY, TC_MAX, POOL = 50, 0.35, 600
WIN = dict(MolWt=25.0, MolLogP=1.0, NumHDonors=1.0, NumHAcceptors=2.0, NumRotatableBonds=2.0)
PROPS = list(WIN.keys())
PFUN = dict(MolWt=De.MolWt, MolLogP=De.MolLogP, NumHDonors=De.NumHDonors,
            NumHAcceptors=De.NumHAcceptors, NumRotatableBonds=De.NumRotatableBonds)

tr = pd.read_csv(os.path.join(PROJ, "data", "nsd2_test", "nsd2_final_dataset.csv"))
act = tr[tr.label == 1].reset_index(drop=True)
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
mols, keep = [], []
for i, s in enumerate(act.smiles):
    m = Chem.MolFromSmiles(s)
    if m is not None: mols.append(m); keep.append(i)
act = act.iloc[keep].reset_index(drop=True)
act_fp = [gen.GetFingerprint(m) for m in mols]
act_props = pd.DataFrame({p: [PFUN[p](m) for m in mols] for p in PROPS})
print(f"curated actives parsed: {len(act)}")
print(act_props.describe().loc[["min", "50%", "max"]].round(2).to_string())

A = act_props[PROPS].values.astype(np.float32)
w = np.array([WIN[p] for p in PROPS], dtype=np.float32)
lo, hi = A - w, A + w

pf = pq.ParquetFile(ZF)
cand = [[] for _ in range(len(act))]
filled = np.zeros(len(act), dtype=int)
rng = np.random.default_rng(20260704)
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["zinc_id", "smiles"] + PROPS).to_pandas()
    V = t[PROPS].values.astype(np.float32)
    for j in range(len(act)):
        if filled[j] >= POOL: continue
        ok = np.all((V >= lo[j]) & (V <= hi[j]), axis=1)
        if not ok.any(): continue
        idx = np.flatnonzero(ok)
        if len(idx) > POOL - filled[j]:
            idx = rng.choice(idx, POOL - filled[j], replace=False)
        cand[j].append(t.iloc[idx][["zinc_id", "smiles"]])
        filled[j] += len(idx)
    if rg % 20 == 0:
        print(f"  row group {rg+1}/{pf.metadata.num_row_groups}, "
              f"full pools {int((filled>=POOL).sum())}/{len(act)}", flush=True)
    if (filled >= POOL).all(): break
print(f"pools built: median size {np.median(filled):.0f}, empty {int((filled==0).sum())}")

sim_cache, rows, coverage = {}, [], []
for j in range(len(act)):
    aid = act.molecule_chembl_id.iloc[j]
    if not cand[j]:
        coverage.append(dict(active=aid, pool=0, selected=0)); continue
    pool = pd.concat(cand[j], ignore_index=True).drop_duplicates("zinc_id")
    sel = 0
    for _, r in pool.iterrows():
        if sel >= N_DECOY: break
        zid = int(r.zinc_id)
        if zid not in sim_cache:
            mm = Chem.MolFromSmiles(r.smiles)
            sim_cache[zid] = 1.0 if mm is None else \
                max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(mm), act_fp))
        if sim_cache[zid] >= TC_MAX: continue
        sel += 1
        rows.append(dict(zinc_id=zid, smiles=r.smiles, matched_active=aid,
                         max_tanimoto_to_actives=sim_cache[zid]))
    coverage.append(dict(active=aid, pool=len(pool), selected=sel))
    if j % 100 == 0: print(f"  active {j+1}/{len(act)}: pool {len(pool)}, {sel} decoys", flush=True)

PAIR = pd.DataFrame(rows)
PAIR.to_csv(os.path.join(ANA, "R11_decoy_active_pairs.csv"), index=False)
D = PAIR.drop_duplicates("zinc_id")[["zinc_id", "smiles", "max_tanimoto_to_actives"]].reset_index(drop=True)
D.to_csv(os.path.join(ANA, "R11_decoy_set.csv"), index=False)
COV = pd.DataFrame(coverage); COV.to_csv(os.path.join(ANA, "R11_decoy_coverage.csv"), index=False)

print(f"\nactives with at least one matched decoy: {int((COV.selected>0).sum())}/{len(act)}")
print(f"actives with the full {N_DECOY}: {int((COV.selected>=N_DECOY).sum())}")
print(f"unique decoys: {len(D)}  (active-decoy pairs: {len(PAIR)})")
print(f"decoy max Tanimoto to any active: mean {D.max_tanimoto_to_actives.mean():.3f}, "
      f"max {D.max_tanimoto_to_actives.max():.3f}")

dp = [[PFUN[p](Chem.MolFromSmiles(s)) for p in PROPS] for s in D.smiles if Chem.MolFromSmiles(s)]
DP = pd.DataFrame(dp, columns=PROPS)
cmp = pd.DataFrame(dict(active_mean=act_props[PROPS].mean(), decoy_mean=DP.mean(),
                        active_sd=act_props[PROPS].std(), decoy_sd=DP.std()))
cmp.to_csv(os.path.join(TAB, "R11_decoy_property_match.csv"))
print("\nproperty match (actives vs decoys):")
print(cmp.round(2).to_string())

json.dump(dict(n_actives=int(len(act)), n_actives_with_decoys=int((COV.selected>0).sum()),
               n_full=int((COV.selected>=N_DECOY).sum()), n_decoys=int(len(D)),
               n_pairs=int(len(PAIR)), tc_max=TC_MAX, windows=WIN,
               decoy_maxT_mean=float(D.max_tanimoto_to_actives.mean())),
          open(os.path.join(ANA, "R11a_summary.json"), "w"), indent=2)
print("DONE R11a")
