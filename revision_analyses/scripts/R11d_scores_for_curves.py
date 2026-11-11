"""Dump per-compound classifier scores for the decoy benchmark, for ROC curves."""
import os, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

ZF = os.path.join(PROJ, "data", "database", "zinc_features.parquet")
dec = pd.read_csv(os.path.join(ANA, "R11_decoy_set.csv"))
cov = pd.read_csv(os.path.join(ANA, "R11_decoy_coverage.csv"))
matched = set(cov[cov.selected > 0].active)
want = set(dec.zinc_id.astype(int))

mcols = [f"morgan_{i}" for i in range(2048)]
pf = pq.ParquetFile(ZF)
parts = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["zinc_id"]).to_pandas()
    hit = t.zinc_id.isin(want).values
    if hit.any():
        parts.append(pf.read_row_group(rg, columns=["zinc_id"] + mcols + DESC_NAMES)
                     .to_pandas()[hit])
DF = pd.concat(parts, ignore_index=True).drop_duplicates("zinc_id")
dec_fp = DF[mcols].values.astype(np.float32)
dec_desc = DF[DESC_NAMES].values.astype(np.float32)
print("decoy features:", dec_fp.shape)

rows = []
for seed, sd in sorted(seed_dirs().items()):
    ext = load_split(sd, "external_test")
    m = fp_mask(sd).astype(bool)
    model, scaler, thr, _ = load_svc(sd, seed)
    keep = np.array([(y == 1) and (i in matched) for i, y in zip(ext["ids"], ext["y"])])
    if keep.sum() < 10:
        continue
    X = np.vstack([np.hstack([ext["fp"][:, m][keep], scaler.transform(ext["desc"][keep])]),
                   np.hstack([dec_fp[:, m], scaler.transform(dec_desc)])])
    y = np.r_[np.ones(keep.sum(), int), np.zeros(len(dec_fp), int)]
    p = model.predict_proba(X)[:, 1]
    rows.append(pd.DataFrame(dict(split_seed=seed, y=y, score=p)))
    print(f"  seed {seed}: {keep.sum()} actives")
S = pd.concat(rows, ignore_index=True)
S.to_csv(os.path.join(ANA, "R11_scores_for_curves.csv"), index=False)
print("saved scores:", S.shape)
