"""Clarify the two fingerprint spaces in which Tanimoto similarity is reported.

The AD gate (Tc >= 0.85) operates on the 129-bit variance/correlation-filtered
fingerprint actually used by the model; conventional chemical-similarity reporting uses
the full 2048-bit Morgan fingerprint. The two are not interchangeable.
"""
import os, sys, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
from R_ad import tanimoto_max, binarize
RDLogger.DisableLog("rdApp.*")

sd = seed_dirs()[REP_SEED]
dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
m = fp_mask(sd).astype(bool)
print(f"fingerprint mask retains {m.sum()} of {len(m)} bits")

# external test set, both spaces
tc129 = tanimoto_max(binarize(dev["fp"][:, m]), binarize(ext["fp"][:, m]))
tc2048 = tanimoto_max(binarize(dev["fp"]), binarize(ext["fp"]))
print("\nExternal test set (n=111):")
print(f"  129-bit filtered space : mean {tc129.mean():.3f}, median {np.median(tc129):.3f}, "
      f"fraction >= 0.85 = {(tc129>=0.85).mean():.3f}")
print(f"  full 2048-bit space    : mean {tc2048.mean():.3f}, median {np.median(tc2048):.3f}, "
      f"fraction >= 0.85 = {(tc2048>=0.85).mean():.3f}")
print(f"  rank correlation between the two: {pd.Series(tc129).corr(pd.Series(tc2048), method='spearman'):.3f}")

# screening candidates: recompute both spaces from SMILES
gen2048 = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
def fp_matrix(smis):
    rows, keep = [], []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        a = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(gen2048.GetFingerprint(mol), a)
        rows.append(a); keep.append(s)
    return np.array(rows), keep

t1000 = pd.read_csv(os.path.join(MODELS, "virtual_screening",
        "ad_screening_results_20260419_140044", "top1000_full_info.csv"))
X, kept = fp_matrix(t1000.smiles.tolist())
print(f"\nTop-1000 candidates ({len(kept)} parsed):")
a129 = tanimoto_max(binarize(dev["fp"][:, m]), binarize(X[:, m]))
a2048 = tanimoto_max(binarize(dev["fp"]), binarize(X))
print(f"  129-bit filtered space : mean {a129.mean():.3f}, median {np.median(a129):.3f}, "
      f"min {a129.min():.3f}, fraction >= 0.85 = {(a129>=0.85).mean():.3f}")
print(f"  full 2048-bit space    : mean {a2048.mean():.3f}, median {np.median(a2048):.3f}, "
      f"max {a2048.max():.3f}, fraction >= 0.85 = {(a2048>=0.85).mean():.3f}")
print(f"\n  AD_Score column recorded in the screening output: "
      f"mean {t1000.AD_Score.mean():.3f}, min {t1000.AD_Score.min():.3f}, max {t1000.AD_Score.max():.3f}")
print(f"  ad_max_tanimoto column recorded: mean {t1000.ad_max_tanimoto.mean():.3f}, "
      f"min {t1000.ad_max_tanimoto.min():.3f}, max {t1000.ad_max_tanimoto.max():.3f}")
print(f"  reproduced 129-bit Tc vs recorded ad_max_tanimoto: "
      f"Pearson r = {np.corrcoef(a129, t1000.ad_max_tanimoto.values[:len(a129)])[0,1]:.4f}")

out = pd.DataFrame(dict(smiles=kept, tanimoto_129bit=a129, tanimoto_2048bit=a2048))
out.to_csv(os.path.join(ANA, "R07_top1000_tanimoto_two_spaces.csv"), index=False)
json.dump(dict(mask_bits=int(m.sum()),
               external=dict(t129_mean=float(tc129.mean()), t2048_mean=float(tc2048.mean()),
                             t129_frac_ge085=float((tc129>=0.85).mean()),
                             t2048_frac_ge085=float((tc2048>=0.85).mean())),
               top1000=dict(t129_mean=float(a129.mean()), t129_min=float(a129.min()),
                            t129_frac_ge085=float((a129>=0.85).mean()),
                            t2048_mean=float(a2048.mean()), t2048_max=float(a2048.max()),
                            recorded_ad_tanimoto_mean=float(t1000.ad_max_tanimoto.mean()),
                            recorded_ad_tanimoto_min=float(t1000.ad_max_tanimoto.min()))),
          open(os.path.join(ANA, "R07_summary.json"), "w"), indent=2)
print("\nDONE R07")
