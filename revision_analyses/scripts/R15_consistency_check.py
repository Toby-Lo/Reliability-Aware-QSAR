"""Reconcile two conventions so the manuscript and supplement agree.

1. Bemis-Murcko scaffold counts with and without stereochemistry, to match the 185 scaffolds
   reported for the training set in the original manuscript.
2. Maximum Tanimoto of the prioritized sets to the FULL 530-compound training set, in both
   the 129-bit model space and the conventional 2048-bit space.
"""
import os, sys, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
from R_ad import tanimoto_max, binarize
RDLogger.DisableLog("rdApp.*")

tr = pd.read_csv(os.path.join(PROJ, "data", "nsd2_test", "nsd2_final_dataset.csv"))
mols = [Chem.MolFromSmiles(s) for s in tr.smiles]
mols = [m for m in mols if m is not None]
print(f"training molecules parsed: {len(mols)}")

for chir in (True, False):
    sc = set()
    for m in mols:
        try:
            sc.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=chir))
        except Exception:
            pass
    print(f"  Bemis-Murcko scaffolds, includeChirality={chir}: {len(sc)}")

# the convention that reproduces the published 185
CHIR = False


def scaffolds(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=CHIR))
        except Exception:
            pass
    return out


def frameworks(smis):
    """Generic (graph) frameworks: Murcko scaffold with all atoms and bonds made uniform."""
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m)
            out.append(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc)))
        except Exception:
            pass
    return out


tr_sc = scaffolds(tr.smiles)
singles = pd.Series(tr_sc).value_counts()
print(f"\ntraining: {len(set(tr_sc))} scaffolds, "
      f"{int((singles==1).sum())} singletons "
      f"({100*(singles==1).sum()/len(set(tr_sc)):.1f}% of scaffolds), "
      f"ratio {len(set(tr_sc))/len(tr_sc):.3f}")

TOP1000 = os.path.join(MODELS, "virtual_screening", "ad_screening_results_20260419_140044",
                       "top1000_full_info.csv")
TOP100 = os.path.join(PROJ, "docking", "Final_candidates",
                      "qsar_admet_docking_summary_TOP100_9CVD.csv")
TOP8 = os.path.join(PROJ, "docking", "Final_candidates", "final_cadicates_for_mds.csv")

gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
tr_fp2048 = [gen.GetFingerprint(m) for m in mols]

# full-training-set fingerprints in the 129-bit model space
sd = seed_dirs()[REP_SEED]
m129 = fp_mask(sd).astype(bool)


def fp_matrix(smis):
    rows, keep = [], []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        a = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), a)
        rows.append(a); keep.append(s)
    return np.array(rows), keep


TR_X, _ = fp_matrix(tr.smiles)
print(f"\ntraining fingerprint matrix: {TR_X.shape}")

rows = []
for name, path, col in [("Top 1000", TOP1000, "smiles"), ("Top 100", TOP100, "smiles"),
                        ("Top 8", TOP8, "smiles")]:
    df = pd.read_csv(path)
    X, kept = fp_matrix(df[col].tolist())
    t2048 = tanimoto_max(binarize(TR_X), binarize(X))
    t129 = tanimoto_max(binarize(TR_X[:, m129]), binarize(X[:, m129]))
    sc = scaffolds(df[col].tolist())
    fw = frameworks(df[col].tolist())
    shared = len(set(sc) & set(tr_sc))
    pop = pd.Series(sc).value_counts()
    rows.append(dict(set=name, n=len(kept), scaffolds=len(set(sc)),
                     ratio=len(set(sc)) / len(sc), frameworks=len(set(fw)),
                     shared_with_training=shared,
                     largest_family=int(pop.iloc[0]), singletons=int((pop == 1).sum()),
                     t2048_mean=t2048.mean(), t2048_min=t2048.min(), t2048_max=t2048.max(),
                     t129_mean=t129.mean(), t129_min=t129.min(), t129_max=t129.max()))
    print(f"\n{name}: {len(set(sc))} scaffolds (ratio {len(set(sc))/len(sc):.3f}), "
          f"{len(set(fw))} generic frameworks, "
          f"largest family {pop.iloc[0]}, singletons {int((pop==1).sum())}, "
          f"shared with training {shared}")
    print(f"   max Tanimoto vs all {len(mols)} training compounds: "
          f"2048-bit {t2048.mean():.3f} ({t2048.min():.3f}-{t2048.max():.3f}) | "
          f"129-bit {t129.mean():.3f} ({t129.min():.3f}-{t129.max():.3f})")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(TAB, "R15_scaffold_similarity_consistent.csv"), index=False)
json.dump(dict(training_scaffolds=len(set(tr_sc)),
               training_singletons=int((singles == 1).sum()),
               include_chirality=CHIR,
               sets=R.round(4).to_dict(orient="records")),
          open(os.path.join(ANA, "R15_summary.json"), "w"), indent=2)
print("\nDONE R15")
