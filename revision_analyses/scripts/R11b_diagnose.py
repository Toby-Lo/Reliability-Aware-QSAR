import os, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors as De
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
RDLogger.DisableLog("rdApp.*")

tr = pd.read_csv(os.path.join(PROJ, "data", "nsd2_test", "nsd2_final_dataset.csv"))
act = tr[tr.label == 1].reset_index(drop=True)
cov = pd.read_csv(os.path.join(ANA, "R11_decoy_coverage.csv"))
P = []
for s in act.smiles:
    m = Chem.MolFromSmiles(s)
    P.append([De.MolWt(m), De.MolLogP(m), De.NumHDonors(m), De.NumHAcceptors(m),
              De.NumRotatableBonds(m), De.qed(m)] if m else [np.nan]*6)
A = pd.DataFrame(P, columns=["MolWt","MolLogP","NumHDonors","NumHAcceptors","NumRotatableBonds","QED"])
A["molecule_chembl_id"] = act.molecule_chembl_id.values
A = A.merge(cov, left_on="molecule_chembl_id", right_on="active", how="left")
A["matched"] = A.selected.fillna(0) > 0
print("actives matched vs unmatched (mean properties):")
print(A.groupby("matched")[["MolWt","MolLogP","NumHDonors","NumHAcceptors","NumRotatableBonds","QED"]].mean().round(2).to_string())
print("\nlibrary drug-like window: MW<=500, LogP<=5, HBD<=5, HBA<=10, RotB<=10, QED>=0.4")
for lab, sub in A.groupby("matched"):
    viol = ((sub.MolWt > 500) | (sub.MolLogP > 5) | (sub.NumHDonors > 5) |
            (sub.NumHAcceptors > 10) | (sub.NumRotatableBonds > 10) | (sub.QED < 0.4))
    print(f"  matched={lab}: n={len(sub)}, violating the library window: {int(viol.sum())} ({100*viol.mean():.0f}%)")
print(f"\nactives with MW > 500: {int((A.MolWt>500).sum())}/{len(A)} ({100*(A.MolWt>500).mean():.0f}%)")
print(f"actives with QED < 0.4: {int((A.QED<0.4).sum())}/{len(A)} ({100*(A.QED<0.4).mean():.0f}%)")
A.to_csv(os.path.join(ANA, "R11_active_matchability.csv"), index=False)
