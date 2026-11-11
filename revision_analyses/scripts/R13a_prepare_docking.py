"""R1 comment 5, step 1: build the retrospective docking enrichment set.

Actives (property-matched subset) and property-matched library decoys are embedded in 3D
and written as PDBQT for AutoDock Vina against the same rigid 9CVD receptor, grid centre
and box used in the manuscript.
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
RDLogger.DisableLog("rdApp.*")

OUT = os.path.join(OUT if False else os.path.join(ROOT, "Revision_R1"), "docking_enrichment")
LIG = os.path.join(OUT, "ligands"); os.makedirs(LIG, exist_ok=True)
OBABEL = "/home/wiky/miniforge3/envs/vsdock/bin/obabel"
N_ACT, N_DEC = 100, 3000
rng = np.random.default_rng(20260704)

cov = pd.read_csv(os.path.join(ANA, "R11_decoy_coverage.csv"))
matched = set(cov[cov.selected > 0].active)
tr = pd.read_csv(os.path.join(PROJ, "data", "nsd2_test", "nsd2_final_dataset.csv"))
act = tr[(tr.label == 1) & (tr.molecule_chembl_id.isin(matched))].reset_index(drop=True)
act = act.iloc[rng.choice(len(act), min(N_ACT, len(act)), replace=False)].reset_index(drop=True)
dec = pd.read_csv(os.path.join(ANA, "R11_decoy_set.csv"))
dec = dec.iloc[rng.choice(len(dec), min(N_DEC, len(dec)), replace=False)].reset_index(drop=True)
print(f"actives selected: {len(act)}   decoys selected: {len(dec)}")

rows = []
for lab, df, idc, smic in [("active", act, "molecule_chembl_id", "smiles"),
                           ("decoy", dec, "zinc_id", "smiles")]:
    for _, r in df.iterrows():
        rows.append(dict(name=f"{lab}_{r[idc]}", label=1 if lab == "active" else 0,
                         smiles=r[smic]))
# the co-crystallised reference, as a positive control
ref = Chem.MolFromMolFile(os.path.join(PROJ, "docking", "9CVD_data_preparation", "A1A0M.sdf"))
if ref is not None:
    rows.append(dict(name="reference_A1A0M", label=1, smiles=Chem.MolToSmiles(ref)))
L = pd.DataFrame(rows)
L.to_csv(os.path.join(OUT, "ligand_manifest.csv"), index=False)
print(f"total ligands to prepare: {len(L)}")

ok, fail = 0, 0
for _, r in L.iterrows():
    out = os.path.join(LIG, r["name"] + ".pdbqt")
    if os.path.exists(out) and os.path.getsize(out) > 0: ok += 1; continue
    m = Chem.MolFromSmiles(r.smiles)
    if m is None: fail += 1; continue
    m = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = 20260704
    if AllChem.EmbedMolecule(m, ps) != 0:
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(m, ps) != 0: fail += 1; continue
    try: AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception: pass
    sdf = os.path.join(LIG, r["name"] + ".sdf")
    w = Chem.SDWriter(sdf); w.write(m); w.close()
    p = subprocess.run([OBABEL, sdf, "-O", out, "--partialcharge", "gasteiger"],
                       capture_output=True, text=True)
    if os.path.exists(out) and os.path.getsize(out) > 0: ok += 1
    else: fail += 1
    os.remove(sdf)
    if (ok + fail) % 500 == 0: print(f"  prepared {ok}, failed {fail}", flush=True)
print(f"\nligands prepared: {ok}, failed: {fail}")

# receptor
rec_pdb = os.path.join(PROJ, "docking", "9CVD_data_preparation", "9cvd.pdb")
rec_out = os.path.join(OUT, "receptor_9CVD.pdbqt")
subprocess.run([OBABEL, rec_pdb, "-O", rec_out, "-xr", "--partialcharge", "gasteiger"],
               capture_output=True, text=True)
print(f"receptor written: {os.path.exists(rec_out)}, size {os.path.getsize(rec_out) if os.path.exists(rec_out) else 0}")
import re
types = set(re.findall(r"\s(\w{1,2})\s*$", open(rec_out).read(), re.M))
print("receptor atom types present:", sorted(t for t in types if t)[:25])
json.dump(dict(n_actives=int(L.label.sum()-1), n_decoys=int((L.label==0).sum()),
               prepared=ok, failed=fail), open(os.path.join(OUT, "prep_summary.json"), "w"), indent=2)
print("DONE R13a")
