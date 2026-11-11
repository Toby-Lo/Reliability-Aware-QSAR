"""R1 comment 6: ligand net charges, and MM/GBSA errors corrected for trajectory autocorrelation.

MMPBSA.py reports a standard error that treats all 9250 snapshots as independent. The
effective sample size from the Ca RMSD autocorrelation over the same window is of order
5, so the realistic uncertainty is larger by sqrt(N/N_eff). Both are reported.
"""
import os, sys, json, re
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
RDLogger.DisableLog("rdApp.*")

MDR = os.path.join(PROJ, "mds", "Results")
SYS = {"18198200": "Z20D-18198200", "a1a0m": "A1A0M", "1857797811": "Z20D-1857797811",
       "58175203": "Z20D-58175203", "219210673": "Z20D-219210673",
       "18198198": "Z20D-18198198", "36354596": "Z20D-36354596",
       "1857797734": "Z20D-1857797734", "18198196": "Z20D-18198196"}
KEYS = ["VDWAALS", "EEL", "EGB", "ESURF", "DELTA G gas", "DELTA G solv", "DELTA TOTAL"]

def parse_delta(path):
    txt = open(path, errors="ignore").read()
    # the DELTA block is the final section of the file
    blk = txt.split("Differences (Complex - Receptor - Ligand):")[-1]
    out, nframes = {}, None
    mfr = re.search(r"using\s+([\d.]+)\s+complex frames", txt)
    if mfr: nframes = float(mfr.group(1))
    for line in blk.splitlines():
        for k in KEYS:
            if line.strip().startswith(k):
                nums = re.findall(r"-?\d+\.\d+", line[len(k):])
                if len(nums) >= 3:
                    out[k] = dict(mean=float(nums[0]), sd=float(nums[1]), sem=float(nums[2]))
                break
    return out, nframes

conv = pd.read_csv(os.path.join(TAB, "R09_md_convergence.csv")).set_index("system")

# ligand net formal charges
cand = pd.read_csv(os.path.join(PROJ, "docking", "Final_candidates", "final_cadicates_for_mds.csv"))
smi = {f"Z20D-{int(r.zinc_id)}": r.smiles for _, r in cand.iterrows()}
ref = Chem.MolFromMolFile(os.path.join(PROJ, "docking", "9CVD_data_preparation", "A1A0M.sdf"))
if ref is not None:
    smi["A1A0M"] = Chem.MolToSmiles(ref)

rows = []
for key, label in SYS.items():
    p = os.path.join(MDR, f"{key}-Analysis", "MMPBSA-GBSA", "FINAL_RESULTS_MMPBSA.dat")
    d, nfr = parse_delta(p)
    tot = d.get("DELTA TOTAL", {})
    ess = float(conv.loc[label, "ess_ca"]) if label in conv.index else np.nan
    infl = np.sqrt(nfr / ess) if (nfr and ess and ess > 0) else np.nan
    m = Chem.MolFromSmiles(smi[label]) if label in smi else None
    q = Chem.GetFormalCharge(m) if m is not None else np.nan
    nheavy = m.GetNumHeavyAtoms() if m is not None else np.nan
    rows.append(dict(system=label, net_formal_charge=q, heavy_atoms=nheavy,
                     vdw=d.get("VDWAALS", {}).get("mean"), eel=d.get("EEL", {}).get("mean"),
                     egb=d.get("EGB", {}).get("mean"), esurf=d.get("ESURF", {}).get("mean"),
                     dG=tot.get("mean"), sd=tot.get("sd"), sem_mmpbsa=tot.get("sem"),
                     n_frames=nfr, ess_trajectory=ess, inflation_factor=infl,
                     sem_corrected=tot.get("sd") / np.sqrt(ess) if (tot and ess and ess > 0) else np.nan))
M = pd.DataFrame(rows).sort_values("dG")
M["dG_vs_reference"] = M.dG - float(M.loc[M.system == "A1A0M", "dG"].iloc[0])
M.to_csv(os.path.join(TAB, "R10_mmgbsa_charges_errors.csv"), index=False)
print(M[["system","net_formal_charge","heavy_atoms","eel","egb","dG","sd","sem_mmpbsa",
         "ess_trajectory","sem_corrected","dG_vs_reference"]].round(2).to_string(index=False))

print("\nNet formal charge distribution:", dict(M.net_formal_charge.value_counts().sort_index()))
print(f"Electrostatic term range: {M.eel.min():.1f} to {M.eel.max():.1f} kcal/mol")
print(f"Polar solvation range:    {M.egb.min():.1f} to {M.egb.max():.1f} kcal/mol")
print(f"EEL + EGB (compensated):  {(M.eel + M.egb).min():.1f} to {(M.eel + M.egb).max():.1f} kcal/mol")
print(f"  correlation between EEL and EGB across systems: r = {np.corrcoef(M.eel, M.egb)[0,1]:.3f}")
print(f"\nMMPBSA.py SEM: {M.sem_mmpbsa.min():.3f}-{M.sem_mmpbsa.max():.3f} kcal/mol")
print(f"Autocorrelation-corrected SEM: {M.sem_corrected.min():.2f}-{M.sem_corrected.max():.2f} kcal/mol "
      f"(inflation {M.inflation_factor.min():.0f}-{M.inflation_factor.max():.0f} x)")

# do the corrected errors still separate the top candidate from the reference?
a = M[M.system == "Z20D-18198200"].iloc[0]; b = M[M.system == "A1A0M"].iloc[0]
diff = a.dG - b.dG
se = np.sqrt(a.sem_corrected ** 2 + b.sem_corrected ** 2)
print(f"\nZ20D-18198200 vs A1A0M: {diff:+.2f} kcal/mol, corrected SE {se:.2f} "
      f"-> |diff|/SE = {abs(diff)/se:.2f}")
print("  (a ratio below ~2 means the two are not separated once correlation is accounted for)")

json.dump(M.round(4).to_dict(orient="records"), open(os.path.join(ANA, "R10_summary.json"), "w"), indent=2)
print("\nDONE R10")
