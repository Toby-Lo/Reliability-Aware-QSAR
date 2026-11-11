"""R1 comment 8: scaffold diversity and training-set similarity of the prioritised sets."""
import os, sys, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
RDLogger.DisableLog("rdApp.*")

TOP1000 = os.path.join(MODELS, "virtual_screening", "ad_screening_results_20260419_140044", "top1000_full_info.csv")
TOP100  = os.path.join(PROJ, "docking", "Final_candidates", "qsar_admet_docking_summary_TOP100_9CVD.csv")
TOP8    = os.path.join(PROJ, "docking", "Final_candidates", "final_cadicates_for_mds.csv")
TRAIN   = os.path.join(PROJ, "data", "nsd2_test", "nsd2_final_dataset.csv")

def mols(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None: out.append((s, m))
    return out

def murcko(m, generic=False):
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if generic: sc = MurckoScaffold.MakeScaffoldGeneric(sc)
        return Chem.MolToSmiles(sc)
    except Exception:
        return None

gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
def fps(ms): return [gen.GetFingerprint(m) for _, m in ms]

tr = pd.read_csv(TRAIN)
tr_m = mols(tr.smiles.tolist()); tr_fp = fps(tr_m)
tr_act = tr[tr.label == 1]
tr_act_m = mols(tr_act.smiles.tolist()); tr_act_fp = fps(tr_act_m)
tr_scaf = set(filter(None, (murcko(m) for _, m in tr_m)))
print(f"training set: {len(tr_m)} molecules, {len(tr_scaf)} Bemis-Murcko scaffolds "
      f"({len(tr_act_m)} actives)")

def profile(name, smis, ids=None):
    ms = mols(smis); f = fps(ms)
    sc  = [murcko(m) for _, m in ms]
    scg = [murcko(m, True) for _, m in ms]
    maxT  = [max(DataStructs.BulkTanimotoSimilarity(x, tr_fp)) for x in f]
    maxTa = [max(DataStructs.BulkTanimotoSimilarity(x, tr_act_fp)) for x in f]
    uniq_sc = len(set(filter(None, sc))); uniq_scg = len(set(filter(None, scg)))
    shared = len(set(filter(None, sc)) & tr_scaf)
    row = dict(set=name, n=len(ms),
               n_murcko_scaffolds=uniq_sc, scaffold_to_compound=uniq_sc/len(ms),
               n_generic_frameworks=uniq_scg,
               scaffolds_shared_with_training=shared,
               maxT_train_mean=float(np.mean(maxT)), maxT_train_median=float(np.median(maxT)),
               maxT_train_min=float(np.min(maxT)), maxT_train_max=float(np.max(maxT)),
               maxT_actives_mean=float(np.mean(maxTa)), maxT_actives_median=float(np.median(maxTa)),
               frac_maxT_ge_0p85=float(np.mean(np.array(maxT) >= 0.85)),
               frac_maxT_ge_0p40=float(np.mean(np.array(maxT) >= 0.40)))
    det = pd.DataFrame(dict(set=name, id=(ids if ids is not None else range(len(ms))),
                            smiles=[s for s, _ in ms], murcko=sc, generic_framework=scg,
                            max_tanimoto_train=maxT, max_tanimoto_actives=maxTa))
    return row, det

t1000 = pd.read_csv(TOP1000)
t100  = pd.read_csv(TOP100)
t8    = pd.read_csv(TOP8)

rows, dets = [], []
for name, df, idc in [("Top 1000 (ranked candidates)", t1000, "zinc_id"),
                      ("Top 100 (docked)", t100, "zinc_id"),
                      ("Top 8 (MD shortlist)", t8, "zinc_id")]:
    r, d = profile(name, df.smiles.tolist(), df[idc].astype(str).tolist())
    rows.append(r); dets.append(d)
    print(f"\n{name}: n={r['n']}, Murcko scaffolds={r['n_murcko_scaffolds']} "
          f"(ratio {r['scaffold_to_compound']:.3f}), generic frameworks={r['n_generic_frameworks']}, "
          f"shared with training={r['scaffolds_shared_with_training']}")
    print(f"   max Tanimoto to training set: mean {r['maxT_train_mean']:.3f}, "
          f"median {r['maxT_train_median']:.3f}, range {r['maxT_train_min']:.3f}-{r['maxT_train_max']:.3f}")
    print(f"   fraction with max Tanimoto >= 0.85: {r['frac_maxT_ge_0p85']:.3f}")

S = pd.DataFrame(rows); S.to_csv(os.path.join(TAB, "R06_scaffold_novelty_summary.csv"), index=False)
DET = pd.concat(dets, ignore_index=True); DET.to_csv(os.path.join(ANA, "R06_scaffold_details.csv"), index=False)

# scaffold population of the top 1000
d1 = dets[0]
pop = d1.murcko.value_counts()
pop.to_csv(os.path.join(ANA, "R06_top1000_scaffold_population.csv"))
print(f"\nTop-1000 scaffold population: largest family {pop.iloc[0]} compounds "
      f"({100*pop.iloc[0]/len(d1):.1f}%), singletons {int((pop==1).sum())} "
      f"({100*(pop==1).sum()/len(pop):.1f}% of scaffolds)")
print("ten most populated scaffolds (count):", pop.head(10).tolist())

# the shortlist in detail, including constitutional (stereochemistry-free) identity
d8 = dets[2].copy()
d8["flat_smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in d8.smiles]
d8.to_csv(os.path.join(TAB, "R06_top8_scaffold_detail.csv"), index=False)
print("\nMD shortlist detail:")
for _, r in d8.iterrows():
    print(f"  {r['id']:>12}  maxT(train)={r.max_tanimoto_train:.3f}  murcko={r.murcko}")
print(f"\n  distinct Murcko scaffolds among the 8: {d8.murcko.nunique()}")
print(f"  distinct generic frameworks:            {d8.generic_framework.nunique()}")
print(f"  distinct constitutions (stereochemistry ignored): {d8.flat_smiles.nunique()}")
grp = d8.groupby("flat_smiles").id.apply(list)
for k, v in grp.items():
    if len(v) > 1: print(f"  stereoisomer set: {v}")

json.dump(dict(summary=S.to_dict(orient="records"),
               training_scaffolds=len(tr_scaf),
               top8_murcko=int(d8.murcko.nunique()),
               top8_generic=int(d8.generic_framework.nunique()),
               top8_constitutions=int(d8.flat_smiles.nunique()),
               top1000_largest_family=int(pop.iloc[0]),
               top1000_n_scaffolds=int(len(pop)),
               top1000_singletons=int((pop == 1).sum())),
          open(os.path.join(ANA, "R06_summary.json"), "w"), indent=2)
print("\nDONE R06")
