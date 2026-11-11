"""R2 comment 6: systematic sensitivity of the multi-objective ranking to its weights.

S_final = wP * P_cal + wR * S_rel + wA * S_ADMET, with wP + wR + wA = 1. The full
simplex is scanned on a 0.05 grid and every scheme is compared with the published
0.5 / 0.2 / 0.3 setting: Jaccard and overlap of the top-N sets, Spearman rank
correlation of the whole ranking, and retention of the eight MD-shortlisted compounds.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

SRC = os.path.join(MODELS, "virtual_screening", "ad_screening_results_20260419_140044",
                   "admet_scoring_20260419_173114", "admet_scored.csv")
DEFAULT = (0.5, 0.2, 0.3)
TOPN = [10, 25, 50, 100, 250, 500]
STEP = 0.05

df = pd.read_csv(SRC)
need = ["zinc_id", "qsar_prob", "ad_score", "admet_score", "qsar_ad_rank_score_raw"]
df = df[need].dropna().reset_index(drop=True)
print(f"ranked candidate set: {len(df)} compounds")

# S_rel = P_cal * S_AD^gamma (gamma = 2), exactly as in Eq 9
P = df.qsar_prob.values.astype(float)
S_rel = P * df.ad_score.values.astype(float) ** 2.0
A = df.admet_score.values.astype(float)
print(f"check vs recorded S_rel column: max abs diff = "
      f"{np.max(np.abs(S_rel - df.qsar_ad_rank_score_raw.values)):.2e}")

def score(wP, wR, wA):
    return wP * P + wR * S_rel + wA * A

base = score(*DEFAULT)
base_order = np.argsort(-base)
base_top = {n: set(df.zinc_id.values[base_order[:n]]) for n in TOPN}

shortlist = [18198198, 1857797811, 36354596, 18198200, 58175203,
             18198196, 219210673, 1857797734]
present = [z for z in shortlist if z in set(df.zinc_id.values)]
print(f"MD-shortlisted compounds present in the ranked set: {len(present)}/8")

grid = [(round(a, 2), round(b, 2), round(1 - a - b, 2))
        for a in np.arange(0, 1.0001, STEP) for b in np.arange(0, 1.0001 - a + 1e-9, STEP)]
grid = [g for g in grid if abs(sum(g) - 1) < 1e-9 and min(g) >= 0]
print(f"weight schemes on the simplex: {len(grid)}")

rows = []
for wP, wR, wA in grid:
    s = score(wP, wR, wA)
    order = np.argsort(-s)
    ids = df.zinc_id.values[order]
    r = dict(wP=wP, wR=wR, wA=wA, spearman_full=spearmanr(base, s)[0])
    for n in TOPN:
        top = set(ids[:n])
        r[f"overlap_top{n}"] = len(top & base_top[n]) / n
        r[f"jaccard_top{n}"] = len(top & base_top[n]) / len(top | base_top[n])
    r["shortlist_in_top100"] = sum(1 for z in present if z in set(ids[:100]))
    r["shortlist_in_top250"] = sum(1 for z in present if z in set(ids[:250]))
    rows.append(r)
W = pd.DataFrame(rows)
W.to_csv(os.path.join(ANA, "R12_weight_simplex_full.csv"), index=False)

print("\nAcross all %d weight schemes on the simplex:" % len(W))
for n in TOPN:
    c = W[f"overlap_top{n}"]
    print(f"  Top-{n:<4} overlap with the published scheme: "
          f"min {c.min():.2f}, median {c.median():.2f}, mean {c.mean():.2f}")
print(f"  full-ranking Spearman vs published: min {W.spearman_full.min():.3f}, "
      f"median {W.spearman_full.median():.3f}")
print(f"  MD shortlist retained in Top-100: min {W.shortlist_in_top100.min()}/{len(present)}, "
      f"median {W.shortlist_in_top100.median():.0f}/{len(present)}, "
      f"schemes retaining all {len(present)}: {int((W.shortlist_in_top100==len(present)).sum())}/{len(W)}")

# restrict to a plausible neighbourhood of the published scheme (each weight +/- 0.15)
nb = W[(W.wP.between(0.35, 0.65)) & (W.wR.between(0.05, 0.35)) & (W.wA.between(0.15, 0.45))]
print(f"\nWithin a +/-0.15 neighbourhood of the published scheme ({len(nb)} schemes):")
for n in TOPN:
    c = nb[f"overlap_top{n}"]
    print(f"  Top-{n:<4} overlap: min {c.min():.2f}, median {c.median():.2f}")
print(f"  full-ranking Spearman: min {nb.spearman_full.min():.3f}")
print(f"  MD shortlist retained in Top-100: min {nb.shortlist_in_top100.min()}/{len(present)}")

# compact table for the supplement
tab = []
for n in TOPN:
    tab.append(dict(candidate_set=f"Top {n}",
                    overlap_min_all=W[f"overlap_top{n}"].min(),
                    overlap_median_all=W[f"overlap_top{n}"].median(),
                    overlap_min_neighbourhood=nb[f"overlap_top{n}"].min(),
                    overlap_median_neighbourhood=nb[f"overlap_top{n}"].median()))
T = pd.DataFrame(tab)
T.to_csv(os.path.join(TAB, "R12_weight_sensitivity_summary.csv"), index=False)
print("\nSupplementary table:")
print(T.round(3).to_string(index=False))

# the four schemes already reported, for continuity with the original Table S7
orig = [(0.5,0.2,0.3), (0.6,0.2,0.2), (0.4,0.3,0.3), (0.5,0.3,0.2)]
sub = W[[tuple(r) in [tuple(o) for o in orig] for r in W[["wP","wR","wA"]].values]]
sub.to_csv(os.path.join(ANA, "R12_original_four_schemes.csv"), index=False)

json.dump(dict(n_schemes=int(len(W)), n_neighbourhood=int(len(nb)),
               shortlist_present=len(present),
               overlap_all={f"top{n}": [float(W[f'overlap_top{n}'].min()),
                                        float(W[f'overlap_top{n}'].median())] for n in TOPN},
               overlap_nb={f"top{n}": [float(nb[f'overlap_top{n}'].min()),
                                       float(nb[f'overlap_top{n}'].median())] for n in TOPN},
               spearman_min_all=float(W.spearman_full.min()),
               spearman_min_nb=float(nb.spearman_full.min()),
               shortlist_min_top100_nb=int(nb.shortlist_in_top100.min()),
               shortlist_all_retained_schemes=int((W.shortlist_in_top100==len(present)).sum())),
          open(os.path.join(ANA, "R12_summary.json"), "w"), indent=2)
print("\nDONE R12")
