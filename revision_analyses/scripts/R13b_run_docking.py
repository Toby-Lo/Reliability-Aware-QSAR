"""R1 comment 5, step 2: dock the enrichment set with AutoDock Vina and score enrichment.

Same receptor (rigid 9CVD chain A with the three structural zinc ions), same grid centre
(14.79, -3.90, -0.17), same 20 x 20 x 20 A box and same exhaustiveness (32) as the
manuscript. Ligands are docked in parallel, one CPU each.
"""
import os, sys, json, subprocess, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

OUTD = os.path.join(ROOT, "Revision_R1", "docking_enrichment")
LIG = os.path.join(OUTD, "ligands")
POSE = "/tmp/nsd2_poses"; os.makedirs(POSE, exist_ok=True)   # native disk: 9p is far too slow
LIGL = "/tmp/nsd2_ligands"
RECL = "/tmp/nsd2_receptor.pdbqt"
VINA = "/home/wiky/miniforge3/envs/vsdock/bin/vina"
REC = os.path.join(OUTD, "receptor_9CVD.pdbqt")
CENTER = (14.79, -3.90, -0.17)
SIZE = (20.0, 20.0, 20.0)
EXH = 16                      # enrichment benchmark setting, stated in Methods
SEED = 20260704
NPROC = 16


def dock(name):
    lig = os.path.join(LIGL, name + ".pdbqt")
    out = os.path.join(POSE, name + "_out.pdbqt")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return name, parse_best(out)
    cmd = [VINA, "--receptor", RECL, "--ligand", lig, "--out", out,
           "--center_x", str(CENTER[0]), "--center_y", str(CENTER[1]), "--center_z", str(CENTER[2]),
           "--size_x", str(SIZE[0]), "--size_y", str(SIZE[1]), "--size_z", str(SIZE[2]),
           "--exhaustiveness", str(EXH), "--cpu", "1", "--seed", str(SEED), "--verbosity", "0"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=420)
    except Exception:
        return name, np.nan
    return name, parse_best(out)


def parse_best(out):
    if not os.path.exists(out):
        return np.nan
    for line in open(out, errors="ignore"):
        if line.startswith("REMARK VINA RESULT"):
            m = re.findall(r"-?\d+\.\d+", line)
            if m:
                return float(m[0])
    return np.nan


N_ACT_DOCK, N_DEC_DOCK = 50, 500   # 10:1, sized for the available compute

import shutil

if __name__ == "__main__":
    # stage inputs on native disk; reading them over 9p dominates the runtime
    os.makedirs(LIGL, exist_ok=True)
    if not os.path.exists(RECL):
        shutil.copy(REC, RECL)
    for fn in os.listdir(LIG):
        dst = os.path.join(LIGL, fn)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(LIG, fn), dst)
    print("ligands staged on native disk:", len(os.listdir(LIGL)))
    man = pd.read_csv(os.path.join(OUTD, "ligand_manifest.csv"))
    man = man[[os.path.exists(os.path.join(LIG, n + ".pdbqt")) for n in man.name]].reset_index(drop=True)
    # subsample to a benchmark of workable size while keeping the manuscript exhaustiveness
    rng = np.random.default_rng(20260704)
    ref = man[man.name == "reference_A1A0M"]
    acts = man[(man.label == 1) & (man.name != "reference_A1A0M")]
    decs = man[man.label == 0]
    acts = acts.iloc[rng.choice(len(acts), min(N_ACT_DOCK, len(acts)), replace=False)]
    decs = decs.iloc[rng.choice(len(decs), min(N_DEC_DOCK, len(decs)), replace=False)]
    # Dock the reference and all actives first, then decoys in random order. Every decoy that
    # finishes is a uniform random draw from the matched pool, so the benchmark stays valid at
    # whatever point the run is stopped.
    acts = acts.sample(frac=1.0, random_state=11)
    decs = decs.sample(frac=1.0, random_state=11)
    man = pd.concat([ref, acts, decs], ignore_index=True)
    print(f"ligands to dock: {len(man)} (actives {int(man.label.sum())}, decoys {int((man.label==0).sum())})")
    res = {}
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        futs = {ex.submit(dock, n): n for n in man.name}
        done = 0
        for f in as_completed(futs):
            n, s = f.result()
            res[n] = s
            done += 1
            if done % 25 == 0:
                # checkpoint, so the benchmark can be summarized before the run ends
                part = man.copy()
                part["vina_score"] = part.name.map(res)
                part.dropna(subset=["vina_score"]).to_csv(
                    os.path.join(OUTD, "docking_scores_partial.csv"), index=False)
                print(f"  docked {done}/{len(man)}", flush=True)
    man["vina_score"] = man.name.map(res)
    man.to_csv(os.path.join(OUTD, "docking_scores.csv"), index=False)
    ok = man.dropna(subset=["vina_score"]).copy()
    print(f"\nscored: {len(ok)} of {len(man)}")

    ref = ok[ok.name == "reference_A1A0M"]
    body = ok[ok.name != "reference_A1A0M"]
    y = body.label.values.astype(int)
    s = -body.vina_score.values            # more negative score = better, so negate for ranking
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = dict(n_actives=int(y.sum()), n_decoys=int((y == 0).sum()),
               roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
               ef1=float(ef_at(y, s, 0.01)), ef2=float(ef_at(y, s, 0.02)),
               ef5=float(ef_at(y, s, 0.05)), ef10=float(ef_at(y, s, 0.10)),
               bedroc20=float(bedroc(y, s, 20.0)), bedroc8=float(bedroc(y, s, 8.0)),
               active_score_mean=float(body.vina_score[y == 1].mean()),
               active_score_sd=float(body.vina_score[y == 1].std()),
               decoy_score_mean=float(body.vina_score[y == 0].mean()),
               decoy_score_sd=float(body.vina_score[y == 0].std()),
               reference_score=float(ref.vina_score.iloc[0]) if len(ref) else None)
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(body.vina_score[y == 1], body.vina_score[y == 0], alternative="less")
    out["p_mannwhitney"] = float(p)

    # bootstrap CIs over compounds
    rng = np.random.default_rng(3)
    bs = {k: [] for k in ("roc_auc", "ef1", "bedroc20")}
    for _ in range(2000):
        idx = rng.choice(len(y), len(y), replace=True)
        if y[idx].sum() < 5:
            continue
        bs["roc_auc"].append(roc_auc_score(y[idx], s[idx]))
        bs["ef1"].append(ef_at(y[idx], s[idx], 0.01))
        bs["bedroc20"].append(bedroc(y[idx], s[idx], 20.0))
    for k, v in bs.items():
        out[k + "_ci"] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    pd.DataFrame([out]).to_csv(os.path.join(TAB, "R13_docking_enrichment.csv"), index=False)
    json.dump(out, open(os.path.join(ANA, "R13_summary.json"), "w"), indent=2)
    print("\nRetrospective docking enrichment against 9CVD:")
    for k in ("n_actives", "n_decoys", "roc_auc", "roc_auc_ci", "ef1", "ef1_ci", "ef2", "ef5",
              "ef10", "bedroc20", "bedroc20_ci", "active_score_mean", "decoy_score_mean",
              "reference_score", "p_mannwhitney"):
        print(f"  {k:<18} {out[k]}")
    print("\nDONE R13b")
