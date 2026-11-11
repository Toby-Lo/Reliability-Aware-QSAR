"""R1 comment 7: quantitative convergence diagnostics for the 200 ns trajectories.

For every system: block averages of Ca RMSD over 25 ns windows, integrated
autocorrelation time and effective sample size, split-half comparison of the MM/GBSA
window (15-200 ns), PCA cosine content of the first two modes, and the drift of a
running mean. These establish which observables are converged enough to interpret and
which are reported qualitatively.
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *

MDR = os.path.join(PROJ, "mds", "Results")
SYS = {"a1a0m": "A1A0M", "18198200": "Z20D-18198200", "18198198": "Z20D-18198198",
       "18198196": "Z20D-18198196", "1857797811": "Z20D-1857797811",
       "1857797734": "Z20D-1857797734", "219210673": "Z20D-219210673",
       "36354596": "Z20D-36354596", "58175203": "Z20D-58175203"}
DT_NS = 0.01           # 10 ps per saved frame
MMGBSA_FRAMES = (1501, 20000)

def load_dat(p, col=1):
    a = np.loadtxt(p, comments="#")
    return a[:, col] if a.ndim > 1 else a

def act_ess(x):
    """Integrated autocorrelation time (initial positive sequence) and effective sample size."""
    x = np.asarray(x, float); x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    if ac[0] <= 0: return np.nan, np.nan
    ac /= ac[0]
    tau = 1.0
    for k in range(1, n):
        if ac[k] <= 0: break
        tau += 2.0 * ac[k]
    return float(tau), float(n / tau) if tau > 0 else np.nan

def cosine_content(proj):
    """Hess cosine content of a PC projection; values near 1 indicate random diffusion."""
    p = np.asarray(proj, float); n = len(p)
    t = np.arange(n)
    num = (np.trapezoid(np.cos(np.pi * t / n) * p, t)) ** 2
    den = np.trapezoid(p ** 2, t)
    return float(2.0 / n * num / den) if den > 0 else np.nan

rows, blocks = [], []
for key, label in SYS.items():
    d = os.path.join(MDR, f"{key}-Analysis")
    ca = load_dat(os.path.join(d, "RMSD-RMSF", "Name_Calpha_RMSD.dat"))
    ligp = [p for p in (os.path.join(d, "Ligand-RMSD", "lig_rmsd.dat"),
                        os.path.join(d, "Ligand-RMSD", "Ligand_RMSD.dat")) if os.path.exists(p)][0]
    lig = load_dat(ligp)
    rog = load_dat(os.path.join(d, "RoG", "Name-RoG.dat"))
    proj = np.loadtxt(os.path.join(d, "PCA", "projection.txt"), comments="#")
    n = len(ca); t_ns = np.arange(n) * DT_NS

    # 25 ns block averages of Ca RMSD
    nb = 8
    edges = np.linspace(0, n, nb + 1).astype(int)
    bm = [ca[edges[i]:edges[i+1]].mean() for i in range(nb)]
    for i, v in enumerate(bm):
        blocks.append(dict(system=label, block=i+1,
                           window_ns=f"{edges[i]*DT_NS:.0f}-{edges[i+1]*DT_NS:.0f}",
                           ca_rmsd_mean=v))

    # MM/GBSA analysis window
    w = slice(MMGBSA_FRAMES[0]-1, MMGBSA_FRAMES[1])
    caw, ligw = ca[w], lig[w]
    half = len(caw) // 2
    tau_ca, ess_ca = act_ess(caw)
    tau_lig, ess_lig = act_ess(ligw)

    # drift: slope of Ca RMSD over the analysis window, per 100 ns
    sl = np.polyfit(t_ns[w], caw, 1)[0] * 100.0

    rows.append(dict(system=label,
        ca_rmsd_mean=caw.mean(), ca_rmsd_sd=caw.std(ddof=1),
        ca_rmsd_last50_mean=ca[-5000:].mean(), ca_rmsd_last50_sd=ca[-5000:].std(ddof=1),
        ca_block_range=max(bm)-min(bm), ca_block_sd=np.std(bm, ddof=1),
        drift_A_per_100ns=sl,
        half1_ca=caw[:half].mean(), half2_ca=caw[half:].mean(),
        half_diff_ca=caw[half:].mean()-caw[:half].mean(),
        half1_lig=ligw[:half].mean(), half2_lig=ligw[half:].mean(),
        half_diff_lig=ligw[half:].mean()-ligw[:half].mean(),
        tau_ca_ns=tau_ca*DT_NS, ess_ca=ess_ca, tau_lig_ns=tau_lig*DT_NS, ess_lig=ess_lig,
        cos_pc1=cosine_content(proj[w, 1]), cos_pc2=cosine_content(proj[w, 2]),
        rog_mean=rog[w].mean(), rog_sd=rog[w].std(ddof=1)))
    print(f"{label:<18} drift {sl:+.3f} A/100ns | half-half dCa {rows[-1]['half_diff_ca']:+.3f} A | "
          f"tau_Ca {tau_ca*DT_NS:.1f} ns, ESS {ess_ca:.0f} | cos(PC1) {rows[-1]['cos_pc1']:.2f}")

C = pd.DataFrame(rows)
C.to_csv(os.path.join(TAB, "R09_md_convergence.csv"), index=False)
pd.DataFrame(blocks).to_csv(os.path.join(ANA, "R09_block_averages.csv"), index=False)

print("\nSummary across the nine systems:")
print(f"  |drift| over 15-200 ns: median {C.drift_A_per_100ns.abs().median():.3f}, "
      f"max {C.drift_A_per_100ns.abs().max():.3f} A/100 ns")
print(f"  |half-half difference| in Ca RMSD: median {C.half_diff_ca.abs().median():.3f}, "
      f"max {C.half_diff_ca.abs().max():.3f} A")
print(f"  Ca autocorrelation time: median {C.tau_ca_ns.median():.1f} ns "
      f"(range {C.tau_ca_ns.min():.1f}-{C.tau_ca_ns.max():.1f})")
print(f"  effective independent samples (Ca): median {C.ess_ca.median():.0f} "
      f"(range {C.ess_ca.min():.0f}-{C.ess_ca.max():.0f})")
print(f"  cosine content PC1: median {C.cos_pc1.median():.2f} "
      f"(range {C.cos_pc1.min():.2f}-{C.cos_pc1.max():.2f})")
print("  (cosine content near 1 indicates the mode is dominated by random diffusion,")
print("   so the PC1-PC2 landscapes are interpreted as sampling descriptions only)")

json.dump(C.round(4).to_dict(orient="records"),
          open(os.path.join(ANA, "R09_summary.json"), "w"), indent=2)
print("\nDONE R09")
