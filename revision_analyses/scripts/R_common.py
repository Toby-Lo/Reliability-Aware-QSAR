"""Shared helpers for the R1 revision analyses (NSD2 reliability-aware QSAR)."""
import os, json, glob
import numpy as np, pandas as pd
import joblib

ROOT = os.environ.get(
    "NSD2_R1_ROOT",
    "/mnt/e/Post-Dock/Luo Paper/NSD2_Manuscript_3rd_version20260705/NSD2_Manuscript_3nd_version20260705/Submission/R1",
)
PROJ = os.path.join(ROOT, "1.5_NSD2")
MODELS = os.path.join(PROJ, "models_out", "qsar_ml_20260412_162829")
OUT = os.path.join(ROOT, "Revision_R1")
TAB = os.path.join(OUT, "tables"); FIG = os.path.join(OUT, "figures"); ANA = os.path.join(OUT, "analysis")
for d in (TAB, FIG, ANA): os.makedirs(d, exist_ok=True)

MODEL_NAMES = ["SVC", "ETC", "RFC", "MLP", "XGBC", "LR"]
REF_MODEL = "SVC"
REP_SEED = 12345

# the 20-descriptor panel, in order
DESC_NAMES = ["MolWt","MolLogP","NumHDonors","NumHAcceptors","NumRotatableBonds","TPSA",
              "HeavyAtomCount","NumValenceElectrons","NumAliphaticRings","NumAromaticRings",
              "FractionCSP3","RingCount","LabuteASA","VSA_EState1","VSA_EState2",
              "SlogP_VSA1","SlogP_VSA2","SMR_VSA1","SMR_VSA2","EState_VSA1"]

# descriptors that are direct proxies of molecular size / bulk (Reviewer 1, comment 4)
SIZE_DESC = ["MolWt","HeavyAtomCount","NumValenceElectrons","LabuteASA","RingCount",
             "NumAromaticRings","NumAliphaticRings"]


def seed_dirs():
    out = {}
    for p in sorted(glob.glob(os.path.join(MODELS, "split_seed_*"))):
        out[int(os.path.basename(p).split("_")[-1])] = p
    return out


def load_split(seed_dir, which):
    """which in {'dev_train','external_test'}"""
    d = np.load(os.path.join(seed_dir, "data", "splits", which + ".npz"), allow_pickle=True)
    return dict(fp=d["fp"], desc=d["desc"], y=d["y"].astype(int),
                ids=d["id"].astype(str), smiles=d["smiles"].astype(str))


def fp_mask(seed_dir):
    return np.load(os.path.join(seed_dir, "feature_processors", "fp_mask.npy"))


def build_X(seed_dir, split, scaler=None, desc_keep=None):
    """Reproduce the 149-dim schema: masked fp first, then (optionally scaled) descriptors."""
    m = fp_mask(seed_dir)
    fp = split["fp"][:, m.astype(bool)] if m.dtype == bool or set(np.unique(m)) <= {0, 1} else split["fp"][:, m]
    desc = split["desc"].copy()
    if scaler is not None:
        desc = scaler.transform(desc)
    if desc_keep is not None:
        desc = desc[:, desc_keep]
    return np.hstack([fp, desc]).astype(np.float64)


def load_svc(seed_dir, seed):
    md = os.path.join(seed_dir, "models", "full_dev", "SVC", f"seed_{seed}")
    model = joblib.load(os.path.join(md, "model.joblib"))
    scaler = joblib.load(os.path.join(md, "scaler.joblib"))
    cfg = json.load(open(os.path.join(md, "model_config.json")))
    thr = cfg["best_hyperparameters"].get("classification_threshold", 0.5)
    return model, scaler, thr, cfg


def load_model(seed_dir, name, seed):
    md = os.path.join(seed_dir, "models", "full_dev", name, f"seed_{seed}")
    model = joblib.load(os.path.join(md, "model.joblib"))
    sp = os.path.join(md, "scaler.joblib")
    scaler = joblib.load(sp) if os.path.exists(sp) else None
    cfg = json.load(open(os.path.join(md, "model_config.json")))
    thr = cfg["best_hyperparameters"].get("classification_threshold", 0.5)
    return model, scaler, thr, cfg


# ---------------- enrichment metrics ----------------
def ef_at(y_true, score, frac):
    """Enrichment factor at a top fraction of the ranked list."""
    y_true = np.asarray(y_true); score = np.asarray(score)
    n = len(y_true); n_act = int(y_true.sum())
    if n_act == 0: return np.nan
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-score)[:k]
    return (y_true[idx].sum() / k) / (n_act / n)


def bedroc(y_true, score, alpha=20.0):
    """Boltzmann-enhanced discrimination of ROC (Truchon & Bayly, 2007)."""
    y_true = np.asarray(y_true); score = np.asarray(score)
    n = len(y_true); n_act = int(y_true.sum())
    if n_act == 0 or n_act == n:
        return np.nan
    order = np.argsort(-score)
    ranks = np.where(y_true[order] == 1)[0] + 1
    ra = n_act / n
    rie = (np.sum(np.exp(-alpha * ranks / n)) / n_act) / ((1.0 / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    return float((rie - rie_min) / (rie_max - rie_min))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        val = (m - i) * p[j]
        running = max(running, val)
        adj[j] = min(1.0, running)
    return adj


def boot_ci(x, func=np.mean, n=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    stats = np.array([func(rng.choice(x, size=len(x), replace=True)) for _ in range(n)])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def fmt(x, nd=3):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pstr(p):
    if p is None or not np.isfinite(p): return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
