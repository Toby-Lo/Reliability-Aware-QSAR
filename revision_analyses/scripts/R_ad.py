"""Applicability-domain terms, reimplemented exactly as published (Eqs 3-8)."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PCA_COMPONENTS = 70
AD_POWER = 2.0
TAN_THR = 0.85
COS_THR = 0.70


def binarize(fp):
    return (np.asarray(fp) > 0).astype(np.int32)


def tanimoto_max(train_fp_bin, query_fp_bin):
    tr = train_fp_bin.astype(np.int32); q = query_fp_bin.astype(np.int32)
    tr_sum = tr.sum(1)
    out = np.zeros(len(q))
    for i, v in enumerate(q):
        inter = tr @ v
        union = tr_sum + int(v.sum()) - inter
        union = np.where(union == 0, 1, union)
        out[i] = float(np.max(inter / union))
    return out


def cosine_max(train_X, query_X, eps=1e-12):
    tr = train_X.astype(np.float64)
    tn = np.linalg.norm(tr, axis=1, keepdims=True); tn = np.where(tn < eps, 1.0, tn)
    tru = tr / tn
    q = query_X.astype(np.float64)
    qn = np.linalg.norm(q, axis=1, keepdims=True); qn = np.where(qn < eps, 1.0, qn)
    return np.max((q / qn) @ tru.T, axis=1)


def leverage(train_Xs, query_Xs, n_comp=PCA_COMPONENTS):
    n_comp = int(min(n_comp, train_Xs.shape[0], train_Xs.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="full").fit(train_Xs)
    Z = pca.transform(train_Xs); Zq = pca.transform(query_Xs)
    G = np.linalg.pinv(Z.T @ Z)
    h_q = np.einsum("ij,jk,ik->i", Zq, G, Zq)
    h_tr = np.einsum("ij,jk,ik->i", Z, G, Z)
    h_star = 3.0 * Z.shape[1] / Z.shape[0]
    return h_q, h_tr, h_star, n_comp


def density_support(h, h0):
    return 1.0 / (1.0 + (np.clip(h, 0, None) / max(h0, 1e-12)))


def ad_terms(train_fp, train_desc, query_fp, query_desc):
    """Return dict with tanimoto, cosine, density, leverage, h_star for the query set."""
    tr_bin = binarize(train_fp); q_bin = binarize(query_fp)
    tan = tanimoto_max(tr_bin, q_bin)
    # cosine similarity over STANDARDISED DESCRIPTOR vectors (Eq 4)
    dsc = StandardScaler().fit(train_desc.astype(np.float64))
    cos = cosine_max(dsc.transform(train_desc.astype(np.float64)), dsc.transform(query_desc.astype(np.float64)))
    # leverage / density in the standardised full 149-dim space (Eqs 6-7)
    full_tr = np.hstack([train_fp, train_desc]).astype(np.float64)
    full_q = np.hstack([query_fp, query_desc]).astype(np.float64)
    sc = StandardScaler().fit(full_tr)
    trs = sc.transform(full_tr); qs = sc.transform(full_q)
    h_q, h_tr, h_star, ncomp = leverage(trs, qs)
    h0 = float(np.median(h_tr))
    dens = density_support(h_q, h0)
    return dict(tanimoto=tan, cosine=np.clip(cos, 0, 1), density=np.clip(dens, 0, 1),
                leverage=h_q, h_star=h_star, h0=h0, n_components=ncomp)


def ad_score(terms, w1, w3):
    sim = w1 * terms["tanimoto"] + (1 - w1) * terms["cosine"]
    return np.clip(w3 * sim + (1 - w3) * terms["density"], 0, 1)


def per_sample_logloss(y, p, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
