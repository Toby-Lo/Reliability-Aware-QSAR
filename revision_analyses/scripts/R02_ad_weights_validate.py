import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_common import *
from R_ad import *

sd = seed_dirs()[REP_SEED]
dev = load_split(sd, "dev_train"); ext = load_split(sd, "external_test")
m = fp_mask(sd).astype(bool)
t = ad_terms(dev["fp"][:, m], dev["desc"], ext["fp"][:, m], ext["desc"])
print(f"n_comp={t['n_components']} h_star={t['h_star']:.7f} h0={t['h0']:.8f}")
print("published  n_comp=70 h_star=0.5011933 h0=0.15180606")
pred = pd.read_csv(os.path.join(sd, "predictions", "external_test_predictions.csv"))
pred = pred[pred.model == REF_MODEL].set_index("id").loc[ext["ids"]]
err = per_sample_logloss(pred.y_true.values, pred.y_prob.values)
s = ad_score(t, 0.9, 0.05)
print(f"\nPearson  r(S_AD, logloss) = {pearsonr(s, err)[0]:.6f}   published = -0.23324020")
print(f"Spearman rho              = {spearmanr(s, err)[0]:.6f}")
print(f"\nrates: tanimoto>=0.85 {(t['tanimoto']>=TAN_THR).mean():.6f} (pub 0.6036036)"
      f" | cosine>=0.70 {(t['cosine']>=COS_THR).mean():.6f} (pub 1.0)"
      f" | strict {((t['tanimoto']>=TAN_THR)&(t['cosine']>=COS_THR)).mean():.6f} (pub 0.6036036)"
      f" | leverage<=h* {(t['leverage']<=t['h_star']).mean():.6f} (pub 0.9369369)")
print(f"\nS_AD mean={s.mean():.6f} sd={s.std():.6f} min={s.min():.6f} max={s.max():.6f}")
