"""
Round-3 reviewer Q1: is there a low-dimensional summary of (eta_up, eta_down) that
predicts asymmetric-heuristic complementarity better than the ratio or eta_down alone?
Uses the 5-pair sweep already run in followup.py (results_eta_ratio.csv, 20 seeds/cell).
Candidate summaries: ratio (eta_down/eta_up), difference (eta_down-eta_up), product.
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results_eta_ratio.csv")
g = df.groupby(["label", "eta_up", "eta_down", "ratio", "p_ai"])["complementarity"].mean().reset_index()
g["diff"] = g.eta_down - g.eta_up
g["prod"] = g.eta_down * g.eta_up

print(g.to_string())

for p_ai, sub in g.groupby("p_ai"):
    print(f"\n--- p_ai={p_ai} (n={len(sub)} pairs) ---")
    y = sub["complementarity"].values
    for pred in ["ratio", "diff", "prod", "eta_down", "eta_up"]:
        x = sub[pred].values
        r, p = stats.pearsonr(x, y)
        print(f"  {pred:10s} r={r:+.4f} r2={r**2:.4f} p={p:.4f}")
    # two-predictor OLS: eta_up, eta_down jointly (rank-2, 5 pts -> 2 df resid)
    X = np.column_stack([sub.eta_up.values, sub.eta_down.values, np.ones(len(sub))])
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred_y = X @ beta
    ss_res = np.sum((y - pred_y) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_full = 1 - ss_res / ss_tot
    print(f"  full (eta_up,eta_down) OLS: coef_up={beta[0]:+.4f} coef_down={beta[1]:+.4f} intercept={beta[2]:+.4f} R2={r2_full:.4f}")
