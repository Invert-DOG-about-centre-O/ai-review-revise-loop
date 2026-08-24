"""
Follow-up analysis for round-2 review questions:
1) Does eta_down/eta_up RATIO (not eta_down alone) drive the flip-point shift?
   Sweep (eta_up, eta_down) pairs: same eta_down at different eta_up (breaks ratio),
   and same ratio at different absolute scale.
2) Does Bayesian flip-point insensitivity to conf_beta hold as beta -> 0 (near-uninformative)?
   Extend the beta sweep down to 0.4, 0.7 in addition to 1.0/1.5/2.5.
3) Formal paired test of hetero-vs-homogeneous complementarity, per cell, all reliabilities,
   asymmetric + bayesian, to check the ~0.003 gap flagged at p_ai=0.65/asymmetric.
"""
import numpy as np
import pandas as pd
from scipy import stats
import time
from sim import run_one, stable_hash, RNG_GLOBAL_SEED

t0 = time.time()
N, T, n_seeds = 200, 300, 20
reliabilities = [0.75, 0.85]

# ---- Q1: ratio vs absolute eta_down ----
rows = []
pairs = [
    ("ratio3_scale_lo", 0.05, 0.15),   # ratio 3, smaller scale
    ("ratio3_scale_hi", 0.08, 0.24),   # ratio 3, main-text scale (baseline)
    ("ratio3_scale_hi2", 0.12, 0.36),  # ratio 3, larger scale
    ("fixed_ed_lo_eu", 0.04, 0.24),    # same eta_down=0.24, half eta_up -> ratio 6
    ("fixed_ed_hi_eu", 0.16, 0.24),    # same eta_down=0.24, double eta_up -> ratio 1.5
]
for label, eta_up, eta_down in pairs:
    for p_ai in reliabilities:
        for s in range(n_seeds):
            seed = RNG_GLOBAL_SEED + 888 + stable_hash("etapair", label, p_ai, s)
            res = run_one(seed=seed, N=N, T=T, p_ai=p_ai, transparent=True,
                          heuristic="asymmetric", hetero=False,
                          eta_up=eta_up, eta_down=eta_down)
            res.update(label=label, eta_up=eta_up, eta_down=eta_down,
                       ratio=eta_down / eta_up, p_ai=p_ai, seed=s)
            rows.append(res)
df_ratio = pd.DataFrame(rows)
df_ratio.to_csv("results_eta_ratio.csv", index=False)
summary_ratio = df_ratio.groupby(["label", "eta_up", "eta_down", "ratio", "p_ai"])["complementarity"].agg(["mean", "std"]).reset_index()
print("=== Q1: eta_up/eta_down ratio vs absolute value ===")
print(summary_ratio.to_string())

# ---- Q2: extend conf_beta sweep toward uninformative ----
rows2 = []
for conf_beta in [0.4, 0.7, 1.0, 1.5, 2.5]:
    for p_ai in reliabilities:
        for s in range(n_seeds):
            seed = RNG_GLOBAL_SEED + 999 + stable_hash("confbeta_ext", conf_beta, p_ai, s)
            res = run_one(seed=seed, N=N, T=T, p_ai=p_ai, transparent=True,
                          heuristic="bayesian", hetero=False, conf_beta=conf_beta)
            res.update(conf_beta=conf_beta, p_ai=p_ai, seed=s)
            rows2.append(res)
df_beta = pd.DataFrame(rows2)
df_beta.to_csv("results_beta_ext.csv", index=False)
summary_beta = df_beta.groupby(["conf_beta", "p_ai"])["complementarity"].agg(["mean", "std"]).reset_index()
print("\n=== Q2: extended conf_beta sweep (toward uninformative) ===")
print(summary_beta.to_string())

# ---- Q3: formal paired test hetero vs homogeneous, all cells ----
df_hetero = pd.read_csv("results_hetero_ablation.csv")
rows3 = []
for (p_ai, heuristic), g in df_hetero.groupby(["p_ai", "heuristic"]):
    hom = g[g.hetero == False].sort_values("seed")["complementarity"].values
    het = g[g.hetero == True].sort_values("seed")["complementarity"].values
    n = min(len(hom), len(het))
    tstat, pval = stats.ttest_rel(het[:n], hom[:n])
    rows3.append(dict(p_ai=p_ai, heuristic=heuristic, mean_hom=hom.mean(), mean_het=het.mean(),
                       diff=(het[:n] - hom[:n]).mean(), t=tstat, p=pval, n=n))
df_paired = pd.DataFrame(rows3)
df_paired["p_bonferroni"] = np.minimum(df_paired["p"] * len(df_paired), 1.0)
df_paired.to_csv("results_hetero_paired_tests.csv", index=False)
print("\n=== Q3: paired hetero-vs-homogeneous complementarity tests, all cells ===")
print(df_paired.to_string())

print(f"\nelapsed {time.time()-t0:.1f}s")
