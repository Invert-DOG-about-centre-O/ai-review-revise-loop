"""
Sensitivity ablation: does the complementarity regime boundary (~0.75-0.85 AI
reliability) depend on the arbitrary asymmetric-heuristic eta parameters or the
confidence-signal shape (conf_beta), or is it a structural property of the setup?
Varies eta_down and conf_beta one at a time around the main-text defaults
(eta_up=0.08, eta_down=0.24, conf_beta=1.5), transparent condition only,
reliabilities 0.75 and 0.85 (the flip region), 20 seeds/cell.
"""
import numpy as np
import pandas as pd
from scipy import stats
import time
from sim import run_one, stable_hash, RNG_GLOBAL_SEED

t0 = time.time()
N, T, n_seeds = 200, 300, 20
reliabilities = [0.75, 0.85]

rows = []
# eta_down sweep (asymmetric heuristic), eta_up fixed at 0.08
for eta_down in [0.16, 0.24, 0.32]:
    for p_ai in reliabilities:
        for s in range(n_seeds):
            seed = RNG_GLOBAL_SEED + 555 + stable_hash("etadown", eta_down, p_ai, s)
            res = run_one(seed=seed, N=N, T=T, p_ai=p_ai, transparent=True,
                          heuristic="asymmetric", hetero=False, eta_down=eta_down)
            res.update(sweep="eta_down", value=eta_down, p_ai=p_ai, seed=s)
            rows.append(res)

# conf_beta sweep (bayesian heuristic, the one driving the residual positive
# complementarity near the flip point)
for conf_beta in [1.0, 1.5, 2.5]:
    for p_ai in reliabilities:
        for s in range(n_seeds):
            seed = RNG_GLOBAL_SEED + 666 + stable_hash("confbeta", conf_beta, p_ai, s)
            res = run_one(seed=seed, N=N, T=T, p_ai=p_ai, transparent=True,
                          heuristic="bayesian", hetero=False, conf_beta=conf_beta)
            res.update(sweep="conf_beta", value=conf_beta, p_ai=p_ai, seed=s)
            rows.append(res)

df = pd.DataFrame(rows)
df.to_csv("results_sensitivity.csv", index=False)

summary_rows = []
for (sweep, value, p_ai), g in df.groupby(["sweep", "value", "p_ai"]):
    tstat, pval = stats.ttest_1samp(g["complementarity"].values, 0.0)
    summary_rows.append(dict(sweep=sweep, value=value, p_ai=p_ai,
                              mean_complementarity=g["complementarity"].mean(),
                              t=tstat, p=pval, n=len(g)))
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("results_sensitivity_summary.csv", index=False)
print(df_summary.to_string())
print(f"elapsed {time.time()-t0:.1f}s")
