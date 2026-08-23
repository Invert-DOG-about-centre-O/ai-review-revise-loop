import time
import numpy as np
from robustness_analysis import train, test, train_policy, evaluate

SEEDS = list(range(5))
t0 = time.time()
for scale in [0.2, 0.4, 0.6, 0.8]:
    rows = []
    for sd in SEEDS:
        params = train_policy(train, aggregate="mean", lam_penalty=0.0, precommit_q=0.0,
                               n_outliers=0, agree_scale=scale, seed=sd)
        rows.append(evaluate(params, test))
    syco = np.array([r["sycophancy_ambiguous"] for r in rows])
    print(f"[agree_scale={scale}] syco_ambig={syco.mean():.3f}+/-{syco.std():.3f}")
print(f"wall time {time.time()-t0:.2f}s")
