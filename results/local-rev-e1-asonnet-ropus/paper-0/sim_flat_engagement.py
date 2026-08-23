"""Follow-up to reviewer Q3: does the extremism ranking (engagement-max worst, diversity
best) survive when engagement probability is nearly FLAT in distance (weak coupling
between engagement and distance), as an extreme test alongside the proximity and
novelty-seeking shapes already tested?"""
import numpy as np
import json
import time
from sim_robustness import run_simulation, summarize

t0 = time.time()
results = {}
for pol in ["random", "engagement", "diversity"]:
    vals_var, vals_ext = [], []
    for s in range(5):
        hist, _ = run_simulation(policy=pol, seed=s, engagement_sigma=5.0)
        summ = summarize(hist)
        vals_var.append(summ["variance_final_mean"])
        vals_ext.append(summ["extremism_final_mean"])
    results[pol] = {"variance_avg": float(np.mean(vals_var)), "variance_std": float(np.std(vals_var)),
                     "extremism_avg": float(np.mean(vals_ext)), "extremism_std": float(np.std(vals_ext))}
    print(f"{pol}: done ({time.time()-t0:.1f}s)")
results["total_runtime_sec"] = time.time() - t0
with open("results_flat_engagement.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved")
