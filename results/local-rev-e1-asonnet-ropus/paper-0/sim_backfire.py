"""Follow-up to reviewer Q2: at pool sizes where the confidence bound becomes causally
active (20, 8; see results_sparsity.json), is the sign of "wider tolerance -> lower
variance" sensitive to backfire strength? Re-run with a much stronger backfire
(backfire_prob 0.15->0.45, backfire_rate 0.10->0.35) alongside the original baseline."""
import numpy as np
import json
import time
from sim_robustness import run_simulation, summarize

t0 = time.time()
results = {}
pool_sizes = [20, 8]
cb_values = [0.2, 0.5, 0.8]
n_seeds = 4

backfire_settings = {
    "baseline": dict(backfire_prob=0.15, backfire_rate=0.10),
    "strong_backfire": dict(backfire_prob=0.45, backfire_rate=0.35),
}

for label, bf in backfire_settings.items():
    for ps in pool_sizes:
        block = {}
        for cb in cb_values:
            vals_var = []
            for s in range(n_seeds):
                hist, _ = run_simulation(policy="engagement", confidence_bound=cb,
                                          post_frac=0.0, pool_init_size=ps, seed=s, **bf)
                vals_var.append(summarize(hist)["variance_final_mean"])
            block[str(cb)] = {"variance_avg": float(np.mean(vals_var)),
                               "variance_std": float(np.std(vals_var))}
        results[f"{label}_pool{ps}"] = block
        print(f"{label} pool_size={ps} done ({time.time()-t0:.1f}s)")

results["total_runtime_sec"] = time.time() - t0
with open("results_backfire.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved")
