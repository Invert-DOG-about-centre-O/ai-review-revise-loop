"""Quick follow-up: shrink the initial/replenished pool size directly to find the
threshold at which the confidence bound actually starts to bind, holding post_frac=0
(no replenishment) so pool size is the only lever."""
import numpy as np
import json
import time
from sim_robustness import run_simulation, summarize

t0 = time.time()
results = {}
pool_sizes = [900, 100, 20, 8, 5]
cb_values = [0.2, 0.5, 0.8]
n_seeds = 4

for ps in pool_sizes:
    block = {}
    for cb in cb_values:
        seed_summaries = []
        for s in range(n_seeds):
            hist, _ = run_simulation(policy="engagement", confidence_bound=cb,
                                      post_frac=0.0, pool_init_size=ps, seed=s)
            seed_summaries.append(summarize(hist))
        vals_var = [x["variance_final_mean"] for x in seed_summaries]
        vals_near = [x["nearest_rec_dist_final_mean"] for x in seed_summaries]
        block[str(cb)] = {
            "variance_avg": float(np.mean(vals_var)),
            "nearest_dist_avg": float(np.mean(vals_near)),
        }
    results[str(ps)] = block
    print(f"pool_size={ps} done ({time.time()-t0:.1f}s)")

results["total_runtime_sec"] = time.time() - t0
with open("results_sparsity.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved")
