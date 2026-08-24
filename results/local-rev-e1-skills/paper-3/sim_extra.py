"""
Extension: 'diversity-boosting' curation policy, i.e. negative lambda.
Same weight formula w_ij = exp(-BETA * lambda * |opinion_i - opinion_j|):
lambda < 0 up-weights DISSIMILAR peers (an explicit anti-echo-chamber
recommender policy), as a mitigation baseline to compare against the
engagement-optimized (lambda>0) and random (lambda=0) conditions already
run in sim.py. Reuses identical N, K, BETA, MU, T, epsilon grid, seeds.
"""
import numpy as np
import json, time

t0 = time.time()

N = 150
K = 5
BETA = 6.0
MU = 0.5
T = 80
SEEDS = list(range(30))

LAMBDAS = [-1.0, -0.6, -0.2]
EPSILONS = [0.10, 0.15, 0.20, 0.25, 0.30]


def run_once(lam, eps, seed):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=N)
    exposure_vars = []
    for _ in range(T):
        diff = np.abs(opinions[:, None] - opinions[None, :])
        logw = -BETA * lam * diff
        np.fill_diagonal(logw, -np.inf)
        gumbel = rng.gumbel(size=(N, N))
        scores = logw + gumbel
        topk = np.argpartition(-scores, K, axis=1)[:, :K]
        exposed = opinions[topk]
        exposure_vars.append(exposed.var(axis=1).mean())
        own = opinions[:, None]
        mask = np.abs(exposed - own) <= eps
        has_any = mask.any(axis=1)
        masked_sum = np.where(mask, exposed, 0.0).sum(axis=1)
        masked_cnt = mask.sum(axis=1)
        mean_exposed = np.where(has_any, masked_sum / np.maximum(masked_cnt, 1), opinions)
        new_opinions = opinions + MU * has_any * (mean_exposed - opinions)
        opinions = np.clip(new_opinions, -1, 1)
    return {"final_var": float(opinions.var()), "mean_exposure_var": float(np.mean(exposure_vars))}


results = []
for lam in LAMBDAS:
    for eps in EPSILONS:
        for seed in SEEDS:
            r = run_once(lam, eps, seed)
            r.update({"lambda": lam, "epsilon": eps, "seed": seed})
            results.append(r)

elapsed = time.time() - t0
print(f"Extra sim done in {elapsed:.1f}s, {len(results)} runs")

with open("sim_results_extra.json", "w") as f:
    json.dump({"results": results, "params": {
        "N": N, "K": K, "BETA": BETA, "MU": MU, "T": T,
        "LAMBDAS": LAMBDAS, "EPSILONS": EPSILONS, "SEEDS": SEEDS,
        "elapsed_sec": elapsed,
    }}, f, indent=2)
print("Saved sim_results_extra.json")
