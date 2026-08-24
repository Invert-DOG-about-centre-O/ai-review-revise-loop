"""
Robustness check: does the qualitative pattern (engagement-max > random,
diversity-boost > engagement-max, in final opinion variance) survive
changing K (exposures/round), the one structural parameter held fixed
throughout the main sweep? Also logs the fraction of (agent, round)
updates that are "frozen" (no exposed peer within epsilon) to verify the
proposed mechanism for the diversity-boosting result.
"""
import numpy as np
import json, time

t0 = time.time()
N = 150
BETA = 6.0
MU = 0.5
T = 80
SEEDS = list(range(20))
EPSILONS = [0.10, 0.20, 0.30]
LAMBDAS = [-1.0, 0.0, 1.0]
KS = [3, 5, 10]


def run_once(lam, eps, seed, K):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=N)
    frozen_frac = []
    for _ in range(T):
        diff = np.abs(opinions[:, None] - opinions[None, :])
        logw = -BETA * lam * diff
        np.fill_diagonal(logw, -np.inf)
        gumbel = rng.gumbel(size=(N, N))
        scores = logw + gumbel
        topk = np.argpartition(-scores, K, axis=1)[:, :K]
        exposed = opinions[topk]
        own = opinions[:, None]
        mask = np.abs(exposed - own) <= eps
        has_any = mask.any(axis=1)
        frozen_frac.append(1.0 - has_any.mean())
        masked_sum = np.where(mask, exposed, 0.0).sum(axis=1)
        masked_cnt = mask.sum(axis=1)
        mean_exposed = np.where(has_any, masked_sum / np.maximum(masked_cnt, 1), opinions)
        new_opinions = opinions + MU * has_any * (mean_exposed - opinions)
        opinions = np.clip(new_opinions, -1, 1)
    return {"final_var": float(opinions.var()), "frozen_frac": float(np.mean(frozen_frac))}


results = []
for K in KS:
    for lam in LAMBDAS:
        for eps in EPSILONS:
            for seed in SEEDS:
                r = run_once(lam, eps, seed, K)
                r.update({"lambda": lam, "epsilon": eps, "seed": seed, "K": K})
                results.append(r)

elapsed = time.time() - t0
print(f"Robustness sim done in {elapsed:.1f}s, {len(results)} runs")
with open("sim_results_robustness.json", "w") as f:
    json.dump({"results": results, "params": {
        "N": N, "BETA": BETA, "MU": MU, "T": T, "SEEDS": SEEDS,
        "EPSILONS": EPSILONS, "LAMBDAS": LAMBDAS, "KS": KS, "elapsed_sec": elapsed}}, f, indent=2)
print("Saved sim_results_robustness.json")
