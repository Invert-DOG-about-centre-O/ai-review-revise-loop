"""
Engagement-optimized vs. random-exposure recommendation curation
in a bounded-confidence opinion-dynamics model.

Each agent has a continuous opinion in [-1, 1]. Each round, every agent is
exposed to K "recommended" peer opinions drawn (without replacement, via a
Gumbel-top-k / Plackett-Luce weighted sampler) from the full population.
The sampling weight for peer j as seen by agent i is

    w_ij = exp(-beta * lambda * |opinion_i - opinion_j|)

lambda = 0  -> uniform random exposure (baseline recommender / no curation)
lambda = 1  -> fully engagement-optimized exposure (peers are strongly
               biased toward agreement, mimicking similarity-maximizing
               recommender systems)

K (number of exposures/round) is held FIXED across all lambda, so any
difference in outcomes cannot be explained by agents simply seeing more or
fewer peers -- only by WHO they are shown (the exposure-diversity channel).

After exposure, agents apply a standard Hegselmann-Krause bounded-confidence
update: average over exposed opinions within confidence radius epsilon,
move a fraction mu of the way there.

We sweep lambda x epsilon on a grid, with 30 seeds per cell, and report:
  - final opinion variance (polarization proxy; continuous, no thresholds)
  - mean exposure variance during the run (diversity-of-exposure mediator)
  - paired tests (lambda=1 vs lambda=0, same seed & epsilon) for the
    engagement-vs-random effect
"""
import numpy as np
from scipy import stats
import json, time

t0 = time.time()

N = 150          # agents
K = 5             # exposures per agent per round (FIXED across conditions)
BETA = 6.0        # curation sharpness
MU = 0.5          # update rate toward exposed mean
T = 80             # rounds
SEEDS = list(range(30))

LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.10, 0.15, 0.20, 0.25, 0.30]


def run_once(lam, eps, seed):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=N)
    exposure_vars = []

    for _ in range(T):
        diff = np.abs(opinions[:, None] - opinions[None, :])  # N x N
        logw = -BETA * lam * diff
        np.fill_diagonal(logw, -np.inf)
        gumbel = rng.gumbel(size=(N, N))
        scores = logw + gumbel
        topk = np.argpartition(-scores, K, axis=1)[:, :K]  # N x K indices
        exposed = opinions[topk]  # N x K

        exposure_vars.append(exposed.var(axis=1).mean())

        own = opinions[:, None]
        mask = np.abs(exposed - own) <= eps
        has_any = mask.any(axis=1)
        # mean of masked exposures (avoid div0 by using where)
        masked_sum = np.where(mask, exposed, 0.0).sum(axis=1)
        masked_cnt = mask.sum(axis=1)
        mean_exposed = np.where(has_any, masked_sum / np.maximum(masked_cnt, 1), opinions)

        new_opinions = opinions + MU * has_any * (mean_exposed - opinions)
        opinions = np.clip(new_opinions, -1, 1)

    return {
        "final_var": float(opinions.var()),
        "mean_exposure_var": float(np.mean(exposure_vars)),
    }


results = []
for lam in LAMBDAS:
    for eps in EPSILONS:
        for seed in SEEDS:
            r = run_once(lam, eps, seed)
            r.update({"lambda": lam, "epsilon": eps, "seed": seed})
            results.append(r)

elapsed = time.time() - t0
print(f"Simulation done in {elapsed:.1f}s, {len(results)} runs")

with open("sim_results.json", "w") as f:
    json.dump({"results": results, "params": {
        "N": N, "K": K, "BETA": BETA, "MU": MU, "T": T,
        "LAMBDAS": LAMBDAS, "EPSILONS": EPSILONS, "SEEDS": SEEDS,
        "elapsed_sec": elapsed,
    }}, f, indent=2)

print("Saved sim_results.json")
