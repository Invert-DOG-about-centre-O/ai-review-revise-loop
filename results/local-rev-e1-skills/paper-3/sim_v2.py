"""
Revision experiments addressing round-1 review:
(A) Frozen-fraction diagnostic computed on the EXACT lambda=-1 x epsilon grid
    (30 seeds) used for the headline +14.5% / 4.3 result, instead of the
    separate robustness sweep -- closes the "different experiment" gap.
(B) Sensitivity of the diversity-boosting backfire to curation sharpness
    BETA and update rate MU (the two fixed hyperparameters the review flagged).
(C) Sensitivity to population size N (50, 150, 500).
All at epsilon=0.20 (a representative mid-grid value) unless noted.
"""
import numpy as np
import json, time

t0 = time.time()


def run_once(lam, eps, seed, N=150, K=5, beta=6.0, mu=0.5, T=80, track_frozen=False):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=N)
    frozen_frac = []
    for _ in range(T):
        diff = np.abs(opinions[:, None] - opinions[None, :])
        logw = -beta * lam * diff
        np.fill_diagonal(logw, -np.inf)
        gumbel = rng.gumbel(size=(N, N))
        scores = logw + gumbel
        topk = np.argpartition(-scores, K, axis=1)[:, :K]
        exposed = opinions[topk]
        own = opinions[:, None]
        mask = np.abs(exposed - own) <= eps
        has_any = mask.any(axis=1)
        if track_frozen:
            frozen_frac.append(1.0 - has_any.mean())
        masked_sum = np.where(mask, exposed, 0.0).sum(axis=1)
        masked_cnt = mask.sum(axis=1)
        mean_exposed = np.where(has_any, masked_sum / np.maximum(masked_cnt, 1), opinions)
        new_opinions = opinions + mu * has_any * (mean_exposed - opinions)
        opinions = np.clip(new_opinions, -1, 1)
    out = {"final_var": float(opinions.var())}
    if track_frozen:
        out["frozen_frac"] = float(np.mean(frozen_frac))
    return out


# ---------- (A) frozen fraction on the exact main diversity-boosting grid ----------
EPS_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
SEEDS30 = list(range(30))
resA = []
for eps in EPS_GRID:
    for seed in SEEDS30:
        r = run_once(-1.0, eps, seed, track_frozen=True)
        r.update({"lambda": -1.0, "epsilon": eps, "seed": seed})
        resA.append(r)
tA = time.time() - t0
print(f"(A) done in {tA:.1f}s, {len(resA)} runs")

# ---------- (B) sensitivity to BETA and MU at eps=0.20 ----------
t1 = time.time()
BETAS = [3.0, 6.0, 10.0]
MUS = [0.25, 0.5, 0.75]
LAMBDAS_B = [-1.0, 0.0, 1.0]
SEEDS15 = list(range(15))
resB = []
for beta in BETAS:
    for mu in MUS:
        for lam in LAMBDAS_B:
            for seed in SEEDS15:
                r = run_once(lam, 0.20, seed, beta=beta, mu=mu, track_frozen=(lam == -1.0))
                r.update({"lambda": lam, "beta": beta, "mu": mu, "seed": seed})
                resB.append(r)
tB = time.time() - t1
print(f"(B) done in {tB:.1f}s, {len(resB)} runs")

# ---------- (C) sensitivity to N at eps=0.20 ----------
t2 = time.time()
NS = [50, 150, 500]
LAMBDAS_C = [-1.0, 0.0, 1.0]
SEEDS10 = list(range(10))
resC = []
for N in NS:
    for lam in LAMBDAS_C:
        for seed in SEEDS10:
            r = run_once(lam, 0.20, seed, N=N)
            r.update({"lambda": lam, "N": N, "seed": seed})
            resC.append(r)
tC = time.time() - t2
print(f"(C) done in {tC:.1f}s, {len(resC)} runs")

elapsed = time.time() - t0
print(f"Total v2 sim time: {elapsed:.1f}s")

with open("sim_results_v2.json", "w") as f:
    json.dump({
        "frozen_on_main_grid": resA,
        "beta_mu_sensitivity": resB,
        "N_sensitivity": resC,
        "elapsed_sec": elapsed,
    }, f, indent=2)
print("Saved sim_results_v2.json")
