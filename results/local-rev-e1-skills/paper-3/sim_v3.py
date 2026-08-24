"""
Revision-2 experiments addressing round-2 review:
(D) Re-run the beta x mu sensitivity grid with MORE seeds (40 vs 15) to check
    whether the engagement<random sign flip reported in v2 is a real small
    effect or sampling noise at the smaller seed count.
(E) Re-run the N sensitivity grid with MORE seeds (30 vs 10), same question.
(F) NEW: network-topology check. Instead of a fully-connected population,
    agents sit on an Erdos-Renyi random graph (mean degree ~20) and curation
    can only expose peers who are graph-neighbors (falling back to all
    neighbors if fewer than K exist). Tests whether the diversity-boosting
    backfire mechanism survives when exposure is graph-mediated, not global.
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


def run_once_network(lam, eps, seed, N=150, K=5, beta=6.0, mu=0.5, T=80, mean_deg=20, track_frozen=False):
    """Curation restricted to Erdos-Renyi graph neighbors (fixed graph, drawn once per run)."""
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=N)
    p_edge = mean_deg / (N - 1)
    adj = rng.random((N, N)) < p_edge
    adj = np.triu(adj, 1)
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    non_neighbor_penalty = np.where(adj, 0.0, -1e9)
    frozen_frac = []
    for _ in range(T):
        diff = np.abs(opinions[:, None] - opinions[None, :])
        logw = -beta * lam * diff + non_neighbor_penalty
        np.fill_diagonal(logw, -np.inf)
        gumbel = rng.gumbel(size=(N, N))
        scores = logw + gumbel
        topk = np.argpartition(-scores, K, axis=1)[:, :K]
        exposed = opinions[topk]
        valid = np.take_along_axis(adj, topk, axis=1)
        own = opinions[:, None]
        mask = (np.abs(exposed - own) <= eps) & valid
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


# ---------- (D) beta x mu with 40 seeds instead of 15 ----------
t1 = time.time()
BETAS = [3.0, 6.0, 10.0]
MUS = [0.25, 0.5, 0.75]
LAMBDAS_B = [-1.0, 0.0, 1.0]
SEEDS40 = list(range(40))
resD = []
for beta in BETAS:
    for mu in MUS:
        for lam in LAMBDAS_B:
            for seed in SEEDS40:
                r = run_once(lam, 0.20, seed, beta=beta, mu=mu, track_frozen=(lam == -1.0))
                r.update({"lambda": lam, "beta": beta, "mu": mu, "seed": seed})
                resD.append(r)
tD = time.time() - t1
print(f"(D) done in {tD:.1f}s, {len(resD)} runs")

# ---------- (E) N sensitivity with 30 seeds instead of 10 ----------
t2 = time.time()
NS = [50, 150, 500]
LAMBDAS_C = [-1.0, 0.0, 1.0]
SEEDS30 = list(range(30))
resE = []
for N in NS:
    for lam in LAMBDAS_C:
        for seed in SEEDS30:
            r = run_once(lam, 0.20, seed, N=N)
            r.update({"lambda": lam, "N": N, "seed": seed})
            resE.append(r)
tE = time.time() - t2
print(f"(E) done in {tE:.1f}s, {len(resE)} runs")

# ---------- (F) network-topology check ----------
t3 = time.time()
LAMBDAS_F = [-1.0, 0.0, 1.0]
EPS_F = [0.10, 0.20, 0.30]
SEEDS_F = list(range(20))
resF = []
for eps in EPS_F:
    for lam in LAMBDAS_F:
        for seed in SEEDS_F:
            r = run_once_network(lam, eps, seed, track_frozen=(lam == -1.0))
            r.update({"lambda": lam, "epsilon": eps, "seed": seed})
            resF.append(r)
tF = time.time() - t3
print(f"(F) done in {tF:.1f}s, {len(resF)} runs")

elapsed = time.time() - t0
print(f"Total v3 sim time: {elapsed:.1f}s")

with open("sim_results_v3.json", "w") as f:
    json.dump({
        "beta_mu_sensitivity_40seeds": resD,
        "N_sensitivity_30seeds": resE,
        "network_topology": resF,
        "elapsed_sec": elapsed,
    }, f, indent=2)
print("Saved sim_results_v3.json")
