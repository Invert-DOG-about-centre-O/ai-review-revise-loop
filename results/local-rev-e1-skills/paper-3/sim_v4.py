"""
Round-3 revision experiments addressing round-3 review questions:
(G) Network-topology sign-flip check (R3-Q2): the v3 network-topology grid used
    20 seeds/cell and showed the (already fragile) engagement-vs-random gap
    flip sign at eps in {0.10, 0.20}. Re-run those two eps values at 80
    seeds/cell (4x) to test whether this is a real, stable topology-induced
    reversal or more sampling noise, same diagnostic used in v3 for (D)/(E).
(H) Integration-rule robustness (R3-Q3): does the frozen-update backfire
    mechanism depend on simple-mean integration over in-tolerance peers?
    Re-run the main grid (lambda in {-1,0,1} x eps) with a similarity-weighted
    (Gaussian-kernel) integration rule instead of an unweighted mean over the
    in-tolerance mask, holding the confidence filter (who counts as
    "in-window") identical. Tests whether weighting integration by closeness
    rescues diversity-boosted content or the backfire persists regardless.
"""
import numpy as np
import json, time

t0 = time.time()


def run_once_network(lam, eps, seed, N=150, K=5, beta=6.0, mu=0.5, T=80, mean_deg=20, track_frozen=False):
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


def run_once_weighted(lam, eps, seed, N=150, K=5, beta=6.0, mu=0.5, T=80, track_frozen=False):
    """Same curation/confidence-filter as sim.py, but integration among
    in-tolerance peers uses a Gaussian-kernel similarity weight instead of a
    plain mean, so closer in-tolerance peers pull harder."""
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
        edist = np.abs(exposed - own)
        mask = edist <= eps
        has_any = mask.any(axis=1)
        if track_frozen:
            frozen_frac.append(1.0 - has_any.mean())
        kernel = np.exp(-(edist / max(eps, 1e-9)) ** 2) * mask
        wsum = kernel.sum(axis=1)
        weighted_mean = np.where(has_any, (kernel * exposed).sum(axis=1) / np.maximum(wsum, 1e-9), opinions)
        new_opinions = opinions + mu * has_any * (weighted_mean - opinions)
        opinions = np.clip(new_opinions, -1, 1)
    out = {"final_var": float(opinions.var())}
    if track_frozen:
        out["frozen_frac"] = float(np.mean(frozen_frac))
    return out


# ---------- (G) network-topology sign-flip re-check at 80 seeds ----------
t1 = time.time()
LAMBDAS_G = [-1.0, 0.0, 1.0]
EPS_G = [0.10, 0.20]
SEEDS80 = list(range(80))
resG = []
for eps in EPS_G:
    for lam in LAMBDAS_G:
        for seed in SEEDS80:
            r = run_once_network(lam, eps, seed, track_frozen=(lam == -1.0))
            r.update({"lambda": lam, "epsilon": eps, "seed": seed})
            resG.append(r)
tG = time.time() - t1
print(f"(G) done in {tG:.1f}s, {len(resG)} runs")

# ---------- (H) similarity-weighted integration rule ----------
t2 = time.time()
LAMBDAS_H = [-1.0, 0.0, 1.0]
EPS_H = [0.10, 0.20, 0.30]
SEEDS_H = list(range(30))
resH = []
for eps in EPS_H:
    for lam in LAMBDAS_H:
        for seed in SEEDS_H:
            r = run_once_weighted(lam, eps, seed, track_frozen=(lam == -1.0))
            r.update({"lambda": lam, "epsilon": eps, "seed": seed})
            resH.append(r)
tH = time.time() - t2
print(f"(H) done in {tH:.1f}s, {len(resH)} runs")

elapsed = time.time() - t0
print(f"Total v4 sim time: {elapsed:.1f}s")

with open("sim_results_v4.json", "w") as f:
    json.dump({
        "network_topology_80seeds": resG,
        "weighted_integration": resH,
        "elapsed_sec": elapsed,
    }, f, indent=2)
print("Saved sim_results_v4.json")
