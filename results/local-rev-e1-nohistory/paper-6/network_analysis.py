"""
Round-3 review asked (Q2): does the similarity-vs-engagement gap survive a
networked (non-random-pool) topology, or is it an artifact of drawing a fresh
random candidate pool for every agent every round?

This script replaces the per-round random candidate pool with a FIXED
follow-graph: each agent is assigned a static set of `pool_size` neighbors
once at initialization (an Erdos-Renyi-style random directed graph), and the
recommender ranks/filters only among that agent's fixed neighbors every
round, instead of a freshly redrawn random sample. This is the natural
"networked" analogue of the paper's random-candidate-pool design: exposure is
now structurally constrained by a persistent graph, not just policy-filtered
from an unlimited universe each round.

We re-run similarity, engagement, random, and bridging under this fixed
network topology (5 seeds, epsilon=0.35) and permutation-test the
engagement-vs-similarity gap, to see whether the paper's central claim
survives.
"""
import json
import time
import numpy as np

from simulation import cluster_count, recommend

RNG_SEED_BASE = 22345


def build_fixed_network(n, pool_size, seed):
    rng = np.random.default_rng(seed)
    neighbors = np.zeros((n, pool_size), dtype=int)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        neighbors[i] = rng.choice(others, size=pool_size, replace=False)
    return neighbors


def run_networked_simulation(policy, n_agents=200, n_rounds=150, pool_size=30, k_shown=8,
                              epsilon=0.35, mu=0.35, seed=0):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=n_agents)
    neighbors = build_fixed_network(n_agents, pool_size, seed=seed + 99999)

    variance_traj = np.zeros(n_rounds + 1)
    variance_traj[0] = np.var(opinions)

    for t in range(1, n_rounds + 1):
        new_opinions = opinions.copy()
        order = rng.permutation(n_agents)
        for i in order:
            pool_idx = neighbors[i]
            shown_idx = recommend(policy, rng, opinions, i, pool_idx, k_shown)
            shown_ops = opinions[shown_idx]
            dist = np.abs(shown_ops - opinions[i])
            within = dist <= epsilon
            if not np.any(within):
                continue
            influencers = shown_ops[within]
            target = np.mean(influencers)
            new_opinions[i] = opinions[i] + mu * (target - opinions[i])
        opinions = np.clip(new_opinions, -1, 1)
        variance_traj[t] = np.var(opinions)

    return {
        "policy": policy,
        "seed": seed,
        "final_variance": float(variance_traj[-1]),
        "final_extremity": float(np.mean(np.abs(opinions))),
        "n_clusters": cluster_count(opinions),
    }


def permutation_test(a, b, n_perm=100000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:na].mean() - perm[na:].mean()
    p = np.mean(np.abs(diffs) >= np.abs(obs))
    return obs, p


def main():
    t0 = time.time()
    policies = ["random", "similarity", "engagement", "bridging"]
    n_seeds = 5
    results = []
    for policy in policies:
        for s in range(n_seeds):
            r = run_networked_simulation(policy, seed=RNG_SEED_BASE + s)
            results.append(r)
    print(f"networked sim done in {time.time()-t0:.1f}s")

    print(f"{'policy':<12}{'final_var':>14}{'final_extremity':>18}{'n_clusters':>12}")
    for policy in policies:
        rows = [r for r in results if r["policy"] == policy]
        fv = np.array([r["final_variance"] for r in rows])
        fe = np.array([r["final_extremity"] for r in rows])
        nc = np.array([r["n_clusters"] for r in rows])
        print(f"{policy:<12}{fv.mean():>7.4f}+/-{fv.std():<6.4f}{fe.mean():>10.4f}+/-{fe.std():<7.4f}{nc.mean():>6.2f}+/-{nc.std():<5.2f}")

    eng = np.array([r["final_variance"] for r in results if r["policy"] == "engagement"])
    sim = np.array([r["final_variance"] for r in results if r["policy"] == "similarity"])
    rand = np.array([r["final_variance"] for r in results if r["policy"] == "random"])
    obs_es, p_es = permutation_test(eng, sim, n_perm=100000, seed=11)
    obs_er, p_er = permutation_test(eng, rand, n_perm=100000, seed=12)
    print(f"\nengagement - similarity (networked): diff={obs_es:+.4f}, p={p_es:.4f}")
    print(f"engagement - random (networked): diff={obs_er:+.4f}, p={p_er:.4f}")

    with open("network_results.json", "w") as f:
        json.dump({
            "raw": results,
            "engagement_minus_similarity": {"diff": float(obs_es), "p": float(p_es)},
            "engagement_minus_random": {"diff": float(obs_er), "p": float(p_er)},
        }, f, indent=2)
    print(f"\ntotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
