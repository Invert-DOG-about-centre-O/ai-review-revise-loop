"""
Additional analyses requested by peer review of v1:
  1. Permutation significance tests on the main-experiment policy comparisons
     (final variance), since only 10 seeds/policy were used.
  2. A sweep over the engagement score's extremity weight w (score =
     (1-w)*agreement + w*extremity), to check whether the engagement>similarity
     variance gap is an artifact of the specific 0.7/0.3 split.

Reuses select_candidates/run_simulation machinery from simulation.py.
"""
import json
import time
import numpy as np

from simulation import select_candidates, cluster_count, RNG_SEED_BASE


def recommend_weighted(policy, rng, opinions, agent_idx, pool_idx, k, w=0.3):
    own = opinions[agent_idx]
    pool_ops = opinions[pool_idx]
    if policy == "engagement_w":
        dist = np.abs(pool_ops - own)
        agreement_score = 1.0 - dist
        extremity_score = np.abs(pool_ops)
        score = (1 - w) * agreement_score + w * extremity_score
        order = np.argsort(-score)
        return pool_idx[order[:k]]
    raise ValueError(policy)


def run_weighted(w, n_agents=200, n_rounds=150, pool_size=30, k_shown=8,
                  epsilon=0.35, mu=0.35, seed=0):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=n_agents)
    for t in range(1, n_rounds + 1):
        new_opinions = opinions.copy()
        order = rng.permutation(n_agents)
        for i in order:
            pool_idx = select_candidates(rng, opinions, i, pool_size)
            shown_idx = recommend_weighted("engagement_w", rng, opinions, i, pool_idx, k_shown, w=w)
            shown_ops = opinions[shown_idx]
            dist = np.abs(shown_ops - opinions[i])
            within = dist <= epsilon
            if not np.any(within):
                continue
            influencers = shown_ops[within]
            target = np.mean(influencers)
            new_opinions[i] = opinions[i] + mu * (target - opinions[i])
        new_opinions = np.clip(new_opinions, -1, 1)
        opinions = new_opinions
    return {
        "w": w,
        "seed": seed,
        "final_variance": float(np.var(opinions)),
        "final_extremity": float(np.mean(np.abs(opinions))),
    }


def permutation_test(a, b, n_perm=100000, seed=0):
    """Two-sided permutation test on difference of means; a,b are 1D arrays."""
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

    # --- 1. Permutation tests on main-experiment final variance ---
    with open("results.json") as f:
        results = json.load(f)
    policies = ["random", "similarity", "engagement", "bridging"]
    fv = {p: np.array([r["final_variance"] for r in results if r["policy"] == p]) for p in policies}

    pairs = [("engagement", "similarity"), ("bridging", "similarity"),
             ("engagement", "random"), ("similarity", "random"),
             ("bridging", "engagement")]
    sig_lines = ["pair, mean_diff, p_value (permutation, n=100000)"]
    sig_results = []
    for pa, pb in pairs:
        obs, p = permutation_test(fv[pa], fv[pb], n_perm=100000, seed=1)
        sig_lines.append(f"{pa} - {pb}: diff={obs:+.4f}, p={p:.4f}")
        sig_results.append({"a": pa, "b": pb, "diff": obs, "p": p})
    print("\n".join(sig_lines))
    print(f"[{time.time()-t0:.1f}s elapsed]")

    with open("significance_results.json", "w") as f:
        json.dump(sig_results, f, indent=2)

    # --- 2. Extremity-weight sweep for the engagement score ---
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_seeds_w = 5
    weight_results = []
    for w in weights:
        for s in range(n_seeds_w):
            r = run_weighted(w, seed=RNG_SEED_BASE + 200 + s)
            weight_results.append(r)
        vs = np.array([r["final_variance"] for r in weight_results if r["w"] == w])
        print(f"w={w:.1f}: final_var={vs.mean():.4f}+/-{vs.std():.4f}  [{time.time()-t0:.1f}s elapsed]")

    with open("weight_sweep_results.json", "w") as f:
        json.dump(weight_results, f, indent=2)

    # similarity is equivalent to w=0 with pure argsort-by-distance; but our
    # weighted score at w=0 reduces to ranking by agreement only, which for a
    # monotonic transform of distance is the same ordering as pure similarity.
    # We also permutation-test w=0.3 (paper's engagement) vs w=0.0 (~similarity).
    v03 = np.array([r["final_variance"] for r in weight_results if r["w"] == 0.3])
    v00 = np.array([r["final_variance"] for r in weight_results if r["w"] == 0.0])
    obs, p = permutation_test(v03, v00, n_perm=100000, seed=2)
    print(f"\nw=0.3 vs w=0.0: diff={obs:+.4f}, p={p:.4f}")

    print(f"\ntotal extra_analysis time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
