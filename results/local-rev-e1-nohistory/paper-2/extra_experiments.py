"""
Revision experiments addressing round-1 review:
  (A) Full pairwise Welch t-tests (Bonferroni-corrected) across all main-sweep
      conditions and all three metrics, using per-seed final values (not just
      the single syco(1.0)-vs-calibrated test in v1).
  (B) A pool-sparsity / decoupled-corpus experiment testing whether the
      "freezing" effect is a near-tautological consequence of nearest-neighbor
      selection in a dense, self-referential pool (reviewer weakness #1 /
      question #1): we vary pool size K (subsampled from current opinions)
      and also test a fixed EXTERNAL corpus decoupled from agent opinions.
"""
import json
import time
import numpy as np
from scipy import stats

RNG_SEED_BASE = 12345
N_AGENTS = 300
N_ROUNDS = 150
CONFIDENCE_EPS = 0.35
LEARNING_RATE = 0.3
BACKFIRE_RATE = 0.05
N_SEEDS = 20


def bimodality_coefficient(x):
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    skew = np.mean(((x - m) / s) ** 3)
    kurt = np.mean(((x - m) / s) ** 4) - 3.0
    return (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def init_opinions(rng, n):
    return np.clip(rng.normal(0.0, 0.25, size=n), -1, 1)


def recommend(strategy, opinions, pool, rng, param):
    n = len(opinions)
    if strategy in ("random", "calibrated"):
        return rng.choice(pool, size=n, replace=True)
    if strategy == "sycophantic":
        s = param
        out = np.empty(n)
        use_syco = rng.random(n) < s
        for i in range(n):
            if use_syco[i]:
                idx = np.argmin(np.abs(pool - opinions[i]))
                out[i] = pool[idx]
            else:
                out[i] = rng.choice(pool)
        return out
    if strategy == "bridging":
        b = param
        out = np.empty(n)
        use_bridge = rng.random(n) < b
        for i in range(n):
            if use_bridge[i]:
                idx = np.argmin(np.abs(pool - (-opinions[i])))
                out[i] = pool[idx]
            else:
                out[i] = rng.choice(pool)
        return out
    raise ValueError(strategy)


def run_simulation(strategy, param, seed, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                    pool_size=None, fixed_external_pool=False):
    """pool_size: if set, subsample this many opinions (without replacement,
    with fallback to replacement if pool_size > n_agents) each round as the
    content pool instead of using all N agents.
    fixed_external_pool: if True, the pool is drawn ONCE at t=0 from the same
    initial distribution and held fixed thereafter, decoupled from the
    agents' evolving opinions (a "content corpus" rather than peer content).
    """
    rng = np.random.default_rng(seed)
    opinions = init_opinions(rng, n_agents)
    ext_pool = init_opinions(rng, pool_size or n_agents) if fixed_external_pool else None
    for t in range(n_rounds):
        if fixed_external_pool:
            pool = ext_pool
        elif pool_size is not None and pool_size < n_agents:
            idx = rng.choice(n_agents, size=pool_size, replace=False)
            pool = opinions[idx]
        else:
            pool = opinions.copy()
        rec = recommend(strategy, opinions, pool, rng, param)
        diff = rec - opinions
        within = np.abs(diff) <= CONFIDENCE_EPS
        delta = np.where(within, LEARNING_RATE * diff,
                          -BACKFIRE_RATE * np.sign(diff) * np.abs(diff))
        opinions = np.clip(opinions + delta, -1, 1)
    return {
        "final_var": float(np.var(opinions)),
        "final_bc": float(bimodality_coefficient(opinions)),
        "final_extremity": float(np.mean(np.abs(opinions))),
    }


def part_a_pairwise_tests():
    conditions = [
        ("random", 0.0), ("calibrated", 0.0), ("bridging", 0.5),
        ("sycophantic", 0.0), ("sycophantic", 0.25), ("sycophantic", 0.5),
        ("sycophantic", 0.75), ("sycophantic", 1.0),
    ]
    per_seed = {}
    for strategy, param in conditions:
        key = f"{strategy}_{param}"
        runs = [run_simulation(strategy, param, seed=RNG_SEED_BASE + s) for s in range(N_SEEDS)]
        per_seed[key] = {
            "var": np.array([r["final_var"] for r in runs]),
            "bc": np.array([r["final_bc"] for r in runs]),
            "extremity": np.array([r["final_extremity"] for r in runs]),
        }
    baseline = "calibrated_0.0"
    others = [k for k in per_seed if k != baseline and k != "random_0.0"]
    metrics = ["var", "bc", "extremity"]
    n_tests = len(others) * len(metrics)
    alpha_bonf = 0.05 / n_tests
    tests = []
    for k in others:
        for m in metrics:
            t, p = stats.ttest_ind(per_seed[k][m], per_seed[baseline][m], equal_var=False)
            tests.append({"condition": k, "metric": m, "t": float(t), "p": float(p),
                           "sig_bonferroni": bool(p < alpha_bonf)})
    return tests, n_tests, alpha_bonf


def part_b_pool_experiments():
    pool_sizes = [5, 20, 50, 150, 300]
    out = {"subsampled_pool": {}, "fixed_external_pool": {}}
    n_seeds_b = 10
    for K in pool_sizes:
        for strategy, param in [("calibrated", 0.0), ("sycophantic", 1.0)]:
            key = f"{strategy}_{param}_K{K}"
            runs = [run_simulation(strategy, param, seed=RNG_SEED_BASE + s, pool_size=K)
                    for s in range(n_seeds_b)]
            v = np.array([r["final_var"] for r in runs])
            out["subsampled_pool"][key] = {"var_mean": float(v.mean()), "var_std": float(v.std(ddof=1))}
    # fixed external corpus, decoupled from agent opinions entirely
    for strategy, param in [("calibrated", 0.0), ("sycophantic", 1.0)]:
        key = f"{strategy}_{param}_external"
        runs = [run_simulation(strategy, param, seed=RNG_SEED_BASE + s, pool_size=300,
                                fixed_external_pool=True) for s in range(n_seeds_b)]
        v = np.array([r["final_var"] for r in runs])
        out["fixed_external_pool"][key] = {"var_mean": float(v.mean()), "var_std": float(v.std(ddof=1))}
    return out


def main():
    t0 = time.time()
    tests, n_tests, alpha_bonf = part_a_pairwise_tests()
    print(f"Part A: {n_tests} pairwise Welch tests vs calibrated, Bonferroni alpha={alpha_bonf:.5f}")
    for row in tests:
        print(f"  {row['condition']:>20s} {row['metric']:>10s}: t={row['t']:+.3f} p={row['p']:.2e} sig={row['sig_bonferroni']}")

    pool_results = part_b_pool_experiments()
    print("\nPart B: pool sparsity (self-referential, subsampled each round)")
    for key, v in pool_results["subsampled_pool"].items():
        print(f"  {key:>28s}: var={v['var_mean']:.4f}+-{v['var_std']:.4f}")
    print("\nPart B: fixed external corpus (decoupled from agent opinions)")
    for key, v in pool_results["fixed_external_pool"].items():
        print(f"  {key:>28s}: var={v['var_mean']:.4f}+-{v['var_std']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    with open("extra_experiments_results.json", "w") as f:
        json.dump({"pairwise_tests": tests, "n_tests": n_tests, "alpha_bonferroni": alpha_bonf,
                    "pool_experiments": pool_results, "elapsed_seconds": elapsed}, f, indent=2)
    print("Saved extra_experiments_results.json")


if __name__ == "__main__":
    main()
