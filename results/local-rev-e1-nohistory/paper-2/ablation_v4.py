"""
Round-3 revision experiments addressing round-3 review:
  (E) No-backfire ablation (reviewer question 2): does freezing persist if
      the backfire term (movement away from out-of-tolerance content) is
      removed entirely? Isolates whether the frozen fixed point depends on
      backfire vs. purely on within-tolerance near-zero-distance recs.
  (F) Final-opinion histograms at s=0.75 and s=1.0 (reviewer question 3):
      a concrete, non-inferred check of the "residual bumps" explanation for
      non-monotonic bimodality coefficient, via a simple mode-count / gap
      test on the empirical final-round opinion distribution.
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
    if strategy == "calibrated":
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
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# (E) No-backfire ablation: out-of-tolerance recommendations produce NO
# movement at all (hold), instead of the small repulsion term.
# ---------------------------------------------------------------------------
def run_no_backfire(strategy, param, seed, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                     return_final=False):
    rng = np.random.default_rng(seed)
    opinions = init_opinions(rng, n_agents)
    for t in range(n_rounds):
        pool = opinions.copy()
        rec = recommend(strategy, opinions, pool, rng, param)
        diff = rec - opinions
        within = np.abs(diff) <= CONFIDENCE_EPS
        delta = np.where(within, LEARNING_RATE * diff, 0.0)
        opinions = np.clip(opinions + delta, -1, 1)
    out = {
        "final_var": float(np.var(opinions)),
        "final_bc": float(bimodality_coefficient(opinions)),
        "final_extremity": float(np.mean(np.abs(opinions))),
    }
    if return_final:
        out["final_opinions"] = opinions.tolist()
    return out


def run_main(strategy, param, seed, n_agents=N_AGENTS, n_rounds=N_ROUNDS):
    """Standard (with-backfire) rule, but returns final opinion vector too."""
    rng = np.random.default_rng(seed)
    opinions = init_opinions(rng, n_agents)
    for t in range(n_rounds):
        pool = opinions.copy()
        rec = recommend(strategy, opinions, pool, rng, param)
        diff = rec - opinions
        within = np.abs(diff) <= CONFIDENCE_EPS
        delta = np.where(within, LEARNING_RATE * diff,
                          -BACKFIRE_RATE * np.sign(diff) * np.abs(diff))
        opinions = np.clip(opinions + delta, -1, 1)
    return opinions


def summarize(runs, key_metrics=("final_var", "final_bc", "final_extremity")):
    out = {}
    for m in key_metrics:
        v = np.array([r[m] for r in runs])
        out[f"{m}_mean"] = float(v.mean())
        out[f"{m}_std"] = float(v.std(ddof=1))
    return out


def count_modes(x, n_bins=40, min_gap_frac=0.02):
    """Crude empirical mode count: histogram, count local maxima separated
    by at least one bin with <= half the neighboring peak's count."""
    counts, edges = np.histogram(x, bins=n_bins, range=(-1, 1))
    peaks = []
    for i in range(len(counts)):
        left = counts[i - 1] if i > 0 else -1
        right = counts[i + 1] if i < len(counts) - 1 else -1
        if counts[i] > 0 and counts[i] >= left and counts[i] >= right and counts[i] > 2:
            peaks.append((i, counts[i]))
    # merge adjacent peak bins into clusters
    clusters = []
    for idx, c in peaks:
        if clusters and idx - clusters[-1][-1][0] <= 2:
            clusters[-1].append((idx, c))
        else:
            clusters.append([(idx, c)])
    return len(clusters), counts.tolist(), edges.tolist()


def main():
    t0 = time.time()
    results = {"no_backfire": {}, "mode_check": {}}

    print("Part E: no-backfire ablation (out-of-tolerance -> hold, not repel)")
    syco_runs, calib_runs = [], []
    for strategy, param in [("calibrated", 0.0), ("sycophantic", 1.0)]:
        key = f"{strategy}_{param}"
        runs = [run_no_backfire(strategy, param, seed=RNG_SEED_BASE + s) for s in range(N_SEEDS)]
        if strategy == "sycophantic":
            syco_runs = runs
        else:
            calib_runs = runs
        results["no_backfire"][key] = summarize(runs)
        r = results["no_backfire"][key]
        print(f"  {key:>18s}: var={r['final_var_mean']:.4f}+-{r['final_var_std']:.4f}  "
              f"bc={r['final_bc_mean']:.4f}+-{r['final_bc_std']:.4f}")

    t_nb, p_nb = stats.ttest_ind([r["final_var"] for r in syco_runs],
                                  [r["final_var"] for r in calib_runs], equal_var=False)
    results["no_backfire"]["welch_t_var"] = float(t_nb)
    results["no_backfire"]["welch_p_var"] = float(p_nb)
    print(f"  Welch t (var, syco1 vs calibrated, no-backfire rule): t={t_nb:.3f}, p={p_nb:.2e}")

    print("\nPart F: empirical mode count at s=0.75 vs s=1.0 (standard rule, seed=12345)")
    for s in (0.75, 1.0):
        opinions = run_main("sycophantic", s, seed=RNG_SEED_BASE)
        n_modes, counts, edges = count_modes(opinions)
        bc = bimodality_coefficient(opinions)
        results["mode_check"][f"s={s}"] = {
            "n_modes_est": n_modes,
            "bimodality_coeff": float(bc),
            "var": float(np.var(opinions)),
            "hist_counts": counts,
            "hist_edges": edges,
        }
        print(f"  s={s}: estimated modes={n_modes}, BC={bc:.3f}, var={np.var(opinions):.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    results["elapsed_seconds"] = elapsed
    results["config"] = {"n_seeds": N_SEEDS}

    with open("ablation_v4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ablation_v4_results.json")


if __name__ == "__main__":
    main()
