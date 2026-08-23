"""
Additional analyses requested by round-2 peer review:
  1. A trend test (Jonckheere-Terpstra style, via permutation) for monotonicity
     of final variance across the full w=0.0..0.5 weight sweep, rather than
     relying on a single pairwise w=0.3-vs-0.0 comparison.
  2. Permutation tests on the epsilon-ablation table's headline claims
     (engagement "flat" across eps=0.35/0.5; random and bridging "decreasing"
     from eps=0.2 to eps=0.5).
  3. A mechanical/empirical diagnostic for why bridging produces more clusters
     than similarity: cluster-center dispersion using the final opinions
     already stored in results.json.

Reuses raw outputs already on disk (results.json, ablation_results.json,
weight_sweep_results.json) -- no simulation re-run needed.
"""
import json
import numpy as np

from simulation import cluster_count


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


def jonckheere_terpstra_stat(groups):
    """Sum of Mann-Whitney U statistics for all i<j group pairs (ordered groups)."""
    J = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi, gj = groups[i], groups[j]
            # U counts pairs where gj element > gi element (favors increasing trend)
            u = sum(1 for x in gi for y in gj if y > x)
            J += u
    return J


def jt_permutation_test(groups, n_perm=100000, seed=3):
    """Permutation test for the JT statistic: shuffle group labels, keep group sizes fixed."""
    rng = np.random.default_rng(seed)
    obs = jonckheere_terpstra_stat(groups)
    sizes = [len(g) for g in groups]
    pooled = np.concatenate(groups)
    n = len(pooled)
    null = np.empty(n_perm)
    idx = np.arange(n)
    for p in range(n_perm):
        rng.shuffle(idx)
        perm = pooled[idx]
        cuts = np.cumsum([0] + sizes)
        gs = [perm[cuts[k]:cuts[k + 1]] for k in range(len(sizes))]
        null[p] = jonckheere_terpstra_stat(gs)
    p_val = np.mean(null >= obs)  # one-sided: testing for increasing trend
    return obs, p_val


def main():
    # --- 1. Monotonicity (trend) test on the weight sweep ---
    with open("weight_sweep_results.json") as f:
        wsweep = json.load(f)
    weights = sorted(set(r["w"] for r in wsweep))
    groups = [np.array([r["final_variance"] for r in wsweep if r["w"] == w]) for w in weights]
    jt_obs, jt_p = jt_permutation_test(groups, n_perm=50000, seed=3)
    max_j = sum(len(groups[i]) * len(groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups)))
    print(f"Jonckheere-Terpstra trend test across w={weights}: J={jt_obs}/{max_j}, one-sided p={jt_p:.5f}")

    # --- 2. Permutation tests on ablation (epsilon) table claims ---
    with open("ablation_results.json") as f:
        abl = json.load(f)
    policies = ["random", "similarity", "engagement", "bridging"]

    def fv_at(eps, policy):
        return np.array([r["final_variance"] for r in abl
                          if r["epsilon"] == eps and r["policy"] == policy])

    abl_lines = []
    checks = [
        ("engagement", 0.35, 0.5, "engagement variance flat eps=0.35 vs eps=0.5"),
        ("random", 0.2, 0.5, "random variance decreasing eps=0.2 vs eps=0.5"),
        ("bridging", 0.2, 0.5, "bridging variance decreasing eps=0.2 vs eps=0.5"),
        ("similarity", 0.2, 0.5, "similarity variance eps=0.2 vs eps=0.5"),
    ]
    for policy, e1, e2, label in checks:
        a, b = fv_at(e1, policy), fv_at(e2, policy)
        obs, p = permutation_test(a, b, n_perm=100000, seed=7)
        diff_e2_minus_e1 = -obs
        line = f"{label}: diff(eps={e2}-eps={e1})={diff_e2_minus_e1:+.4f}, p={p:.4f}"
        print(line)
        abl_lines.append({"policy": policy, "eps1": e1, "eps2": e2, "diff_eps2_minus_eps1": diff_e2_minus_e1, "p": p})

    # --- 3. Bridging fragmentation diagnostic: cluster-center dispersion ---
    with open("results.json") as f:
        main_results = json.load(f)

    def cluster_centers(opinions, gap=0.08):
        s = np.sort(np.asarray(opinions))
        if len(s) == 0:
            return []
        splits = np.where(np.diff(s) > gap)[0] + 1
        groups_ = np.split(s, splits)
        return [float(np.mean(g)) for g in groups_]

    diag_lines = []
    for policy in ["similarity", "engagement", "bridging"]:
        rows = [r for r in main_results if r["policy"] == policy]
        spreads = []
        extreme_frac = []
        for r in rows:
            centers = cluster_centers(r["final_opinions"])
            if len(centers) >= 2:
                spreads.append(max(centers) - min(centers))
            else:
                spreads.append(0.0)
            ops = np.array(r["final_opinions"])
            extreme_frac.append(float(np.mean(np.abs(ops) > 0.5)))
        spreads = np.array(spreads)
        extreme_frac = np.array(extreme_frac)
        line = (f"{policy}: cluster-center spread={spreads.mean():.3f}+/-{spreads.std():.3f}, "
                f"frac agents |opinion|>0.5 = {extreme_frac.mean():.3f}+/-{extreme_frac.std():.3f}")
        print(line)
        diag_lines.append({"policy": policy, "cluster_spread_mean": float(spreads.mean()),
                            "cluster_spread_std": float(spreads.std()),
                            "extreme_frac_mean": float(extreme_frac.mean())})

    out = {
        "jt_trend_test": {"J": jt_obs, "J_max": max_j, "p_one_sided": jt_p, "weights": weights},
        "ablation_pairwise": abl_lines,
        "bridging_diagnostic": diag_lines,
    }
    with open("round2_analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
