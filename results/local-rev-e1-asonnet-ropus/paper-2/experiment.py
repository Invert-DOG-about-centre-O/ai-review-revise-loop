"""
Agent-based simulation of the engagement-polarization trade-off under
algorithmically biased interaction selection (a simplified model of
recommender-mediated social interaction), built on the Deffuant-Weisbuch
bounded-confidence opinion dynamics model.

Baseline: beta=0 -> uniform random pairing (no algorithm).
Engagement-maximizing recommender: beta>0 -> biases pairing toward similar
    opinions (homophily amplification), a proxy for engagement-optimizing
    curation.
Bridging / diversity-boosting intervention: beta<0 -> biases pairing toward
    dissimilar opinions.

Outputs raw CSV results and a small set of summary plots/tables to this
directory. Designed to run in well under a few minutes on CPU.
"""
import numpy as np
import time
import csv
import json
from scipy import stats

RNG_SEED_BASE = 12345


def make_seed(cond_index, s, n_seeds=1000):
    """Deterministic, integer-only seed (no Python hash(), which is
    salted per-process via PYTHONHASHSEED and was the source of the
    non-reproducibility flagged in review)."""
    return RNG_SEED_BASE + cond_index * n_seeds + s


def run_simulation(N, T, beta, epsilon, mu, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)
    engagement_sum = 0.0
    n_updates = 0

    for t in range(T):
        i = rng.integers(0, N)
        diffs = np.abs(x - x[i])
        score = -beta * diffs
        score[i] = -np.inf
        finite = np.isfinite(score)
        score = score - np.max(score[finite])  # stability
        w = np.exp(np.clip(score, -50, 50))
        w[i] = 0.0
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            j = rng.integers(0, N)
            while j == i:
                j = rng.integers(0, N)
        else:
            p = w / w_sum
            j = rng.choice(N, p=p)

        d = abs(x[i] - x[j])
        engagement_sum += np.exp(-d)  # proxy: similarity of paired content/opinion

        if d < epsilon:
            xi, xj = x[i], x[j]
            x[i] = xi + mu * (xj - xi)
            x[j] = xj + mu * (xi - xj)
            n_updates += 1

    variance = float(np.var(x))
    mean_engagement = engagement_sum / T
    update_rate = n_updates / T

    # bimodality coefficient (Sarle's): >0.555 suggests bimodal for finite samples
    m3 = float(((x - x.mean()) ** 3).mean())
    s = x.std()
    skew = m3 / (s ** 3 + 1e-12)
    m4 = float(((x - x.mean()) ** 4).mean())
    kurt = m4 / (s ** 4 + 1e-12)  # not excess kurtosis
    bc = (skew ** 2 + 1) / (kurt + 3 * ((N - 1) ** 2) / ((N - 2) * (N - 3) + 1e-9)) if N > 3 else float('nan')

    # cluster count via gap statistic on sorted opinions
    xs = np.sort(x)
    gaps = np.diff(xs)
    gap_thresh = 2.0 / N  # heuristic: gap bigger than ~2x average spacing under uniform
    n_clusters = int((gaps > gap_thresh).sum()) + 1

    return {
        "variance": variance,
        "engagement": mean_engagement,
        "update_rate": update_rate,
        "bimodality": bc,
        "n_clusters": n_clusters,
    }


def sweep(betas, N, T, epsilon, mu, n_seeds, tag):
    rows = []
    for cond_index, beta in enumerate(betas):
        for s in range(n_seeds):
            seed = make_seed(cond_index, s, n_seeds)
            res = run_simulation(N, T, beta, epsilon, mu, seed)
            row = {"tag": tag, "beta": beta, "N": N, "T": T, "epsilon": epsilon,
                   "mu": mu, "seed": s, **res}
            rows.append(row)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows, group_key, baseline_key=None):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    baseline_rs = groups.get(baseline_key) if baseline_key is not None else None
    summary = []
    for k in sorted(groups.keys()):
        rs = groups[k]
        var_mean = np.mean([r["variance"] for r in rs])
        var_std = np.std([r["variance"] for r in rs])
        eng_mean = np.mean([r["engagement"] for r in rs])
        eng_std = np.std([r["engagement"] for r in rs])
        clu_mean = np.mean([r["n_clusters"] for r in rs])
        bc_mean = np.mean([r["bimodality"] for r in rs])
        row = {
            group_key: k, "variance_mean": var_mean, "variance_std": var_std,
            "engagement_mean": eng_mean, "engagement_std": eng_std,
            "n_clusters_mean": clu_mean, "bimodality_mean": bc_mean,
            "n_runs": len(rs),
        }
        if baseline_rs is not None and k != baseline_key:
            _, p_var = stats.mannwhitneyu(
                [r["variance"] for r in rs], [r["variance"] for r in baseline_rs],
                alternative="two-sided")
            _, p_clu = stats.mannwhitneyu(
                [r["n_clusters"] for r in rs], [r["n_clusters"] for r in baseline_rs],
                alternative="two-sided")
            row["p_variance_vs_baseline"] = float(p_var)
            row["p_clusters_vs_baseline"] = float(p_clu)
        summary.append(row)
    return summary


def main():
    t0 = time.time()
    N = 100
    T = 4000
    mu = 0.3
    epsilon_default = 0.15
    n_seeds = 30

    betas_main = [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
    rows_main = sweep(betas_main, N, T, epsilon_default, mu, n_seeds, tag="beta_sweep")
    write_csv(rows_main, "results_beta_sweep.csv")
    summary_main = summarize(rows_main, "beta", baseline_key=0)
    with open("summary_beta_sweep.json", "w") as f:
        json.dump(summary_main, f, indent=2)

    # Ablation 1: confidence threshold epsilon, at fixed beta=5 (engagement-maximizing) vs beta=0
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    rows_eps = []
    cond_index = 0
    for beta_fixed in [0, 5]:
        for eps in epsilons:
            for s in range(n_seeds):
                seed = make_seed(1000 + cond_index, s, n_seeds)
                res = run_simulation(N, T, beta_fixed, eps, mu, seed)
                rows_eps.append({"tag": "eps_sweep", "beta": beta_fixed, "epsilon": eps,
                                  "N": N, "T": T, "mu": mu, "seed": s, **res})
            cond_index += 1
    write_csv(rows_eps, "results_epsilon_ablation.csv")

    # Ablation 2: population size N, at fixed beta=5 vs beta=0
    Ns = [50, 100, 200, 400]
    rows_N = []
    cond_index = 0
    for beta_fixed in [0, 5]:
        for Nval in Ns:
            for s in range(n_seeds):
                seed = make_seed(2000 + cond_index, s, n_seeds)
                res = run_simulation(Nval, T, beta_fixed, epsilon_default, mu, seed)
                rows_N.append({"tag": "N_sweep", "beta": beta_fixed, "N": Nval,
                                "T": T, "epsilon": epsilon_default, "mu": mu,
                                "seed": s, **res})
            cond_index += 1
    write_csv(rows_N, "results_N_ablation.csv")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print("\n=== Beta sweep summary (variance, engagement, clusters) ===")
    for row in summary_main:
        pv = row.get("p_variance_vs_baseline")
        pc = row.get("p_clusters_vs_baseline")
        pv_s = f"{pv:.3f}" if pv is not None else "--"
        pc_s = f"{pc:.3f}" if pc is not None else "--"
        print(f"beta={row['beta']:>5}: var={row['variance_mean']:.4f}±{row['variance_std']:.4f} (p={pv_s}) "
              f"eng={row['engagement_mean']:.4f}±{row['engagement_std']:.4f} "
              f"clusters={row['n_clusters_mean']:.2f} (p={pc_s}) bc={row['bimodality_mean']:.3f}")

    with open("run_log.txt", "w") as f:
        f.write(f"Elapsed seconds: {elapsed:.2f}\n")
        f.write(f"N={N} T={T} mu={mu} epsilon_default={epsilon_default} n_seeds={n_seeds}\n")
        f.write("Beta sweep summary:\n")
        for row in summary_main:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
