"""
Revision analyses for round-2 review:
(1) Holm-Bonferroni correction on the 9 beta-vs-baseline comparisons (variance & clusters).
(2) Matched-budget (fixed T/N) N-ablation with a real significance test on beta=0 vs beta=5.
(3) Robustness of cluster counts to an alternative, non-gap-statistic clustering rule
    (fixed absolute-distance single-linkage clustering) on the bridging-side betas.
"""
import csv
import json
import time
import numpy as np
from scipy import stats
from collections import defaultdict

from experiment import run_simulation, make_seed

t0 = time.time()

# ---------- (1) Holm-Bonferroni correction on main beta sweep ----------
rows = list(csv.DictReader(open("results_beta_sweep.csv")))
by_beta = defaultdict(list)
for r in rows:
    by_beta[float(r["beta"])].append(r)

baseline_var = [float(r["variance"]) for r in by_beta[0.0]]
baseline_clu = [float(r["n_clusters"]) for r in by_beta[0.0]]

betas_sorted = sorted(b for b in by_beta if b != 0.0)


def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


var_p, clu_p = [], []
for b in betas_sorted:
    v = [float(r["variance"]) for r in by_beta[b]]
    c = [float(r["n_clusters"]) for r in by_beta[b]]
    _, pv = stats.mannwhitneyu(v, baseline_var, alternative="two-sided")
    _, pc = stats.mannwhitneyu(c, baseline_clu, alternative="two-sided")
    var_p.append(pv)
    clu_p.append(pc)

var_p_adj = holm_bonferroni(np.array(var_p))
clu_p_adj = holm_bonferroni(np.array(clu_p))

print("=== (1) Holm-Bonferroni correction, 9 comparisons each ===")
print(f"{'beta':>6} {'p_var_raw':>10} {'p_var_holm':>11} {'p_clu_raw':>10} {'p_clu_holm':>11}")
holm_results = []
for b, pv, pva, pc, pca in zip(betas_sorted, var_p, var_p_adj, clu_p, clu_p_adj):
    print(f"{b:>6} {pv:>10.4f} {pva:>11.4f} {pc:>10.4f} {pca:>11.4f}")
    holm_results.append({"beta": b, "p_var_raw": pv, "p_var_holm": float(pva),
                          "p_clu_raw": pc, "p_clu_holm": float(pca)})

# ---------- (2) Matched-budget N-ablation with significance test ----------
# Fixed T/N=40 (matches N=100,T=4000 default) so interactions-per-agent is constant.
print("\n=== (2) Matched-budget (T = 40*N) N-ablation, beta=0 vs beta=5, 30 seeds ===")
N_TO_T_RATIO = 40
Ns = [50, 100, 200, 400]
n_seeds = 30
matched_rows = []
cond_index = 0
for beta_fixed in [0, 5]:
    for Nval in Ns:
        T_matched = N_TO_T_RATIO * Nval
        for s in range(n_seeds):
            seed = make_seed(3000 + cond_index, s, n_seeds)
            res = run_simulation(Nval, T_matched, beta_fixed, 0.15, 0.3, seed)
            matched_rows.append({"beta": beta_fixed, "N": Nval, "T": T_matched, "seed": s, **res})
        cond_index += 1

with open("results_N_ablation_matched.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
    w.writeheader()
    for r in matched_rows:
        w.writerow(r)

by_N_beta = defaultdict(list)
for r in matched_rows:
    by_N_beta[(r["N"], r["beta"])].append(r)

matched_summary = []
print(f"{'N':>5} {'clu_b0':>8} {'clu_b5':>8} {'p_clusters':>11}")
for Nval in Ns:
    c0 = [r["n_clusters"] for r in by_N_beta[(Nval, 0)]]
    c5 = [r["n_clusters"] for r in by_N_beta[(Nval, 5)]]
    _, p = stats.mannwhitneyu(c5, c0, alternative="two-sided")
    print(f"{Nval:>5} {np.mean(c0):>8.2f} {np.mean(c5):>8.2f} {p:>11.4f}")
    matched_summary.append({"N": Nval, "clusters_beta0": float(np.mean(c0)),
                             "clusters_beta5": float(np.mean(c5)), "p_clusters": float(p)})

# ---------- (3) Alternative clustering rule: fixed absolute-gap single-linkage ----------
# Instead of gap > 2/N (which scales with N), use a fixed absolute gap threshold
# independent of N/epsilon, applied to the *raw* final opinion vector.
print("\n=== (3) Alternative clustering (fixed absolute gap = 0.02) vs gap-statistic, main sweep ===")


def n_clusters_fixed_gap(x, gap_thresh=0.02):
    xs = np.sort(x)
    gaps = np.diff(xs)
    return int((gaps > gap_thresh).sum()) + 1


betas_check = [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
N, T, mu, eps = 100, 4000, 0.3, 0.15
thresholds = [0.01, 0.04]  # deliberately != 2/N=0.02 used in the paper, to test sensitivity
alt_summary = []
alt_by_beta = {thr: defaultdict(list) for thr in thresholds}
for cond_index, beta in enumerate(betas_check):
    for s in range(n_seeds):
        seed = make_seed(cond_index, s, n_seeds)
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, size=N)
        for t in range(T):
            i = rng.integers(0, N)
            diffs = np.abs(x - x[i])
            score = -beta * diffs
            score[i] = -np.inf
            finite = np.isfinite(score)
            score = score - np.max(score[finite])
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
            if d < eps:
                xi, xj = x[i], x[j]
                x[i] = xi + mu * (xj - xi)
                x[j] = xj + mu * (xi - xj)
        for thr in thresholds:
            alt_by_beta[thr][beta].append(n_clusters_fixed_gap(x, thr))

for thr in thresholds:
    baseline_alt = alt_by_beta[thr][0]
    print(f"-- fixed gap threshold = {thr} --")
    print(f"{'beta':>6} {'clu':>8} {'p_vs_baseline':>14}")
    for beta in betas_check:
        c = alt_by_beta[thr][beta]
        if beta == 0:
            p = float("nan")
        else:
            _, p = stats.mannwhitneyu(c, baseline_alt, alternative="two-sided")
        print(f"{beta:>6} {np.mean(c):>8.2f} {p:>14.4f}")
        alt_summary.append({"threshold": thr, "beta": beta, "clusters_mean": float(np.mean(c)),
                             "p_vs_baseline": (None if beta == 0 else float(p))})

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

with open("revision_analysis_summary.json", "w") as f:
    json.dump({
        "holm_bonferroni": holm_results,
        "matched_budget_N_ablation": matched_summary,
        "alt_clustering_fixed_gap": alt_summary,
        "elapsed_seconds": elapsed,
    }, f, indent=2)
