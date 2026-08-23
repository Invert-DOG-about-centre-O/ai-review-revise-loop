"""
Round-3 review follow-up: locate the population-size threshold at which
engagement bias (beta=5) starts significantly reducing fragmentation
under a matched budget (T = 40*N), by sweeping N at finer resolution
between the N=100 (no effect) and N=200 (large effect) points already
established in revision_analysis.py.
"""
import csv
import json
import time
import numpy as np
from scipy import stats
from collections import defaultdict

from experiment import run_simulation, make_seed

t0 = time.time()

N_TO_T_RATIO = 40
Ns = [100, 120, 140, 160, 180, 200]
n_seeds = 30
rows = []
cond_index = 0
for beta_fixed in [0, 5]:
    for Nval in Ns:
        T_matched = N_TO_T_RATIO * Nval
        for s in range(n_seeds):
            seed = make_seed(9000 + cond_index, s, n_seeds)
            res = run_simulation(Nval, T_matched, beta_fixed, 0.15, 0.3, seed)
            rows.append({"beta": beta_fixed, "N": Nval, "T": T_matched, "seed": s, **res})
        cond_index += 1

with open("results_fine_N_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

by_N_beta = defaultdict(list)
for r in rows:
    by_N_beta[(r["N"], r["beta"])].append(r)

summary = []
print(f"{'N':>5} {'clu_b0':>8} {'clu_b5':>8} {'p_clusters':>11}")
for Nval in Ns:
    c0 = [r["n_clusters"] for r in by_N_beta[(Nval, 0)]]
    c5 = [r["n_clusters"] for r in by_N_beta[(Nval, 5)]]
    _, p = stats.mannwhitneyu(c5, c0, alternative="two-sided")
    print(f"{Nval:>5} {np.mean(c0):>8.2f} {np.mean(c5):>8.2f} {p:>11.4f}")
    summary.append({"N": Nval, "clusters_beta0": float(np.mean(c0)),
                     "clusters_beta5": float(np.mean(c5)), "p_clusters": float(p)})

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

with open("fine_n_sweep_summary.json", "w") as f:
    json.dump({"fine_N_sweep": summary, "elapsed_seconds": elapsed}, f, indent=2)
