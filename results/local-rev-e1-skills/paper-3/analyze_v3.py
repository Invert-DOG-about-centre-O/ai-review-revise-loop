import json
import numpy as np
from scipy import stats

d = json.load(open("sim_results_v3.json"))

print("=== (D) beta x mu, 40 seeds (vs v2's 15) ===")
D = d["beta_mu_sensitivity_40seeds"]
n_flip = 0
n_total = 0
for beta in [3.0, 6.0, 10.0]:
    for mu in [0.25, 0.5, 0.75]:
        row = {}
        for lam in [-1.0, 0.0, 1.0]:
            vals = [r["final_var"] for r in D if r["beta"] == beta and r["mu"] == mu and r["lambda"] == lam]
            row[lam] = (np.mean(vals), vals)
        engage_vals = row[1.0][1]
        random_vals = row[0.0][1]
        tstat, pval = stats.ttest_rel(engage_vals, random_vals)
        flip = row[1.0][0] < row[0.0][0]
        sig = pval < 0.05
        n_total += 1
        if flip and sig:
            n_flip += 1
        order_holds = row[-1.0][0] > row[1.0][0] > row[0.0][0]
        print(f"beta={beta:>4.1f} mu={mu:.2f}  random={row[0.0][0]:.4f}  engage={row[1.0][0]:.4f}  divboost={row[-1.0][0]:.4f}  "
              f"engage<random={flip}  paired_p={pval:.3f}  sig_flip={flip and sig}  divboost_order_holds={order_holds}")
print(f"-> engagement<random cells that are BOTH direction-flipped AND stat-sig (p<.05) at n=40 seeds: {n_flip}/{n_total}")

print()
print("=== (E) N sensitivity, 30 seeds (vs v2's 10) ===")
E = d["N_sensitivity_30seeds"]
for N in [50, 150, 500]:
    row = {}
    for lam in [-1.0, 0.0, 1.0]:
        vals = [r["final_var"] for r in E if r["N"] == N and r["lambda"] == lam]
        row[lam] = (np.mean(vals), np.std(vals), vals)
    tstat, pval = stats.ttest_rel(row[1.0][2], row[0.0][2])
    order_holds = row[-1.0][0] > row[1.0][0] > row[0.0][0]
    print(f"N={N:>4d}  random={row[0.0][0]:.4f}(+-{row[0.0][1]:.4f})  engage={row[1.0][0]:.4f}(+-{row[1.0][1]:.4f})  "
          f"divboost={row[-1.0][0]:.4f}(+-{row[-1.0][1]:.4f})  engage_vs_random_p={pval:.3f}  divboost_order_holds={order_holds}")

print()
print("=== (F) Network-topology (Erdos-Renyi, mean deg=20) ===")
F = d["network_topology"]
for eps in [0.10, 0.20, 0.30]:
    row = {}
    for lam in [-1.0, 0.0, 1.0]:
        vals = [r["final_var"] for r in F if r["epsilon"] == eps and r["lambda"] == lam]
        row[lam] = np.mean(vals)
    frozen = [r["frozen_frac"] for r in F if r["epsilon"] == eps and r["lambda"] == -1.0]
    order_holds = row[-1.0] > row[1.0] > row[0.0]
    print(f"eps={eps:.2f}  random={row[0.0]:.4f}  engage={row[1.0]:.4f}  divboost={row[-1.0]:.4f}  "
          f"divboost_frozen={np.mean(frozen)*100:.1f}%  order_holds={order_holds}")
