import json
import numpy as np
from scipy import stats

d = json.load(open("sim_results_v4.json"))

print("=== (G) Network-topology sign-flip re-check, 80 seeds (vs v3's 20) ===")
G = d["network_topology_80seeds"]
for eps in [0.10, 0.20]:
    row = {}
    for lam in [-1.0, 0.0, 1.0]:
        vals = [r["final_var"] for r in G if r["epsilon"] == eps and r["lambda"] == lam]
        row[lam] = (np.mean(vals), vals)
    tstat, pval = stats.ttest_rel(row[1.0][1], row[0.0][1])
    flip = row[1.0][0] < row[0.0][0]
    order_holds = row[-1.0][0] > row[1.0][0] > row[0.0][0]
    divboost_dominates = row[-1.0][0] > row[1.0][0] and row[-1.0][0] > row[0.0][0]
    frozen = [r["frozen_frac"] for r in G if r["epsilon"] == eps and r["lambda"] == -1.0]
    print(f"eps={eps:.2f}  random={row[0.0][0]:.4f}  engage={row[1.0][0]:.4f}  divboost={row[-1.0][0]:.4f}  "
          f"engage<random={flip}  paired_p={pval:.4f}  sig_flip={flip and pval < 0.05}  "
          f"divboost_dominates_both={divboost_dominates}  divboost_frozen={np.mean(frozen)*100:.1f}%")

print()
print("=== (H) Similarity-weighted (Gaussian-kernel) integration rule ===")
H = d["weighted_integration"]
for eps in [0.10, 0.20, 0.30]:
    row = {}
    for lam in [-1.0, 0.0, 1.0]:
        vals = [r["final_var"] for r in H if r["epsilon"] == eps and r["lambda"] == lam]
        row[lam] = (np.mean(vals), vals)
    tstat, pval = stats.ttest_rel(row[1.0][1], row[0.0][1])
    order_holds = row[-1.0][0] > row[1.0][0] > row[0.0][0]
    divboost_dominates = row[-1.0][0] > row[1.0][0] and row[-1.0][0] > row[0.0][0]
    frozen = [r["frozen_frac"] for r in H if r["epsilon"] == eps and r["lambda"] == -1.0]
    print(f"eps={eps:.2f}  random={row[0.0][0]:.4f}  engage={row[1.0][0]:.4f}  divboost={row[-1.0][0]:.4f}  "
          f"engage_vs_random_p={pval:.4f}  divboost_dominates_both={divboost_dominates}  divboost_frozen={np.mean(frozen)*100:.1f}%")
