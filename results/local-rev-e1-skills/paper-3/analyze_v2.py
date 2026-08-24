import json
import numpy as np

d = json.load(open("sim_results_v2.json"))

print("=== (A) Frozen fraction on exact main lambda=-1 grid ===")
A = d["frozen_on_main_grid"]
eps_vals = sorted(set(r["epsilon"] for r in A))
for eps in eps_vals:
    vals = [r["frozen_frac"] for r in A if r["epsilon"] == eps]
    fv = [r["final_var"] for r in A if r["epsilon"] == eps]
    print(f"eps={eps:.2f}  frozen_frac mean={np.mean(vals)*100:.1f}%  range=[{min(vals)*100:.1f},{max(vals)*100:.1f}]  final_var mean={np.mean(fv):.4f}")

print()
print("=== (B) Sensitivity to BETA, MU (eps=0.20) ===")
B = d["beta_mu_sensitivity"]
for beta in [3.0, 6.0, 10.0]:
    for mu in [0.25, 0.5, 0.75]:
        row = {}
        for lam in [-1.0, 0.0, 1.0]:
            vals = [r["final_var"] for r in B if r["beta"] == beta and r["mu"] == mu and r["lambda"] == lam]
            row[lam] = np.mean(vals)
        frozen = [r["frozen_frac"] for r in B if r["beta"] == beta and r["mu"] == mu and r["lambda"] == -1.0]
        print(f"beta={beta:>4.1f} mu={mu:.2f}  random={row[0.0]:.4f}  engage={row[1.0]:.4f}  divboost={row[-1.0]:.4f}  divboost_frozen={np.mean(frozen)*100:.1f}%  order_holds={row[-1.0] > row[1.0] > row[0.0]}")

print()
print("=== (C) Sensitivity to N (eps=0.20) ===")
C = d["N_sensitivity"]
for N in [50, 150, 500]:
    row = {}
    for lam in [-1.0, 0.0, 1.0]:
        vals = [r["final_var"] for r in C if r["N"] == N and r["lambda"] == lam]
        row[lam] = (np.mean(vals), np.std(vals))
    order_holds = row[-1.0][0] > row[1.0][0] > row[0.0][0]
    print(f"N={N:>4d}  random={row[0.0][0]:.4f}(+-{row[0.0][1]:.4f})  engage={row[1.0][0]:.4f}(+-{row[1.0][1]:.4f})  divboost={row[-1.0][0]:.4f}(+-{row[-1.0][1]:.4f})  order_holds={order_holds}")
