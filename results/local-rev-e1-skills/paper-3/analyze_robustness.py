import json
import numpy as np

with open("sim_results_robustness.json") as f:
    d = json.load(f)
results = d["results"]
KS = d["params"]["KS"]
EPSILONS = d["params"]["EPSILONS"]
LAMBDAS = d["params"]["LAMBDAS"]

by = {}
for r in results:
    by.setdefault((r["K"], r["lambda"], r["epsilon"]), []).append(r)

print("=== final_var and frozen_frac by K, lambda, epsilon (mean over 20 seeds) ===")
for K in KS:
    print(f"--- K={K} ---")
    print("lambda\\eps  " + "  ".join(f"{e:5.2f}" for e in EPSILONS))
    for lam in LAMBDAS:
        row_var = [np.mean([r["final_var"] for r in by[(K, lam, e)]]) for e in EPSILONS]
        row_frz = [np.mean([r["frozen_frac"] for r in by[(K, lam, e)]]) for e in EPSILONS]
        print(f"var  lam={lam:5.1f}  " + "  ".join(f"{v:5.3f}" for v in row_var))
        print(f"frz  lam={lam:5.1f}  " + "  ".join(f"{v:5.3f}" for v in row_frz))

print()
print("=== Ordering check: does final_var(diversity-boost) > final_var(engagement) > final_var(random) hold at eps=0.20 for every K? ===")
for K in KS:
    v_div = np.mean([r["final_var"] for r in by[(K, -1.0, 0.20)]])
    v_rand = np.mean([r["final_var"] for r in by[(K, 0.0, 0.20)]])
    v_eng = np.mean([r["final_var"] for r in by[(K, 1.0, 0.20)]])
    ok = v_div > v_eng > v_rand
    print(f"K={K}: random={v_rand:.3f} engagement={v_eng:.3f} diversity-boost={v_div:.3f}  order_holds={ok}")
