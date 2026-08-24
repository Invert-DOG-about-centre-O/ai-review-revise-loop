import json
import numpy as np
from scipy import stats

with open("sim_results.json") as f:
    main = json.load(f)
with open("sim_results_extra.json") as f:
    extra = json.load(f)

all_results = main["results"] + extra["results"]
EPSILONS = main["params"]["EPSILONS"]
ALL_LAMBDAS = sorted(set(r["lambda"] for r in all_results))

by_cell = {}
for r in all_results:
    by_cell.setdefault((r["lambda"], r["epsilon"]), []).append(r["final_var"])

print("=== Full grid means: final_var (rows=lambda incl. diversity-boosting, cols=epsilon) ===")
print("lambda\\eps  " + "  ".join(f"{e:5.2f}" for e in EPSILONS))
for lam in ALL_LAMBDAS:
    row = [np.mean(by_cell[(lam, eps)]) for eps in EPSILONS]
    print(f"{lam:8.2f}  " + "  ".join(f"{v:5.3f}" for v in row))

# Paired: diversity-boosting (lambda=-1.0) vs random (lambda=0.0) vs engagement (lambda=1.0)
print()
print("=== Paired comparisons vs random baseline (lambda=0.0), pooled across epsilon, n=150 pairs ===")
v0 = {(r["seed"], r["epsilon"]): r["final_var"] for r in main["results"] if r["lambda"] == 0.0}
for target_lam, label in [(1.0, "engagement-max (lambda=1.0)"), (-1.0, "diversity-boost (lambda=-1.0)")]:
    src = main["results"] if target_lam == 1.0 else extra["results"]
    vt = {(r["seed"], r["epsilon"]): r["final_var"] for r in src if r["lambda"] == target_lam}
    keys = sorted(set(v0.keys()) & set(vt.keys()))
    a = np.array([vt[k] for k in keys])
    b = np.array([v0[k] for k in keys])
    diff = a - b
    t, p_t = stats.ttest_rel(a, b)
    w, p_w = stats.wilcoxon(a, b)
    print(f"{label}: mean_diff={diff.mean():+.4f} (relative {100*diff.mean()/b.mean():+.2f}%)  "
          f"paired-t p={p_t:.3g}  wilcoxon p={p_w:.3g}  n={len(keys)}")

print(f"\nextra sim elapsed_sec = {extra['params']['elapsed_sec']:.1f}")
