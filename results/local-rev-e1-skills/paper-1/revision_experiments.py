"""
Revision experiments addressing round-1 review questions:
 (1) Formal significance test for the alpha-variance relationship (permutation
     test on the coarse-grid slope/correlation, per condition), plus a
     higher-seed-count rerun of the two weakest-effect conditions to check
     whether near-zero correlations are underpowered rather than genuinely weak.
 (2) A quick mu/N ablation (mu=0.5 and N=100) at a few alpha values crossed
     with both epsilon settings, to test whether the epsilon>>alpha ranking
     survives outside mu=0.3, N=200.
 (3) Calibration-target sensitivity: recompute the epsilon-vs-alpha comparison
     if we had calibrated to the low (1.3x) or high (1.6x) end of Huszar et
     al.'s range instead of the midpoint (1.4x), reusing the already-collected
     results_calibration.csv (no resimulation needed).
"""
import csv
import json
import time
import numpy as np
from experiment import run_simulation, sweep, write_csv

t0 = time.time()
rng_master = np.random.default_rng(999)

# ---------------------------------------------------------------------------
# (1) Permutation test on the coarse-grid alpha->variance relationship, per
# condition. Null: alpha carries no information about variance, so shuffle
# which alpha label is attached to which mean-variance point and recompute
# Pearson r; two-sided p-value = fraction of |r_perm| >= |r_observed|.
# ---------------------------------------------------------------------------
coarse_rows = list(csv.DictReader(open("results_coarse.csv")))
for r in coarse_rows:
    for k in ("alpha", "epsilon", "variance_mean"):
        r[k] = float(r[k])

perm_results = {}
n_perm = 20000
for topology in ["mixing", "smallworld"]:
    for epsilon in [0.15, 0.35]:
        sub = sorted([r for r in coarse_rows if r["topology"] == topology and r["epsilon"] == epsilon],
                     key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in sub])
        v = np.array([r["variance_mean"] for r in sub])
        r_obs = float(np.corrcoef(a, v)[0, 1])
        perm_rs = np.empty(n_perm)
        for i in range(n_perm):
            v_perm = rng_master.permutation(v)
            perm_rs[i] = np.corrcoef(a, v_perm)[0, 1]
        p_value = float(np.mean(np.abs(perm_rs) >= abs(r_obs)))
        key = f"{topology}_eps{epsilon}"
        perm_results[key] = {"r_observed": round(r_obs, 4), "perm_p_value": round(p_value, 4)}
        print(f"[permtest] {key}: r={r_obs:.4f} p={p_value:.4f}")

print(f"permutation tests done at {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# (1b) Higher-seed rerun of the two weakest-effect conditions (mixing/eps0.35,
# smallworld/eps0.15) to check whether the near-zero / sign-flip correlations
# are underpowered. Use 30 seeds instead of 5, same coarse alpha grid.
# ---------------------------------------------------------------------------
coarse_alphas = [round(a, 2) for a in np.arange(0.0, 6.01, 0.5)]
hp_seeds = list(range(1, 31))
highpower_rows = []
for topology, epsilon in [("mixing", 0.35), ("smallworld", 0.15)]:
    rows = sweep(alphas=coarse_alphas, topologies=[topology], epsilons=[epsilon],
                 seeds=hp_seeds, N=200, T=150, mu=0.3, label=f"highpower-{topology}-{epsilon}")
    highpower_rows.extend(rows)
write_csv(highpower_rows, "results_highpower.csv")

highpower_summary = {}
for topology, epsilon in [("mixing", 0.35), ("smallworld", 0.15)]:
    sub = sorted([r for r in highpower_rows if r["topology"] == topology and r["epsilon"] == epsilon],
                 key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in sub])
    v = np.array([r["variance_mean"] for r in sub])
    r_obs = float(np.corrcoef(a, v)[0, 1])
    v_range = float(v.max() - v.min())
    key = f"{topology}_eps{epsilon}"
    highpower_summary[key] = {"r_30seeds": round(r_obs, 4), "variance_range_30seeds": round(v_range, 4),
                               "r_5seeds_original": None}
    print(f"[highpower] {key}: r(30 seeds)={r_obs:.4f} range={v_range:.4f}")
print(f"high-power rerun done at {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# (2) mu / N ablation: mu=0.5 and N=100, alpha in {0,3,6}, both epsilons,
# mixing topology only, 5 seeds -- cheap, direct test of the flagged gap.
# ---------------------------------------------------------------------------
ablation_rows = []
for mu_val, N_val, tag in [(0.5, 200, "mu0.5"), (0.3, 100, "N100")]:
    rows = sweep(alphas=[0.0, 3.0, 6.0], topologies=["mixing"], epsilons=[0.15, 0.35],
                 seeds=[1, 2, 3, 4, 5], N=N_val, T=150, mu=mu_val, label=f"ablation-{tag}")
    for row in rows:
        row["ablation_tag"] = tag
    ablation_rows.extend(rows)
write_csv(ablation_rows, "results_ablation.csv")
print(f"mu/N ablation done at {time.time()-t0:.1f}s")

ablation_summary = {}
for tag in ["mu0.5", "N100"]:
    for epsilon in [0.15, 0.35]:
        sub = sorted([r for r in ablation_rows if r["ablation_tag"] == tag and r["epsilon"] == epsilon],
                     key=lambda r: r["alpha"])
        v = np.array([r["variance_mean"] for r in sub])
        ablation_summary[f"{tag}_eps{epsilon}"] = {
            "variance_at_alpha0": round(float(v[0]), 4),
            "variance_at_alpha6": round(float(v[-1]), 4),
            "alpha_range_variance_shift": round(float(v.max() - v.min()), 4),
        }
    v_eps15_a0 = [r for r in ablation_rows if r["ablation_tag"] == tag and r["epsilon"] == 0.15 and r["alpha"] == 0.0][0]["variance_mean"]
    v_eps35_a0 = [r for r in ablation_rows if r["ablation_tag"] == tag and r["epsilon"] == 0.35 and r["alpha"] == 0.0][0]["variance_mean"]
    ablation_summary[f"{tag}_epsilon_shift_at_alpha0"] = round(abs(v_eps15_a0 - v_eps35_a0), 4)

for k, v in ablation_summary.items():
    print("[ablation]", k, v)

# ---------------------------------------------------------------------------
# (3) Calibration-target sensitivity: reuse results_calibration.csv (already
# a fine alpha sweep at mixing/eps=0.35) and find the alpha closest to 1.3x
# and 1.6x instead of the 1.4x midpoint used in the main paper.
# ---------------------------------------------------------------------------
calib_rows = list(csv.DictReader(open("results_calibration.csv")))
for r in calib_rows:
    for k in ("alpha", "amp_ratio_mean", "variance_mean"):
        r[k] = float(r[k])

alpha0_variance = [r for r in calib_rows if r["alpha"] == 0.0][0]["variance_mean"]
calib_sensitivity = {}
for target in [1.3, 1.4, 1.6]:
    best = min(calib_rows, key=lambda r: abs(r["amp_ratio_mean"] - target))
    calib_sensitivity[f"target_{target}"] = {
        "calibrated_alpha": best["alpha"],
        "achieved_ratio": round(best["amp_ratio_mean"], 4),
        "variance_at_calibration": round(best["variance_mean"], 4),
        "variance_shift_from_alpha0": round(abs(best["variance_mean"] - alpha0_variance), 4),
    }
print("[calib sensitivity]", json.dumps(calib_sensitivity, indent=2))

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
out = {
    "permutation_tests": perm_results,
    "highpower_rerun": highpower_summary,
    "mu_N_ablation": ablation_summary,
    "calibration_target_sensitivity": calib_sensitivity,
}
with open("revision_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"TOTAL revision experiments time: {time.time()-t0:.1f}s")
