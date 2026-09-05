"""
Secondary (pre-registered as descriptive, non-confirmatory) analysis on top of
results.json: aggregate bias-by-width trends with bootstrap CIs, an effect
size + minimum-detectable-effect for the Mann-Whitney null result, and a
plain-language summary of the mechanism check.

This does NOT re-run the confirmatory Mann-Whitney test or change n; it only
describes the already-collected data further.
"""
import json
import numpy as np

rng = np.random.default_rng(12345)

with open("results.json") as f:
    d = json.load(f)

results = d["raw_results"]
widths = d["widths"]
seeds = d["seeds"]

def bootstrap_ci(vals, n_boot=5000, alpha=0.05):
    vals = np.array(vals, dtype=float)
    if len(vals) < 2:
        return float(vals.mean()) if len(vals) else float("nan"), float("nan"), float("nan")
    boots = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)

bias_by_width = {"digits": {}, "synthetic": {}}
ece_by_width = {"digits": {}, "synthetic": {}}
for dataset_name in ["digits", "synthetic"]:
    for w in widths:
        vals = [r["bias"] for r in results if r["dataset"] == dataset_name and r["width"] == w]
        eces = [r["ece"] for r in results if r["dataset"] == dataset_name and r["width"] == w]
        mean, lo, hi = bootstrap_ci(vals)
        bias_by_width[dataset_name][w] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}
        ece_by_width[dataset_name][w] = {"mean": float(np.mean(eces)), "std": float(np.std(eces, ddof=1))}

# Effect size for the Mann-Whitney crossover-width comparison: rank-biserial correlation
mw = d["mannwhitney_crossover_digits_vs_synthetic"]
digits_cross = [v for v in d["crossover_widths"]["digits"].values() if v is not None]
synth_cross = [v for v in d["crossover_widths"]["synthetic"].values() if v is not None]
n1, n2 = len(digits_cross), len(synth_cross)
U = mw["statistic"]
rank_biserial = 1 - (2 * U) / (n1 * n2) if n1 and n2 else float("nan")

# Minimum detectable effect (rough): what U would correspond to p=0.05 with these n, converted to rank-biserial
# Using normal approximation for Mann-Whitney U distribution under H0
mu_U = n1 * n2 / 2
sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) if (n1 + n2) > 0 else float("nan")
# smallest |U - mu_U| that reaches z=1.96
if sigma_U == sigma_U and sigma_U > 0:
    delta_U_for_sig = 1.96 * sigma_U
    mde_rank_biserial = (2 * delta_U_for_sig) / (n1 * n2)
else:
    mde_rank_biserial = float("nan")

out = {
    "bias_by_width_bootstrap_ci": bias_by_width,
    "ece_by_width": ece_by_width,
    "mannwhitney_effect_size_rank_biserial": rank_biserial,
    "mannwhitney_min_detectable_rank_biserial_at_current_n": mde_rank_biserial,
    "n_digits_crossover": n1,
    "n_synthetic_crossover": n2,
}

d["analysis"] = out
with open("results.json", "w") as f:
    json.dump(d, f, indent=2)

print(json.dumps(out, indent=2))
