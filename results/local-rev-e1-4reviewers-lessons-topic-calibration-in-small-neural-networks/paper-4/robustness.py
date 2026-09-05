"""Leave-one-width-out robustness check for the width vs. ECE_pre correlation
(H1), plus a Fisher-z based minimum-detectable-effect note for the non-
significant H2 (Wilcoxon) results. Reads raw_results.json produced by
experiment.py -- does not retrain anything.
"""
import json
import numpy as np
from scipy import stats

WIDTHS = [2, 4, 8, 16, 32, 64, 128]
DATASETS = ["moons", "circles", "breast_cancer"]

with open("raw_results.json") as f:
    results = json.load(f)

out = {}
for dataset in DATASETS:
    rows = [r for r in results if r["dataset"] == dataset]
    loo = {}
    for drop_w in WIDTHS:
        sub = [r for r in rows if r["width"] != drop_w]
        log2w = np.array([np.log2(r["width"]) for r in sub])
        ece_pre = np.array([r["ece_pre"] for r in sub])
        corr = stats.spearmanr(log2w, ece_pre).correlation
        loo[drop_w] = float(corr)
    full_log2w = np.array([np.log2(r["width"]) for r in rows])
    full_ece = np.array([r["ece_pre"] for r in rows])
    full_corr = stats.spearmanr(full_log2w, full_ece).correlation
    out[dataset] = {
        "full_corr": float(full_corr),
        "leave_one_width_out_corr": loo,
        "min_loo_corr": float(min(loo.values())),
        "max_loo_corr": float(max(loo.values())),
        "sign_stable": all(np.sign(v) == np.sign(full_corr) for v in loo.values()),
    }

# Minimum detectable effect for Wilcoxon signed-rank at n=70, alpha=0.05, power=0.8
# (approx via normal approximation for matched-pairs rank-biserial correlation)
try:
    n = 70
    # Using a paired-sample z-approx: required rank-biserial effect for 80% power
    # is estimated via simulation-free normal approx (Cohen's approx: r ~ z/sqrt(n))
    alpha = 0.05
    power_target = 0.8
    z_alpha = stats.norm.ppf(1 - alpha)  # one-sided
    z_power = stats.norm.ppf(power_target)
    mde_r = (z_alpha + z_power) / np.sqrt(n)
    mde_note = float(mde_r)
except Exception as e:
    mde_note = None

with open("robustness.json", "w") as f:
    json.dump({"leave_one_width_out": out, "wilcoxon_n70_mde_rank_biserial_approx": mde_note}, f, indent=2)

print(json.dumps({"leave_one_width_out": out, "wilcoxon_n70_mde_rank_biserial_approx": mde_note}, indent=2))
