"""
Analysis of raw_results.json for the calibration-in-small-networks study.

Pre-registered hypotheses (fixed BEFORE any of these numbers were inspected):

H1 (one-sided): within the width sweep studied here, ECE before temperature
   scaling (ece_raw) increases with width, replicating the "bigger models
   are more miscalibrated" trend from Guo et al. 2017 in a small-network
   regime. Test: Spearman correlation of width vs ece_raw pooled over all
   seeds, per dataset x condition.

H2 (two-sided): label smoothing during training makes no difference to ECE
   AFTER post-hoc temperature scaling, i.e. temperature scaling alone
   removes whatever benefit label smoothing would have added, in this
   width range. Test: paired Wilcoxon signed-rank on
   (ece_ts_baseline - ece_ts_label_smooth), paired by
   (dataset, width, seed_idx) since each pair shares the identical train
   /val/test split (paired by construction, not by nominal seed label only
   -- see method for why this pairing is valid here).

Null for H1 is fixed in advance (no correlation / non-increasing trend);
null for H2 is fixed in advance (no paired difference). Both are tested
here, not chosen after inspecting results.
"""
import json
from itertools import product

import numpy as np
from scipy import stats

with open("raw_results.json") as f:
    data = json.load(f)

results = data["results"]
widths = data["config"]["widths"]
datasets = sorted(set(r["dataset"] for r in results))
conditions = data["config"]["conditions"]

out = {"config": data["config"], "warnings_count": len(data["warnings"])}

# ---------- Descriptive summary table ----------
summary = []
for ds, cond, w in product(datasets, conditions, widths):
    rows = [
        r
        for r in results
        if r["dataset"] == ds and r["condition"] == cond and r["width"] == w
    ]
    ece_raw = np.array([r["ece_raw"] for r in rows])
    ece_ts = np.array([r["ece_ts"] for r in rows])
    acc = np.array([r["acc_test"] for r in rows])
    T = np.array([r["temperature"] for r in rows])
    summary.append(
        {
            "dataset": ds,
            "condition": cond,
            "width": w,
            "n": len(rows),
            "acc_mean": float(acc.mean()),
            "ece_raw_mean": float(ece_raw.mean()),
            "ece_raw_std": float(ece_raw.std(ddof=1)),
            "ece_ts_mean": float(ece_ts.mean()),
            "ece_ts_std": float(ece_ts.std(ddof=1)),
            "temperature_mean": float(T.mean()),
        }
    )
out["summary_table"] = summary

# ---------- H1: width vs ece_raw (Spearman), pooled per dataset x condition ----------
h1 = {}
for ds, cond in product(datasets, conditions):
    rows = [r for r in results if r["dataset"] == ds and r["condition"] == cond]
    w_arr = np.array([r["width"] for r in rows])
    e_arr = np.array([r["ece_raw"] for r in rows])
    rho, p_two = stats.spearmanr(w_arr, e_arr)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    # leave-one-width-out robustness (lesson: check sensitivity to single group)
    loo_rhos = []
    for w_drop in widths:
        mask = w_arr != w_drop
        r_loo, _ = stats.spearmanr(w_arr[mask], e_arr[mask])
        loo_rhos.append(float(r_loo))
    h1[f"{ds}__{cond}"] = {
        "spearman_rho": float(rho),
        "p_two_sided": float(p_two),
        "p_one_sided": float(p_one),
        "n": len(rows),
        "loo_rho_min": float(min(loo_rhos)),
        "loo_rho_max": float(max(loo_rhos)),
        "loo_rhos_by_width_dropped": dict(zip(widths, loo_rhos)),
    }
out["H1_width_vs_ece_raw"] = h1

# ---------- H2: paired Wilcoxon, baseline vs label_smooth, post-TS ECE ----------
by_key_base = {
    (r["dataset"], r["width"], r["seed_idx"]): r
    for r in results
    if r["condition"] == "baseline"
}
by_key_ls = {
    (r["dataset"], r["width"], r["seed_idx"]): r
    for r in results
    if r["condition"] == "label_smooth"
}
keys = sorted(set(by_key_base) & set(by_key_ls))
diffs = np.array(
    [by_key_base[k]["ece_ts"] - by_key_ls[k]["ece_ts"] for k in keys]
)
w_stat, p_wilcoxon = stats.wilcoxon(diffs)
n_pairs = len(diffs)
n_pos = int((diffs > 0).sum())
n_neg = int((diffs < 0).sum())
n_zero = int((diffs == 0).sum())
# rank-biserial effect size for Wilcoxon signed-rank
nonzero = diffs[diffs != 0]
ranks = stats.rankdata(np.abs(nonzero))
r_plus = ranks[nonzero > 0].sum()
r_minus = ranks[nonzero < 0].sum()
rank_biserial = (r_plus - r_minus) / ranks.sum() if len(nonzero) else 0.0

# bootstrap CI on mean paired difference
rng = np.random.default_rng(42)
boot_means = [
    diffs[rng.integers(0, n_pairs, n_pairs)].mean() for _ in range(10000)
]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

out["H2_label_smoothing_vs_baseline_post_TS"] = {
    "n_pairs": n_pairs,
    "n_pos_baseline_worse": n_pos,
    "n_neg_baseline_better": n_neg,
    "n_zero": n_zero,
    "mean_diff_baseline_minus_ls": float(diffs.mean()),
    "median_diff_baseline_minus_ls": float(np.median(diffs)),
    "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
    "wilcoxon_statistic": float(w_stat),
    "p_value_two_sided": float(p_wilcoxon),
    "rank_biserial_effect_size": float(rank_biserial),
}

# ---------- Power analysis for H2 (post-hoc, given n=n_pairs) ----------
# Simulate detectable rank-biserial effect size at alpha=0.05, power=0.80
# via simulated paired Wilcoxon under a shifted-normal alternative matched
# to the observed residual SD of the paired differences.
sd_diff = diffs.std(ddof=1)
alpha = 0.05
target_power = 0.80


def simulate_power(effect_mean, sd, n, n_sims=2000, seed=123):
    rng_local = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_sims):
        sample = rng_local.normal(effect_mean, sd, size=n)
        if np.allclose(sample, 0):
            continue
        try:
            _, p = stats.wilcoxon(sample)
        except ValueError:
            continue
        if p < alpha:
            rejects += 1
    return rejects / n_sims


# binary search on effect_mean (in units of sd_diff) for 80% power at n_pairs
lo_eff, hi_eff = 0.0, 3.0 * sd_diff
for _ in range(12):
    mid = (lo_eff + hi_eff) / 2
    p_hit = simulate_power(mid, sd_diff, n_pairs, n_sims=800)
    if p_hit < target_power:
        lo_eff = mid
    else:
        hi_eff = mid
mde_mean_diff = hi_eff
mde_in_sd_units = mde_mean_diff / sd_diff if sd_diff > 0 else float("nan")

out["H2_power_analysis"] = {
    "observed_n_pairs": n_pairs,
    "observed_sd_of_paired_diff": float(sd_diff),
    "alpha": alpha,
    "target_power": target_power,
    "minimum_detectable_mean_diff_ece": float(mde_mean_diff),
    "minimum_detectable_effect_in_sd_units": float(mde_in_sd_units),
    "note": "Simulated via paired-normal approximation matched to observed "
    "residual SD; MDE is the mean paired ECE difference detectable with "
    "80% power at alpha=0.05 given the achieved n_pairs.",
}

# ---------- Descriptive check: does TS reduce ECE regardless of condition? ----------
ts_effect = {}
for cond in conditions:
    rows = [r for r in results if r["condition"] == cond]
    raw = np.array([r["ece_raw"] for r in rows])
    ts = np.array([r["ece_ts"] for r in rows])
    diff = raw - ts
    stat, p = stats.wilcoxon(diff)
    ts_effect[cond] = {
        "n": len(rows),
        "mean_ece_raw": float(raw.mean()),
        "mean_ece_ts": float(ts.mean()),
        "mean_reduction": float(diff.mean()),
        "wilcoxon_p": float(p),
    }
out["descriptive_TS_reduces_ECE"] = ts_effect

# ---------- Multiplicity note ----------
out["multiplicity_note"] = (
    "Two primary pre-registered tests (H1, H2) plus one descriptive "
    "check reported without correction (not treated as a formal test); "
    "leave-one-width-out robustness checks on H1 are diagnostics, not "
    "additional hypothesis tests."
)

with open("results.json", "w") as f:
    json.dump(out, f, indent=1)

print("H1 (width vs ece_raw), per dataset x condition:")
for k, v in h1.items():
    print(
        f"  {k}: rho={v['spearman_rho']:.3f} p_one_sided={v['p_one_sided']:.4g} "
        f"loo_rho_range=[{v['loo_rho_min']:.3f},{v['loo_rho_max']:.3f}]"
    )
print()
h2 = out["H2_label_smoothing_vs_baseline_post_TS"]
print(
    f"H2: n_pairs={h2['n_pairs']} mean_diff(base-ls)={h2['mean_diff_baseline_minus_ls']:.5f} "
    f"p={h2['p_value_two_sided']:.4g} rank_biserial={h2['rank_biserial_effect_size']:.3f} "
    f"95% CI={h2['bootstrap_ci_95']}"
)
print()
pw = out["H2_power_analysis"]
print(
    f"Power: MDE mean diff={pw['minimum_detectable_mean_diff_ece']:.5f} "
    f"({pw['minimum_detectable_effect_in_sd_units']:.2f} SD units) at n={pw['observed_n_pairs']}"
)
print()
for cond, v in ts_effect.items():
    print(
        f"TS effect [{cond}]: raw={v['mean_ece_raw']:.4f} -> ts={v['mean_ece_ts']:.4f} "
        f"(reduction={v['mean_reduction']:.4f}, p={v['wilcoxon_p']:.3g})"
    )
