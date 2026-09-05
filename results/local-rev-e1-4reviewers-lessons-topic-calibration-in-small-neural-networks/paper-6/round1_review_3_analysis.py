"""
Analysis of round1_review_3_raw_results.json produced by experiment.py.

Pre-registered (decided before looking at results) tests:
  H1: ECE before temperature scaling (ece_pre) is monotonically related to
      log2(width), tested via Spearman correlation with a label-permutation
      null (shuffle width labels within dataset, 10000 permutations),
      per dataset. Two-sided.
  H2: The optimal temperature T* is monotonically related to log2(width),
      same test design as H1.
  H3: Whether temperature scaling's raw benefit (reduction = ece_pre -
      ece_post) is associated with ece_pre is tested with a PERMUTATION null
      that shuffles ece_post independently of ece_pre (not a plain Pearson
      p-value), because reduction is arithmetically derived from ece_pre and
      a naive correlation is guaranteed to be biased (see lessons on
      pre/delta correlation traps).
  Multiplicity: 3 primary hypotheses x 2 datasets = 6 primary tests.
  We apply Bonferroni correction (alpha = 0.05 / 6 = 0.0083) and also report
  raw p-values so readers can see both.
Secondary / robustness (not corrected, explicitly exploratory):
  - Leave-one-width-out check on H1/H2 (drop width=2 and drop width=256
    separately, recompute Spearman rho).
  - Does TS ever leave the model worse (ece_post > ece_pre)? tabulated.
  - Convergence-warning audit cross-tabulated with width.
"""
import json
import numpy as np
from scipy import stats

rng_global = np.random.RandomState(777)  # fixed seed for all permutation nulls

with open("round1_review_3_raw_results.json") as f:
    data = json.load(f)

results = data["results"]
warn_log = data["warn_log"]
widths = data["config"]["widths"]

import collections
by_ds = collections.defaultdict(list)
for r in results:
    by_ds[r["dataset"]].append(r)


def permutation_spearman(x, y, n_perm=10000, seed=0):
    rho_obs, _ = stats.spearmanr(x, y)
    rng = np.random.RandomState(seed)
    n = len(x)
    count = 0
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        r, _ = stats.spearmanr(x, yp)
        perm_rhos[i] = r
        if abs(r) >= abs(rho_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return rho_obs, p, perm_rhos


def permutation_corr_pre_reduction(ece_pre, ece_post, n_perm=10000, seed=0):
    """Test corr(ece_pre, reduction) against a null where ece_post is
    shuffled independently of ece_pre (breaks the arithmetic pre/delta
    coupling), per the pre/delta correlation-trap lesson."""
    reduction = np.array(ece_pre) - np.array(ece_post)
    obs_r, _ = stats.pearsonr(ece_pre, reduction)
    rng = np.random.RandomState(seed)
    n = len(ece_pre)
    ece_pre = np.array(ece_pre)
    ece_post = np.array(ece_post)
    count = 0
    null_rs = np.empty(n_perm)
    for i in range(n_perm):
        post_shuf = rng.permutation(ece_post)
        red_shuf = ece_pre - post_shuf
        r, _ = stats.pearsonr(ece_pre, red_shuf)
        null_rs[i] = r
        if abs(r) >= abs(obs_r):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs_r, p, null_rs


out = {"per_cell_summary": {}, "primary_tests": {}, "robustness": {}, "warn_audit": {}, "config": data["config"]}

# ---- per-cell summary (mean +/- 95% CI via normal approx, n=N_SEEDS) ----
for ds, rows in by_ds.items():
    out["per_cell_summary"][ds] = {}
    for w in widths:
        cell = [r for r in rows if r["width"] == w]
        def summ(key):
            vals = np.array([c[key] for c in cell])
            m = float(vals.mean())
            se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            return {"mean": m, "se": se, "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se, "n": len(vals)}
        out["per_cell_summary"][ds][str(w)] = {
            "acc": summ("acc"),
            "ece_pre": summ("ece_pre"),
            "ece_post": summ("ece_post"),
            "T_star": summ("T_star"),
            "nll_pre": summ("nll_pre"),
            "nll_post": summ("nll_post"),
            "brier_pre": summ("brier_pre"),
            "brier_post": summ("brier_post"),
        }

# ---- primary tests (H1, H2, H3) per dataset ----
alpha_bonf = 0.05 / 6
for ds, rows in by_ds.items():
    log2w = np.array([np.log2(r["width"]) for r in rows])
    ece_pre = np.array([r["ece_pre"] for r in rows])
    ece_post = np.array([r["ece_post"] for r in rows])
    T_star = np.array([r["T_star"] for r in rows])

    rho1, p1, _ = permutation_spearman(log2w, ece_pre, seed=1)
    rho2, p2, _ = permutation_spearman(log2w, T_star, seed=2)
    r3, p3, _ = permutation_corr_pre_reduction(ece_pre, ece_post, seed=3)

    out["primary_tests"][ds] = {
        "H1_width_vs_ece_pre": {"spearman_rho": rho1, "perm_p": p1, "sig_bonf": p1 < alpha_bonf, "n": len(rows)},
        "H2_width_vs_Tstar": {"spearman_rho": rho2, "perm_p": p2, "sig_bonf": p2 < alpha_bonf, "n": len(rows)},
        "H3_ecepre_vs_reduction_permnull": {"pearson_r": r3, "perm_p": p3, "sig_bonf": p3 < alpha_bonf, "n": len(rows)},
    }

# naive (unpermuted-null) Pearson for H3 for comparison, to show the inflation
for ds, rows in by_ds.items():
    ece_pre = np.array([r["ece_pre"] for r in rows])
    ece_post = np.array([r["ece_post"] for r in rows])
    reduction = ece_pre - ece_post
    naive_r, naive_p = stats.pearsonr(ece_pre, reduction)
    pre_post_r, pre_post_p = stats.pearsonr(ece_pre, ece_post)
    out["primary_tests"][ds]["H3_naive_pearson_pre_vs_reduction"] = {"r": naive_r, "p": naive_p}
    out["primary_tests"][ds]["diagnostic_pre_vs_post_corr"] = {"r": pre_post_r, "p": pre_post_p}

# ---- robustness: leave-one-width-out for H1 and H2 ----
for ds, rows in by_ds.items():
    out["robustness"][ds] = {}
    for drop_w in [min(widths), max(widths)]:
        sub = [r for r in rows if r["width"] != drop_w]
        log2w = np.array([np.log2(r["width"]) for r in sub])
        ece_pre = np.array([r["ece_pre"] for r in sub])
        T_star = np.array([r["T_star"] for r in sub])
        rho1, p1, _ = permutation_spearman(log2w, ece_pre, seed=11)
        rho2, p2, _ = permutation_spearman(log2w, T_star, seed=12)
        out["robustness"][ds][f"drop_width_{drop_w}"] = {
            "H1_rho": rho1, "H1_p": p1, "H2_rho": rho2, "H2_p": p2, "n": len(sub),
        }

# ---- does TS ever hurt? ----
for ds, rows in by_ds.items():
    worse = sum(1 for r in rows if r["ece_post"] > r["ece_pre"])
    out["robustness"].setdefault(ds, {})["ts_made_worse_count"] = {"count": worse, "n": len(rows), "frac": worse / len(rows)}

# ---- power: minimum detectable rho given n, alpha=0.0083, power=0.8 (approx via Fisher z) ----
def min_detectable_rho(n, alpha=alpha_bonf, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    z_r = (z_alpha + z_power) / np.sqrt(n - 3)
    return float(np.tanh(z_r))

for ds, rows in by_ds.items():
    out["primary_tests"][ds]["power_min_detectable_rho"] = min_detectable_rho(len(rows))

# ---- warning audit cross-tab ----
warn_by_cell = collections.Counter((w["dataset"], w["width"]) for w in warn_log)
out["warn_audit"] = {f"{ds}_w{w}": c for (ds, w), c in warn_by_cell.items()}
out["warn_audit"]["total_nonconverged"] = len(warn_log)
out["warn_audit"]["total_runs"] = len(results)

with open("round1_review_3_analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)

# ---- print concise summary ----
print("=== PRIMARY TESTS (Bonferroni alpha = %.4f) ===" % alpha_bonf)
for ds in out["primary_tests"]:
    print(f"-- {ds} --")
    for k, v in out["primary_tests"][ds].items():
        print(" ", k, v)
print("\n=== ROBUSTNESS ===")
for ds in out["robustness"]:
    print(f"-- {ds} --")
    for k, v in out["robustness"][ds].items():
        print(" ", k, v)
print("\n=== WARN AUDIT ===", out["warn_audit"])
