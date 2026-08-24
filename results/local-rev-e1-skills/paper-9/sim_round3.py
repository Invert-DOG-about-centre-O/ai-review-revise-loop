"""
Round-3 additions, addressing round3_review.json questions:

(C) W-sensitivity sweep (question 3): does the ECE sign-flip (verbalized vs.
    frequency confidence) survive varying the number of wrong-answer semantic
    clusters W, at extreme heterogeneity, primary sigmoid-logit family?

(D) Grid-rounding check (question 2): the exact analytic self-consistency
    check caches exact_correct_prob() on p rounded to a 0.01 grid for speed.
    Quantify how much of the ~1.2pp Monte-Carlo-vs-analytic residual gap is
    attributable to that rounding, by re-running the analytic computation at
    extreme heterogeneity with NO rounding (exact per-p_i evaluation) and
    comparing to the rounded version.
"""
import json
import time

import numpy as np
from scipy import stats

from sim import stable_seed, sigmoid, logit, ece
from sim_extra import exact_correct_prob

# ---------- (C) W-sensitivity sweep ----------

def run_setting_W(name, beta_a, beta_b, W, N=500, n_reps=30, k_cal=10,
                   over_a=0.55, over_b=0.9, noise_sd=0.35):
    ece_v_list, ece_f_list = [], []
    for rep in range(n_reps):
        seed = stable_seed("wsweep", name, W, rep)
        rng = np.random.default_rng(seed)
        p = rng.beta(beta_a, beta_b, size=N)
        p = 0.55 + (p - p.mean()) * 1.0
        p = np.clip(p, 0.02, 0.98)

        correct = rng.random((N, k_cal)) < p[:, None]
        wrong_cluster = rng.integers(1, W + 1, size=(N, k_cal))
        cluster = np.where(correct, 0, wrong_cluster)
        maxc = cluster.max() + 1
        counts = np.zeros((N, maxc), dtype=int)
        for c in range(maxc):
            counts[:, c] = (cluster == c).sum(axis=1)
        mode_cluster = counts.argmax(axis=1)
        freq_conf = counts.max(axis=1) / k_cal
        sc_correct = (mode_cluster == 0).astype(int)

        verb_conf = sigmoid(over_a * logit(p) + over_b + rng.normal(0, noise_sd, size=N))

        ece_v_list.append(ece(verb_conf, sc_correct))
        ece_f_list.append(ece(freq_conf, sc_correct))
    ece_v = np.array(ece_v_list); ece_f = np.array(ece_f_list)
    t, pval = stats.ttest_rel(ece_v, ece_f)
    return dict(ece_verb=float(ece_v.mean()), ece_freq=float(ece_f.mean()),
                t=float(t), p=float(pval))

def w_sensitivity_check():
    settings = {
        "moderate_heterogeneity": dict(beta_a=4, beta_b=4),
        "extreme_heterogeneity":  dict(beta_a=0.4, beta_b=0.4),
    }
    Ws = [2, 4, 8]
    out = {}
    for W in Ws:
        out[f"W={W}"] = {}
        for sname, sparams in settings.items():
            res = run_setting_W(f"W{W}_{sname}", W=W, **sparams)
            out[f"W={W}"][sname] = res
            flip = "freq wins" if res["ece_freq"] < res["ece_verb"] else "verb wins"
            print(f"[W={W}] {sname}: ECE_verb={res['ece_verb']:.4f} ECE_freq={res['ece_freq']:.4f} "
                  f"({flip}) t={res['t']:.2f} p={res['p']:.2e}")
    return out

# ---------- (D) grid-rounding sensitivity for the analytic check ----------

def analytic_grid_vs_exact(name, beta_a, beta_b, k, W=4, n_reps=30):
    all_p = []
    for rep in range(n_reps):
        seed = stable_seed("uncertainty_sim", name, rep)
        rng = np.random.default_rng(seed)
        p = rng.beta(beta_a, beta_b, size=500)
        p = 0.55 + (p - p.mean()) * 1.0
        p = np.clip(p, 0.02, 0.98)
        all_p.append(p)
    all_p = np.concatenate(all_p)

    # coarse-grid version (as in sim_extra.py, 0.01 grid, memoized)
    cache_coarse = {}
    accs_coarse = np.empty_like(all_p)
    for i, pv in enumerate(all_p):
        key = round(float(pv), 2)
        if key not in cache_coarse:
            cache_coarse[key] = exact_correct_prob(key, k, W)
        accs_coarse[i] = cache_coarse[key]

    # fine-grid version: 0.001 grid, still memoized (10x finer, ~2000 unique
    # keys max instead of ~100) -- if coarse and fine agree, rounding to 0.01
    # is not a meaningful source of the Monte-Carlo-vs-analytic residual.
    cache_fine = {}
    accs_fine = np.empty_like(all_p)
    for i, pv in enumerate(all_p):
        key = round(float(pv), 3)
        if key not in cache_fine:
            cache_fine[key] = exact_correct_prob(key, k, W)
        accs_fine[i] = cache_fine[key]

    return dict(
        coarse_mean=float(accs_coarse.mean()),
        fine_mean=float(accs_fine.mean()),
        n_unique_coarse=len(cache_coarse),
        n_unique_fine=len(cache_fine),
        max_abs_diff_per_point=float(np.max(np.abs(accs_coarse - accs_fine))),
        mean_abs_diff_per_point=float(np.mean(np.abs(accs_coarse - accs_fine))),
    )

def grid_rounding_check():
    out = {}
    for k in (1, 20):
        res = analytic_grid_vs_exact("extreme_heterogeneity", 0.4, 0.4, k)
        out[f"k={k}"] = res
        print(f"[grid-check extreme, k={k}] coarse(0.01)_mean={res['coarse_mean']:.6f} "
              f"fine(0.001)_mean={res['fine_mean']:.6f} "
              f"n_unique coarse/fine={res['n_unique_coarse']}/{res['n_unique_fine']} "
              f"mean_abs_diff_per_point={res['mean_abs_diff_per_point']:.2e} "
              f"max_abs_diff_per_point={res['max_abs_diff_per_point']:.2e}")
    coarse_gain = (out["k=20"]["coarse_mean"] - out["k=1"]["coarse_mean"]) * 100
    fine_gain = (out["k=20"]["fine_mean"] - out["k=1"]["fine_mean"]) * 100
    out["coarse_gain_pp"] = coarse_gain
    out["fine_gain_pp"] = fine_gain
    out["gain_diff_from_rounding_pp"] = coarse_gain - fine_gain
    print(f"[grid-check extreme] gain via 0.01 grid = {coarse_gain:.4f}pp, "
          f"gain via 0.001 grid = {fine_gain:.4f}pp, "
          f"difference attributable to 0.01 vs 0.001 rounding = {coarse_gain - fine_gain:.4f}pp")
    return out

if __name__ == "__main__":
    t0 = time.time()
    print("=== (C) W-sensitivity of the ECE sign-flip ===")
    w_results = w_sensitivity_check()
    print()
    print("=== (D) Grid-rounding contribution to the Monte-Carlo-vs-analytic residual ===")
    grid_results = grid_rounding_check()
    elapsed = time.time() - t0
    print(f"\nRound-3 runtime: {elapsed:.1f}s")
    with open("sim_round3_results.json", "w") as f:
        json.dump({"w_sensitivity": w_results, "grid_rounding_check": grid_results,
                    "runtime_sec": elapsed}, f, indent=2)
    print("Saved sim_round3_results.json")
