"""
Additional analyses for the revision, addressing round-1 review questions:

(A) Analytic/exact check of the self-consistency shrinkage: for each question's
    true p_i, compute the EXACT probability that the correct cluster is the
    (tie-broken-to-correct) mode of k multinomial draws over {correct, W wrong
    clusters}, via full enumeration -- no Monte Carlo sampling noise. Average
    this exact per-question probability over the same p_i pools used in
    sim.py (same seeds) and compare to the Monte-Carlo simulated accuracy.
    If they match closely, the shrinkage is a deterministic function of the
    p_i marginal distribution (mechanical), not a noisy discovery.

(B) A qualitatively different verbalized-confidence functional family
    (clipped-linear-in-p instead of sigmoid-of-logit(p)) to check whether the
    ECE sign-flip survives a change of functional family, not just a change
    of (a,b) within the same sigmoid-logit family.
"""
import itertools
import json
import math
import time

import numpy as np
from scipy import stats

from sim import stable_seed, simulate_replicate, ece, sigmoid, logit

# ---------- (A) exact multinomial majority-vote accuracy ----------

def exact_correct_prob(p, k, W):
    """Exact P(correct cluster is the argmax, ties broken to correct=index0)
    for k iid draws over categories {correct (prob p), wrong_1..wrong_W (prob (1-p)/W each)}."""
    q = (1 - p) / W
    probs = [p] + [q] * W
    total = 0.0
    # enumerate all compositions of k into W+1 nonnegative parts
    for counts in itertools.product(range(k + 1), repeat=W):
        if sum(counts) > k:
            continue
        c0 = k - sum(counts)
        full = (c0,) + counts
        # tie-break: argmax picks first index attaining max -> correct wins ties
        if c0 >= max(full):
            # multinomial pmf
            log_pmf = (math.lgamma(k + 1)
                       - sum(math.lgamma(c + 1) for c in full)
                       + sum(c * np.log(pr) for c, pr in zip(full, probs) if c > 0))
            total += np.exp(log_pmf)
    return total

def exact_accuracy_grid(k, W, grid):
    return {round(float(p), 6): exact_correct_prob(p, k, W) for p in grid}

def analytic_gain_check():
    settings = {
        "low_heterogeneity":      dict(beta_a=20, beta_b=20),
        "moderate_heterogeneity": dict(beta_a=4, beta_b=4),
        "mild_heterogeneity":     dict(beta_a=1, beta_b=1),
        "extreme_heterogeneity":  dict(beta_a=0.4, beta_b=0.4),
    }
    W = 4
    ks = (1, 20)
    results = {}
    for name, params in settings.items():
        # collect all p_i across 30 replicates, same seeds as sim.py's run_setting
        all_p = []
        for rep in range(30):
            seed = stable_seed("uncertainty_sim", name, rep)
            rng = np.random.default_rng(seed)
            p = rng.beta(params["beta_a"], params["beta_b"], size=500)
            p = 0.55 + (p - p.mean()) * 1.0
            p = np.clip(p, 0.02, 0.98)
            all_p.append(p)
        all_p = np.concatenate(all_p)
        # round p to a coarse grid to memoize the expensive exact computation
        grid_res = {}
        for k in ks:
            cache = {}
            accs = np.empty_like(all_p)
            for i, pv in enumerate(all_p):
                key = round(float(pv), 2)
                if key not in cache:
                    cache[key] = exact_correct_prob(key, k, W)
                accs[i] = cache[key]
            grid_res[k] = float(accs.mean())
        analytic_gain = (grid_res[20] - grid_res[1]) * 100
        results[name] = {
            "analytic_acc_k1": grid_res[1],
            "analytic_acc_k20": grid_res[20],
            "analytic_gain_pp": analytic_gain,
        }
        print(f"[analytic] {name}: acc@k1={grid_res[1]:.4f} acc@k20={grid_res[20]:.4f} "
              f"gain={analytic_gain:.2f}pp")
    return results

# ---------- (B) alternative functional family for verbalized confidence ----------

def clipped_linear_conf(p, slope, shift, noise_sd, rng):
    raw = slope * p + shift + rng.normal(0, noise_sd, size=len(p))
    return np.clip(raw, 1e-6, 1 - 1e-6)

def run_alt_family(name, beta_a, beta_b, slope, shift, noise_sd, N=500, n_reps=30, k_cal=10, W=4):
    ece_v_list, ece_f_list = [], []
    for rep in range(n_reps):
        seed = stable_seed("altfamily", name, rep)
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

        verb_conf = clipped_linear_conf(p, slope, shift, noise_sd, rng)

        ece_v_list.append(ece(verb_conf, sc_correct))
        ece_f_list.append(ece(freq_conf, sc_correct))
    ece_v = np.array(ece_v_list); ece_f = np.array(ece_f_list)
    t, pval = stats.ttest_rel(ece_v, ece_f)
    return dict(ece_verb=float(ece_v.mean()), ece_freq=float(ece_f.mean()),
                t=float(t), p=float(pval))

def alt_family_check():
    # clipped-linear family: conf = clip(slope*p + shift + noise, 0, 1)
    # slope>1 with shift<0 stretches near p=0.55 and clips hard at boundaries --
    # a qualitatively different saturation behavior from sigmoid(logit(p)),
    # which compresses smoothly. Calibrated so mean overconfidence direction matches sim.py's.
    settings = {
        "moderate_heterogeneity": dict(beta_a=4, beta_b=4),
        "extreme_heterogeneity":  dict(beta_a=0.4, beta_b=0.4),
    }
    # slope<1 = compressive (insensitive to true difficulty), slope=1 = neutral,
    # slope>1 = expansive. Same noise_sd and shift-centering logic throughout.
    slope_configs = {
        "compressive_slope0.6":     dict(slope=0.6, shift=0.5,  noise_sd=0.30),
        "mildly_compressive_slope1.0": dict(slope=1.0, shift=0.35, noise_sd=0.30),
        "expansive_slope1.3":       dict(slope=1.3, shift=0.15, noise_sd=0.30),
    }
    out = {}
    for cfg_name, cfg in slope_configs.items():
        out[cfg_name] = {}
        for sname, sparams in settings.items():
            res = run_alt_family(f"{cfg_name}_{sname}", **cfg, **sparams)
            out[cfg_name][sname] = res
            print(f"[alt-family clipped-linear {cfg_name}] {sname}: ECE_verb={res['ece_verb']:.4f} "
                  f"ECE_freq={res['ece_freq']:.4f} t={res['t']:.2f} p={res['p']:.2e}")
    return out

if __name__ == "__main__":
    t0 = time.time()
    print("=== (A) Exact analytic self-consistency gain (no Monte Carlo sampling noise) ===")
    analytic = analytic_gain_check()
    print()
    print("=== (B) ECE sign-flip under a qualitatively different verbalized-confidence family ===")
    alt = alt_family_check()
    elapsed = time.time() - t0
    print(f"\nExtra runtime: {elapsed:.1f}s")
    with open("sim_extra_results.json", "w") as f:
        json.dump({"analytic_gain_check": analytic, "alt_family_check": alt,
                    "runtime_sec": elapsed}, f, indent=2)
    print("Saved sim_extra_results.json")
