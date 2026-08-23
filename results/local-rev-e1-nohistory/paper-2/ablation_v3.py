"""
Round-2 revision experiments addressing round-2 review:
  (C) Drifting external corpus: the fixed corpus in extra_experiments.py is
      static; here the corpus undergoes its own slow random-walk drift each
      round, fully decoupled from agent opinions (reviewer question 2), to
      check whether freezing depends on the corpus being perfectly static.
  (D) Alternate, non-magnitude-keyed update rule: sycophantic(1.0) forces
      |r_i - x_i| ~ 0, so the standard update (delta proportional to
      r_i - x_i) trivially yields near-zero movement. Here agents instead
      move by a FIXED step size in the *sign* direction of (r_i - x_i),
      decoupling update magnitude from recommendation distance, to test
      whether freezing is a real property of closest-match selection or an
      artifact of magnitude-scaled updates (reviewer question 3).
"""
import json
import time
import numpy as np
from scipy import stats

RNG_SEED_BASE = 12345
N_AGENTS = 300
N_ROUNDS = 150
CONFIDENCE_EPS = 0.35
LEARNING_RATE = 0.3
BACKFIRE_RATE = 0.05
FIXED_STEP = 0.03          # fixed step size for the sign-based rule
N_SEEDS = 10
DRIFT_SIGMA = 0.01          # per-round corpus drift std


def bimodality_coefficient(x):
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    skew = np.mean(((x - m) / s) ** 3)
    kurt = np.mean(((x - m) / s) ** 4) - 3.0
    return (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def init_opinions(rng, n):
    return np.clip(rng.normal(0.0, 0.25, size=n), -1, 1)


def recommend(strategy, opinions, pool, rng, param):
    n = len(opinions)
    if strategy == "calibrated":
        return rng.choice(pool, size=n, replace=True)
    if strategy == "sycophantic":
        s = param
        out = np.empty(n)
        use_syco = rng.random(n) < s
        for i in range(n):
            if use_syco[i]:
                idx = np.argmin(np.abs(pool - opinions[i]))
                out[i] = pool[idx]
            else:
                out[i] = rng.choice(pool)
        return out
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# (C) Drifting external corpus: corpus is decoupled from agent opinions AND
# not static -- it random-walks on its own each round.
# ---------------------------------------------------------------------------
def run_drifting_corpus(strategy, param, seed, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                         drift_sigma=DRIFT_SIGMA):
    rng = np.random.default_rng(seed)
    opinions = init_opinions(rng, n_agents)
    corpus = init_opinions(rng, n_agents)
    for t in range(n_rounds):
        corpus = np.clip(corpus + rng.normal(0.0, drift_sigma, size=n_agents), -1, 1)
        rec = recommend(strategy, opinions, corpus, rng, param)
        diff = rec - opinions
        within = np.abs(diff) <= CONFIDENCE_EPS
        delta = np.where(within, LEARNING_RATE * diff,
                          -BACKFIRE_RATE * np.sign(diff) * np.abs(diff))
        opinions = np.clip(opinions + delta, -1, 1)
    return {
        "final_var": float(np.var(opinions)),
        "final_bc": float(bimodality_coefficient(opinions)),
        "final_extremity": float(np.mean(np.abs(opinions))),
    }


# ---------------------------------------------------------------------------
# (D) Sign-based fixed-step update: movement magnitude no longer scales with
# |r_i - x_i|, so it cannot be trivially near-zero just because r_i ~ x_i.
# ---------------------------------------------------------------------------
def run_sign_update(strategy, param, seed, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                     fixed_step=FIXED_STEP):
    rng = np.random.default_rng(seed)
    opinions = init_opinions(rng, n_agents)
    for t in range(n_rounds):
        pool = opinions.copy()
        rec = recommend(strategy, opinions, pool, rng, param)
        diff = rec - opinions
        within = np.abs(diff) <= CONFIDENCE_EPS
        step = fixed_step * np.sign(diff)  # fixed magnitude, direction only
        delta = np.where(within, step, -BACKFIRE_RATE * np.sign(diff) * np.abs(diff))
        opinions = np.clip(opinions + delta, -1, 1)
    return {
        "final_var": float(np.var(opinions)),
        "final_bc": float(bimodality_coefficient(opinions)),
        "final_extremity": float(np.mean(np.abs(opinions))),
    }


def summarize(runs, key_metrics=("final_var", "final_bc", "final_extremity")):
    out = {}
    for m in key_metrics:
        v = np.array([r[m] for r in runs])
        out[f"{m}_mean"] = float(v.mean())
        out[f"{m}_std"] = float(v.std(ddof=1))
    return out


def main():
    t0 = time.time()
    results = {"drifting_corpus": {}, "sign_update": {}}

    print("Part C: drifting external corpus (decoupled AND non-static)")
    for strategy, param in [("calibrated", 0.0), ("sycophantic", 1.0)]:
        key = f"{strategy}_{param}"
        runs = [run_drifting_corpus(strategy, param, seed=RNG_SEED_BASE + s) for s in range(N_SEEDS)]
        results["drifting_corpus"][key] = summarize(runs)
        r = results["drifting_corpus"][key]
        print(f"  {key:>18s}: var={r['final_var_mean']:.4f}+-{r['final_var_std']:.4f}")

    dc = results["drifting_corpus"]
    t_dc, p_dc = stats.ttest_ind(
        [r["final_var"] for r in [run_drifting_corpus("sycophantic", 1.0, seed=RNG_SEED_BASE + s) for s in range(N_SEEDS)]],
        [r["final_var"] for r in [run_drifting_corpus("calibrated", 0.0, seed=RNG_SEED_BASE + s) for s in range(N_SEEDS)]],
        equal_var=False)
    results["drifting_corpus"]["welch_t_var"] = float(t_dc)
    results["drifting_corpus"]["welch_p_var"] = float(p_dc)
    print(f"  Welch t (var, syco1 vs calibrated, drifting corpus): t={t_dc:.3f}, p={p_dc:.2e}")

    print("\nPart D: sign-based fixed-step update rule (decoupled from |r_i - x_i| magnitude)")
    syco_runs, calib_runs = [], []
    for strategy, param in [("calibrated", 0.0), ("sycophantic", 1.0)]:
        key = f"{strategy}_{param}"
        runs = [run_sign_update(strategy, param, seed=RNG_SEED_BASE + s) for s in range(N_SEEDS)]
        if strategy == "sycophantic":
            syco_runs = runs
        else:
            calib_runs = runs
        results["sign_update"][key] = summarize(runs)
        r = results["sign_update"][key]
        print(f"  {key:>18s}: var={r['final_var_mean']:.4f}+-{r['final_var_std']:.4f}  "
              f"bc={r['final_bc_mean']:.4f}+-{r['final_bc_std']:.4f}")

    t_su, p_su = stats.ttest_ind([r["final_var"] for r in syco_runs],
                                  [r["final_var"] for r in calib_runs], equal_var=False)
    results["sign_update"]["welch_t_var"] = float(t_su)
    results["sign_update"]["welch_p_var"] = float(p_su)
    print(f"  Welch t (var, syco1 vs calibrated, sign-update rule): t={t_su:.3f}, p={p_su:.2e}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    results["elapsed_seconds"] = elapsed
    results["config"] = {"n_seeds": N_SEEDS, "drift_sigma": DRIFT_SIGMA, "fixed_step": FIXED_STEP}

    with open("ablation_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ablation_v3_results.json")


if __name__ == "__main__":
    main()
