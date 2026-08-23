"""
Algorithmic Sycophancy and Opinion Polarization: simulation code.

Models a population of agents with scalar opinions in [-1, 1]. At each round,
a "recommender" selects one content item (drawn from a fixed content pool
representing the population's opinion distribution) to show each agent. The
agent updates its opinion toward the shown content only if the content is
within its bounded-confidence tolerance (Hegselmann-Krause style), otherwise
it holds (or, under a "backfire" condition, moves slightly away).

Four recommender strategies are compared:
  - random:      uniformly sample content from the pool (no personalization)
  - calibrated:  sample content proportional to the true population opinion
                 distribution (accuracy-faithful, non-personalized)
  - sycophantic: with probability s ("sycophancy strength"), show content
                 that agrees with the agent's current opinion (nearest
                 neighbor in the pool); otherwise fall back to calibrated
  - bridging:    with probability b, deliberately show content from the
                 opposite side of the agent's opinion; otherwise calibrated

We sweep sycophancy strength s in {0.0, 0.25, 0.5, 0.75, 1.0} and compare
against random/calibrated/bridging baselines, tracking population
polarization (variance and bimodality coefficient) and mean absolute opinion
extremity over time.
"""
import json
import time
import numpy as np

RNG_SEED_BASE = 12345
N_AGENTS = 300
N_ROUNDS = 150
CONFIDENCE_EPS = 0.35   # bounded-confidence threshold
LEARNING_RATE = 0.3     # how far an agent moves toward accepted content
BACKFIRE_RATE = 0.05    # small movement AWAY from content outside tolerance
N_SEEDS = 20


def bimodality_coefficient(x):
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    skew = np.mean(((x - m) / s) ** 3)
    kurt = np.mean(((x - m) / s) ** 4) - 3.0
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return bc


def init_opinions(rng, n):
    # bimodal-ish start is avoided; begin from a mild unimodal spread so
    # any polarization observed is *induced* by the recommender dynamics.
    return np.clip(rng.normal(0.0, 0.25, size=n), -1, 1)


def recommend(strategy, opinions, pool, rng, param):
    """Return an array of recommended content values, one per agent."""
    n = len(opinions)
    if strategy == "random":
        return rng.choice(pool, size=n, replace=True)
    if strategy == "calibrated":
        return rng.choice(pool, size=n, replace=True)
    if strategy == "sycophantic":
        s = param
        out = np.empty(n)
        use_syco = rng.random(n) < s
        # sycophantic: nearest-content-to-current-opinion (agreement-seeking)
        for i in range(n):
            if use_syco[i]:
                idx = np.argmin(np.abs(pool - opinions[i]))
                out[i] = pool[idx]
            else:
                out[i] = rng.choice(pool)
        return out
    if strategy == "bridging":
        b = param
        out = np.empty(n)
        use_bridge = rng.random(n) < b
        for i in range(n):
            if use_bridge[i]:
                idx = np.argmin(np.abs(pool - (-opinions[i])))
                out[i] = pool[idx]
            else:
                out[i] = rng.choice(pool)
        return out
    raise ValueError(strategy)


def run_simulation(strategy, param, seed, n_agents=N_AGENTS, n_rounds=N_ROUNDS):
    rng = np.random.default_rng(seed)
    opinions = init_opinions(rng, n_agents)
    history_var = []
    history_bc = []
    history_extremity = []
    for t in range(n_rounds):
        # content pool = current population opinions (agents also act as
        # content sources, i.e. peer content / user-generated posts)
        pool = opinions.copy()
        rec = recommend(strategy, opinions, pool, rng, param)
        diff = rec - opinions
        within = np.abs(diff) <= CONFIDENCE_EPS
        delta = np.where(
            within,
            LEARNING_RATE * diff,
            -BACKFIRE_RATE * np.sign(diff) * np.abs(diff),
        )
        opinions = np.clip(opinions + delta, -1, 1)
        history_var.append(float(np.var(opinions)))
        history_bc.append(float(bimodality_coefficient(opinions)))
        history_extremity.append(float(np.mean(np.abs(opinions))))
    return {
        "final_var": history_var[-1],
        "final_bc": history_bc[-1],
        "final_extremity": history_extremity[-1],
        "var_curve": history_var,
        "bc_curve": history_bc,
        "extremity_curve": history_extremity,
    }


def main():
    t0 = time.time()
    conditions = [
        ("random", 0.0),
        ("calibrated", 0.0),
        ("bridging", 0.5),
        ("sycophantic", 0.0),
        ("sycophantic", 0.25),
        ("sycophantic", 0.5),
        ("sycophantic", 0.75),
        ("sycophantic", 1.0),
    ]
    results = {}
    for strategy, param in conditions:
        key = f"{strategy}_{param}"
        runs = [run_simulation(strategy, param, seed=RNG_SEED_BASE + s)
                for s in range(N_SEEDS)]
        final_vars = np.array([r["final_var"] for r in runs])
        final_bcs = np.array([r["final_bc"] for r in runs])
        final_ext = np.array([r["final_extremity"] for r in runs])
        mean_var_curve = np.mean([r["var_curve"] for r in runs], axis=0)
        mean_bc_curve = np.mean([r["bc_curve"] for r in runs], axis=0)
        mean_ext_curve = np.mean([r["extremity_curve"] for r in runs], axis=0)
        results[key] = {
            "strategy": strategy,
            "param": param,
            "final_var_mean": float(final_vars.mean()),
            "final_var_std": float(final_vars.std(ddof=1)),
            "final_bc_mean": float(final_bcs.mean()),
            "final_bc_std": float(final_bcs.std(ddof=1)),
            "final_extremity_mean": float(final_ext.mean()),
            "final_extremity_std": float(final_ext.std(ddof=1)),
            "var_curve": mean_var_curve.tolist(),
            "bc_curve": mean_bc_curve.tolist(),
            "extremity_curve": mean_ext_curve.tolist(),
            "n_seeds": N_SEEDS,
        }
        print(f"{key:>22s}: var={final_vars.mean():.4f}+-{final_vars.std(ddof=1):.4f}  "
              f"bc={final_bcs.mean():.4f}+-{final_bcs.std(ddof=1):.4f}  "
              f"extremity={final_ext.mean():.4f}+-{final_ext.std(ddof=1):.4f}")

    # simple two-sample t-test (Welch) between sycophantic(1.0) and calibrated
    from math import sqrt
    def welch_t(a_mean, a_std, a_n, b_mean, b_std, b_n):
        se = sqrt(a_std**2 / a_n + b_std**2 / b_n)
        t = (a_mean - b_mean) / se if se > 0 else float("nan")
        return t, se

    syco1 = results["sycophantic_1.0"]
    calib = results["calibrated_0.0"]
    t_var, se_var = welch_t(syco1["final_var_mean"], syco1["final_var_std"], N_SEEDS,
                             calib["final_var_mean"], calib["final_var_std"], N_SEEDS)
    print(f"\nWelch t-stat (final variance, sycophantic@1.0 vs calibrated): t={t_var:.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal simulation time: {elapsed:.1f}s")

    with open("results.json", "w") as f:
        json.dump({
            "results": results,
            "welch_t_var_syco1_vs_calibrated": t_var,
            "elapsed_seconds": elapsed,
            "config": {
                "n_agents": N_AGENTS, "n_rounds": N_ROUNDS,
                "confidence_eps": CONFIDENCE_EPS, "learning_rate": LEARNING_RATE,
                "backfire_rate": BACKFIRE_RATE, "n_seeds": N_SEEDS,
            }
        }, f, indent=2)
    print("Saved results.json")


if __name__ == "__main__":
    main()
