"""
Agent-based simulation of human-AI feedback loops in recommender systems.

Studies whether a self-retraining recommender (one that learns from the
engagement it itself generated) amplifies opinion polarization and reduces
collective content diversity, relative to a random baseline, a static
(train-once, never-retrained) recommender, and an adaptive recommender with
a diversity-injection intervention.

All output is deterministic given a seed; results are written to
results.csv and results_summary.json.
"""
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------------------- config -----------------------------------
N_AGENTS = 300
N_ROUNDS = 40
CANDIDATES_PER_ROUND = 500
K_RECS = 5
COLDSTART_ROUNDS = 3          # rounds of random exposure used to fit the first model
RETRAIN_EVERY = 4             # rounds between retrains for the adaptive conditions
DIVERSITY_EPS = 0.3           # fraction of recs replaced by random candidates
ASSIM_ALPHA = 0.08            # opinion moves this fraction of the way to engaged content
ENGAGE_SENS = 4.0             # sensitivity of engagement prob to opinion-item distance
ENGAGE_BASE = -1.0            # base rate (logit) of engagement
OPINION_NOISE = 0.01          # per-round random drift
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ["random", "static", "adaptive", "adaptive_diverse"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def engagement_prob(agent_opinions, item_positions):
    """Elementwise (paired) engagement probability, not a cross product."""
    dist = np.abs(agent_opinions - item_positions)
    logit = ENGAGE_BASE + ENGAGE_SENS * (1.0 - dist)
    return sigmoid(logit)


def features(agent_opinions, item_positions):
    a = agent_opinions
    it = item_positions
    dist = np.abs(a - it)
    return np.stack([a, it, a * it, dist], axis=1)


def bimodality_coefficient(x):
    n = len(x)
    m3 = ((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)
    m4 = ((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-12)
    bc = (m3 ** 2 + 1) / (m4 + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3) + 1e-12))
    return float(bc)


def diversity_entropy(item_positions, bins=10):
    if len(item_positions) == 0:
        return 0.0
    hist, _ = np.histogram(item_positions, bins=bins, range=(-1, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_condition(condition, seed):
    rng = np.random.default_rng(seed)
    opinions = np.clip(rng.normal(0, 0.3, size=N_AGENTS), -1, 1)

    log_feats = []
    log_labels = []
    model = None

    history = []
    all_engaged_positions = []

    for rnd in range(N_ROUNDS):
        candidates = rng.uniform(-1, 1, size=CANDIDATES_PER_ROUND)

        use_random = (condition == "random") or (model is None)
        recs_idx = np.zeros((N_AGENTS, K_RECS), dtype=int)

        if use_random:
            for i in range(N_AGENTS):
                recs_idx[i] = rng.choice(CANDIDATES_PER_ROUND, size=K_RECS, replace=False)
        else:
            X_all = np.zeros((N_AGENTS, CANDIDATES_PER_ROUND))
            feats_flat = features(
                np.repeat(opinions, CANDIDATES_PER_ROUND),
                np.tile(candidates, N_AGENTS),
            )
            scores = model.predict_proba(feats_flat)[:, 1].reshape(N_AGENTS, CANDIDATES_PER_ROUND)
            top_k = np.argpartition(-scores, K_RECS, axis=1)[:, :K_RECS]
            recs_idx = top_k
            if condition == "adaptive_diverse":
                n_replace = int(round(DIVERSITY_EPS * K_RECS))
                if n_replace > 0:
                    for i in range(N_AGENTS):
                        replace_slots = rng.choice(K_RECS, size=n_replace, replace=False)
                        recs_idx[i, replace_slots] = rng.choice(
                            CANDIDATES_PER_ROUND, size=n_replace, replace=False
                        )

        rec_positions = candidates[recs_idx]  # (N_AGENTS, K_RECS)
        opinions_rep = np.repeat(opinions, K_RECS)
        probs = engagement_prob(opinions_rep, rec_positions.reshape(-1)).reshape(N_AGENTS, K_RECS)
        draws = rng.uniform(size=probs.shape)
        engaged = draws < probs

        round_engaged_positions = []
        for i in range(N_AGENTS):
            eng_items = rec_positions[i][engaged[i]]
            for pos in eng_items:
                opinions[i] += ASSIM_ALPHA * (pos - opinions[i])
                round_engaged_positions.append(pos)
        opinions += rng.normal(0, OPINION_NOISE, size=N_AGENTS)
        opinions = np.clip(opinions, -1, 1)

        feats = features(np.repeat(opinions, K_RECS), rec_positions.reshape(-1))
        labels = engaged.reshape(-1).astype(int)
        log_feats.append(feats)
        log_labels.append(labels)
        all_engaged_positions.extend(round_engaged_positions)

        if condition != "random":
            should_train = (model is None and rnd + 1 >= COLDSTART_ROUNDS)
            if condition in ("adaptive", "adaptive_diverse"):
                should_train = should_train or (
                    model is not None and (rnd + 1) % RETRAIN_EVERY == 0
                )
            if condition == "static" and model is not None:
                should_train = False
            if should_train:
                X = np.concatenate(log_feats, axis=0)
                y = np.concatenate(log_labels, axis=0)
                if y.sum() > 5 and y.sum() < len(y) - 5:
                    m = LogisticRegression(max_iter=200)
                    m.fit(X, y)
                    model = m

        history.append(
            {
                "round": rnd,
                "opinion_var": float(np.var(opinions)),
                "opinion_bimodality": bimodality_coefficient(opinions),
                "frac_extreme": float(np.mean(np.abs(opinions) > 0.6)),
                "round_diversity_entropy": diversity_entropy(np.array(round_engaged_positions)),
            }
        )

    overall_diversity = diversity_entropy(np.array(all_engaged_positions))
    return {
        "condition": condition,
        "seed": seed,
        "history": history,
        "final_opinion_var": history[-1]["opinion_var"],
        "final_bimodality": history[-1]["opinion_bimodality"],
        "final_frac_extreme": history[-1]["frac_extreme"],
        "overall_content_diversity": overall_diversity,
        "initial_opinion_var": float(np.var(np.clip(np.random.default_rng(seed).normal(0, 0.3, N_AGENTS), -1, 1))),
    }


def main():
    t0 = time.time()
    all_runs = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            res = run_condition(cond, seed)
            all_runs.append(res)
            print(
                f"cond={cond:17s} seed={seed} "
                f"final_var={res['final_opinion_var']:.4f} "
                f"bimodality={res['final_bimodality']:.3f} "
                f"frac_extreme={res['final_frac_extreme']:.3f} "
                f"diversity={res['overall_content_diversity']:.3f}"
            )

    import csv
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "round", "opinion_var", "opinion_bimodality", "frac_extreme", "round_diversity_entropy"])
        for res in all_runs:
            for h in res["history"]:
                w.writerow([res["condition"], res["seed"], h["round"], h["opinion_var"], h["opinion_bimodality"], h["frac_extreme"], h["round_diversity_entropy"]])

    summary = {}
    for cond in CONDITIONS:
        runs = [r for r in all_runs if r["condition"] == cond]
        summary[cond] = {
            "final_opinion_var_mean": float(np.mean([r["final_opinion_var"] for r in runs])),
            "final_opinion_var_std": float(np.std([r["final_opinion_var"] for r in runs])),
            "final_bimodality_mean": float(np.mean([r["final_bimodality"] for r in runs])),
            "final_bimodality_std": float(np.std([r["final_bimodality"] for r in runs])),
            "final_frac_extreme_mean": float(np.mean([r["final_frac_extreme"] for r in runs])),
            "final_frac_extreme_std": float(np.std([r["final_frac_extreme"] for r in runs])),
            "overall_content_diversity_mean": float(np.mean([r["overall_content_diversity"] for r in runs])),
            "overall_content_diversity_std": float(np.std([r["overall_content_diversity"] for r in runs])),
        }
    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY (mean +/- std over seeds) ===")
    for cond, s in summary.items():
        print(
            f"{cond:17s} var={s['final_opinion_var_mean']:.4f}+/-{s['final_opinion_var_std']:.4f} "
            f"bimodality={s['final_bimodality_mean']:.3f}+/-{s['final_bimodality_std']:.3f} "
            f"frac_extreme={s['final_frac_extreme_mean']:.3f}+/-{s['final_frac_extreme_std']:.3f} "
            f"diversity={s['overall_content_diversity_mean']:.3f}+/-{s['overall_content_diversity_std']:.3f}"
        )
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
