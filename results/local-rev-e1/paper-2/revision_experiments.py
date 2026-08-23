"""
Revision experiments addressing round-1 review:
 (1) Paired significance tests (matched seeds) for static-vs-adaptive and the
     two ablation sweeps, instead of eyeballing mean +/- std.
 (2) A genuine concept-drift condition (the ground-truth engagement target
     shifts over time) to test the paper's own conjecture that retraining
     cadence should matter more under drift.
 (3) A misspecified/low-capacity recommender (drops the product and distance
     features, so it cannot represent the ground-truth engagement function)
     to test whether the retraining-null-result is an artifact of giving the
     recommender exactly the features needed to recover ground truth from a
     short cold start.
"""
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import simulate as sim

SEEDS3 = [0, 1, 2]

# ---------- (1) paired significance tests on existing results ----------

def paired_tests_from_results_csv():
    df = pd.read_csv("results.csv")
    final = df[df["round"] == sim.N_ROUNDS - 1]
    out = {}

    def paired(cond_a, cond_b, col="opinion_var"):
        a = final[final.condition == cond_a].sort_values("seed")[col].values
        b = final[final.condition == cond_b].sort_values("seed")[col].values
        t, p = stats.ttest_rel(a, b)
        diff = a - b
        sd = diff.std(ddof=1)
        d = float(diff.mean() / sd) if sd > 0 else float("nan")
        return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "t": float(t), "p": float(p), "n": len(a), "cohens_d": d}

    out["static_vs_adaptive_var"] = paired("static", "adaptive")
    out["static_vs_adaptive_diversity"] = paired("static", "adaptive", "round_diversity_entropy")
    out["static_vs_random_var"] = paired("static", "random")
    return out


# ---------- (2) concept drift: shift the ground-truth similarity target ----

DRIFT_RATE = 0.01  # per-round shift of the "ideal" content offset


def engagement_prob_drift(agent_opinions, item_positions, rnd):
    shift = DRIFT_RATE * rnd
    dist = np.abs(agent_opinions - (item_positions + shift))
    logit = sim.ENGAGE_BASE + sim.ENGAGE_SENS * (1.0 - dist)
    return sim.sigmoid(logit)


def run_condition_drift(condition, seed, retrain_every):
    rng = np.random.default_rng(seed)
    opinions = np.clip(rng.normal(0, 0.3, size=sim.N_AGENTS), -1, 1)
    log_feats, log_labels = [], []
    model = None

    for rnd in range(sim.N_ROUNDS):
        candidates = rng.uniform(-1, 1, size=sim.CANDIDATES_PER_ROUND)
        use_random = (model is None)
        if use_random:
            recs_idx = np.array([rng.choice(sim.CANDIDATES_PER_ROUND, size=sim.K_RECS, replace=False)
                                  for _ in range(sim.N_AGENTS)])
        else:
            feats_flat = sim.features(np.repeat(opinions, sim.CANDIDATES_PER_ROUND),
                                       np.tile(candidates, sim.N_AGENTS))
            scores = model.predict_proba(feats_flat)[:, 1].reshape(sim.N_AGENTS, sim.CANDIDATES_PER_ROUND)
            recs_idx = np.argpartition(-scores, sim.K_RECS, axis=1)[:, :sim.K_RECS]

        rec_positions = candidates[recs_idx]
        opinions_rep = np.repeat(opinions, sim.K_RECS)
        probs = engagement_prob_drift(opinions_rep, rec_positions.reshape(-1), rnd).reshape(sim.N_AGENTS, sim.K_RECS)
        draws = rng.uniform(size=probs.shape)
        engaged = draws < probs

        for i in range(sim.N_AGENTS):
            eng_items = rec_positions[i][engaged[i]]
            for pos in eng_items:
                opinions[i] += sim.ASSIM_ALPHA * (pos - opinions[i])
        opinions += rng.normal(0, sim.OPINION_NOISE, size=sim.N_AGENTS)
        opinions = np.clip(opinions, -1, 1)

        feats = sim.features(np.repeat(opinions, sim.K_RECS), rec_positions.reshape(-1))
        labels = engaged.reshape(-1).astype(int)
        log_feats.append(feats)
        log_labels.append(labels)

        should_train = (model is None and rnd + 1 >= sim.COLDSTART_ROUNDS)
        if condition == "adaptive":
            should_train = should_train or (model is not None and (rnd + 1) % retrain_every == 0)
        if condition == "static" and model is not None:
            should_train = False
        if should_train:
            X = np.concatenate(log_feats, axis=0)
            y = np.concatenate(log_labels, axis=0)
            if y.sum() > 5 and y.sum() < len(y) - 5:
                m = LogisticRegression(max_iter=200)
                m.fit(X, y)
                model = m

    return float(np.var(opinions))


def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def run_drift_experiment():
    # NOTE (round-2 review): with N_ROUNDS=40, retrain_every=40 means
    # (rnd+1) % 40 == 0 fires only at rnd=39, the very last round -- that
    # refit is never actually used to generate a recommendation, so it was
    # mechanically identical to "static" (confirmed: identical raw seed
    # values). We replace it with retrain_every=20, which fires mid-run
    # (at rnd=19) and is actually used for rounds 20-39.
    rows = []
    for condition, freq in [("static", None), ("adaptive", 1), ("adaptive", 20)]:
        vals = [run_condition_drift(condition, s, freq if freq else 4) for s in SEEDS3]
        label = f"{condition}" + (f"_retrain{freq}" if freq else "")
        rows.append({"label": label, "final_var_mean": float(np.mean(vals)), "final_var_std": float(np.std(vals)), "raw": vals})
        print(f"[drift] {label:22s} var={np.mean(vals):.4f}+/-{np.std(vals):.4f}")
    t, p = stats.ttest_rel(rows[1]["raw"], rows[2]["raw"])
    d = cohens_d_paired(rows[1]["raw"], rows[2]["raw"])
    return {"rows": rows, "retrain1_vs_retrain20_paired_t": float(t), "retrain1_vs_retrain20_paired_p": float(p),
            "retrain1_vs_retrain20_cohens_d": d}


# ---------- (3) misspecified recommender (drops product & distance feats) --

def features_misspec(agent_opinions, item_positions):
    a = agent_opinions
    it = item_positions
    return np.stack([a, it], axis=1)  # linear-only, cannot represent |a-it|


def features_no_dist(agent_opinions, item_positions):
    a, it = agent_opinions, item_positions
    return np.stack([a, it, a * it], axis=1)  # has product, missing |a-it|


def features_no_product(agent_opinions, item_positions):
    a, it = agent_opinions, item_positions
    return np.stack([a, it, np.abs(a - it)], axis=1)  # has |a-it|, missing product


def run_condition_misspec(condition, seed, feat_fn=features_misspec):
    rng = np.random.default_rng(seed)
    opinions = np.clip(rng.normal(0, 0.3, size=sim.N_AGENTS), -1, 1)
    log_feats, log_labels = [], []
    model = None

    for rnd in range(sim.N_ROUNDS):
        candidates = rng.uniform(-1, 1, size=sim.CANDIDATES_PER_ROUND)
        use_random = (model is None)
        if use_random:
            recs_idx = np.array([rng.choice(sim.CANDIDATES_PER_ROUND, size=sim.K_RECS, replace=False)
                                  for _ in range(sim.N_AGENTS)])
        else:
            feats_flat = feat_fn(np.repeat(opinions, sim.CANDIDATES_PER_ROUND),
                                  np.tile(candidates, sim.N_AGENTS))
            scores = model.predict_proba(feats_flat)[:, 1].reshape(sim.N_AGENTS, sim.CANDIDATES_PER_ROUND)
            recs_idx = np.argpartition(-scores, sim.K_RECS, axis=1)[:, :sim.K_RECS]

        rec_positions = candidates[recs_idx]
        opinions_rep = np.repeat(opinions, sim.K_RECS)
        probs = sim.engagement_prob(opinions_rep, rec_positions.reshape(-1)).reshape(sim.N_AGENTS, sim.K_RECS)
        draws = rng.uniform(size=probs.shape)
        engaged = draws < probs

        for i in range(sim.N_AGENTS):
            eng_items = rec_positions[i][engaged[i]]
            for pos in eng_items:
                opinions[i] += sim.ASSIM_ALPHA * (pos - opinions[i])
        opinions += rng.normal(0, sim.OPINION_NOISE, size=sim.N_AGENTS)
        opinions = np.clip(opinions, -1, 1)

        feats = feat_fn(np.repeat(opinions, sim.K_RECS), rec_positions.reshape(-1))
        labels = engaged.reshape(-1).astype(int)
        log_feats.append(feats)
        log_labels.append(labels)

        should_train = (model is None and rnd + 1 >= sim.COLDSTART_ROUNDS)
        if condition == "adaptive":
            should_train = should_train or (model is not None and (rnd + 1) % sim.RETRAIN_EVERY == 0)
        if condition == "static" and model is not None:
            should_train = False
        if should_train:
            X = np.concatenate(log_feats, axis=0)
            y = np.concatenate(log_labels, axis=0)
            if y.sum() > 5 and y.sum() < len(y) - 5:
                m = LogisticRegression(max_iter=200)
                m.fit(X, y)
                model = m

    return float(np.var(opinions))


def run_misspec_experiment():
    rows = []
    for condition in ["static", "adaptive"]:
        vals = [run_condition_misspec(condition, s) for s in SEEDS3]
        rows.append({"condition": condition, "final_var_mean": float(np.mean(vals)), "final_var_std": float(np.std(vals)), "raw": vals})
        print(f"[misspec] {condition:10s} var={np.mean(vals):.4f}+/-{np.std(vals):.4f}")
    t, p = stats.ttest_rel(rows[0]["raw"], rows[1]["raw"])
    d = cohens_d_paired(rows[0]["raw"], rows[1]["raw"])
    return {"rows": rows, "static_vs_adaptive_paired_t": float(t), "static_vs_adaptive_paired_p": float(p),
            "static_vs_adaptive_cohens_d": d}


# ---------- (4) partial misspecification sweep (new, round-2 review Q3) ----
# Tests whether the well-specified -> misspecified transition is smooth or a
# sharp threshold, by giving the recommender intermediate feature sets.

FEATURE_SETS = {
    "full": sim.features,                # [a, it, a*it, dist] -- well-specified
    "no_dist": features_no_dist,         # [a, it, a*it] -- missing |a-it|
    "no_product": features_no_product,   # [a, it, dist] -- missing a*it (has the true rule's feature)
    "linear_only": features_misspec,     # [a, it] -- fully misspecified
}


def run_partial_misspec_sweep():
    rows = []
    for name, feat_fn in FEATURE_SETS.items():
        adaptive_vals = [run_condition_misspec("adaptive", s, feat_fn) for s in SEEDS3]
        static_vals = [run_condition_misspec("static", s, feat_fn) for s in SEEDS3]
        rows.append({
            "feature_set": name,
            "adaptive_var_mean": float(np.mean(adaptive_vals)), "adaptive_var_std": float(np.std(adaptive_vals)),
            "static_var_mean": float(np.mean(static_vals)), "static_var_std": float(np.std(static_vals)),
        })
        print(f"[partial-misspec] {name:12s} static={np.mean(static_vals):.4f}  adaptive={np.mean(adaptive_vals):.4f}")
    return {"rows": rows}


def main():
    t0 = time.time()
    print("=== (1) Paired significance tests on main results ===")
    sig = paired_tests_from_results_csv()
    for k, v in sig.items():
        print(f"{k}: {v}")

    print("\n=== (2) Concept-drift experiment ===")
    drift = run_drift_experiment()

    print("\n=== (3) Misspecified-recommender experiment ===")
    misspec = run_misspec_experiment()

    print("\n=== (4) Partial-misspecification sweep ===")
    partial = run_partial_misspec_sweep()

    with open("revision_results.json", "w") as f:
        json.dump({"paired_significance": sig, "drift_experiment": drift, "misspec_experiment": misspec,
                    "partial_misspec_sweep": partial}, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
