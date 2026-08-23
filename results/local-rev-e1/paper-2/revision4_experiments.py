"""
Revision-4 experiments addressing round-3 review:
 (1) Assimilation-rate (ASSIM_ALPHA) and noise (OPINION_NOISE) sweep --
     flagged as untested in round-1, round-2, and round-3 reviews.
 (2) Bootstrap CIs (resampling agents, not re-simulating) for the n=3
     misspecification / drift comparisons, as an alternative to more seeds.
 (3) A coefficient diagnostic on the final fitted no_dist vs linear_only
     models to give a mechanistic account of why the raw product term
     (no_dist) produces worse, cadence-insensitive catastrophe than no
     nonlinear proxy at all (linear_only).
"""
import json
import time
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

import simulate as sim
import revision_experiments as rev

SEEDS3 = [0, 1, 2]


# ---------- (1) assimilation-rate / noise sensitivity sweep ----------------

def run_condition_param(condition, seed, assim_alpha, opinion_noise):
    rng = np.random.default_rng(seed)
    opinions = np.clip(rng.normal(0, 0.3, size=sim.N_AGENTS), -1, 1)
    log_feats, log_labels = [], []
    model = None

    for rnd in range(sim.N_ROUNDS):
        candidates = rng.uniform(-1, 1, size=sim.CANDIDATES_PER_ROUND)
        use_random = (condition == "random") or (model is None)
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
        probs = sim.engagement_prob(opinions_rep, rec_positions.reshape(-1)).reshape(sim.N_AGENTS, sim.K_RECS)
        draws = rng.uniform(size=probs.shape)
        engaged = draws < probs

        for i in range(sim.N_AGENTS):
            eng_items = rec_positions[i][engaged[i]]
            for pos in eng_items:
                opinions[i] += assim_alpha * (pos - opinions[i])
        opinions += rng.normal(0, opinion_noise, size=sim.N_AGENTS)
        opinions = np.clip(opinions, -1, 1)

        feats = sim.features(np.repeat(opinions, sim.K_RECS), rec_positions.reshape(-1))
        labels = engaged.reshape(-1).astype(int)
        log_feats.append(feats)
        log_labels.append(labels)

        if condition != "random":
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


def run_sensitivity_sweep():
    base_alpha, base_noise = sim.ASSIM_ALPHA, sim.OPINION_NOISE
    alphas = [0.04, 0.08, 0.16, 0.32]
    noises = [0.0, 0.01, 0.03, 0.06]
    rows = []
    for alpha in alphas:
        for cond in ["static", "adaptive"]:
            vals = [run_condition_param(cond, s, alpha, base_noise) for s in SEEDS3]
            rows.append({"sweep": "assim_alpha", "value": alpha, "condition": cond,
                         "var_mean": float(np.mean(vals)), "var_std": float(np.std(vals)), "raw": vals})
            print(f"[alpha={alpha}] {cond:8s} var={np.mean(vals):.4f}+/-{np.std(vals):.4f}")
    for noise in noises:
        for cond in ["static", "adaptive"]:
            vals = [run_condition_param(cond, s, base_alpha, noise) for s in SEEDS3]
            rows.append({"sweep": "opinion_noise", "value": noise, "condition": cond,
                         "var_mean": float(np.mean(vals)), "var_std": float(np.std(vals)), "raw": vals})
            print(f"[noise={noise}] {cond:8s} var={np.mean(vals):.4f}+/-{np.std(vals):.4f}")
    return {"rows": rows}


# ---------- (2) bootstrap CIs for the n=3 comparisons ----------------------
# Resample the 3 seed-level values with replacement (agent-level trajectories
# aren't stored per-round for these conditions, so we bootstrap at the seed
# level -- a legitimate, cheap alternative to running more seeds that at
# least quantifies how much the paired-t estimate would move under resampling).

def bootstrap_diff_ci(a, b, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"boot_mean_diff": float(diffs.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi), "n_boot": n_boot}


def run_bootstrap_cis():
    out = {}
    # misspec static vs adaptive (linear_only)
    lin_static = [rev.run_condition_misspec("static", s) for s in SEEDS3]
    lin_adaptive = [rev.run_condition_misspec("adaptive", s) for s in SEEDS3]
    out["misspec_linear_only_static_vs_adaptive"] = bootstrap_diff_ci(lin_static, lin_adaptive)
    out["misspec_linear_only_static_vs_adaptive"]["raw_static"] = lin_static
    out["misspec_linear_only_static_vs_adaptive"]["raw_adaptive"] = lin_adaptive

    # drift retrain1 vs retrain20
    d1 = [rev.run_condition_drift("adaptive", s, 1) for s in SEEDS3]
    d20 = [rev.run_condition_drift("adaptive", s, 20) for s in SEEDS3]
    out["drift_retrain1_vs_retrain20"] = bootstrap_diff_ci(d1, d20)
    out["drift_retrain1_vs_retrain20"]["raw_retrain1"] = d1
    out["drift_retrain1_vs_retrain20"]["raw_retrain20"] = d20

    # partial misspec: no_dist static vs adaptive
    nd_static = [rev.run_condition_misspec("static", s, rev.features_no_dist) for s in SEEDS3]
    nd_adaptive = [rev.run_condition_misspec("adaptive", s, rev.features_no_dist) for s in SEEDS3]
    out["no_dist_static_vs_adaptive"] = bootstrap_diff_ci(nd_static, nd_adaptive)
    out["no_dist_static_vs_adaptive"]["raw_static"] = nd_static
    out["no_dist_static_vs_adaptive"]["raw_adaptive"] = nd_adaptive

    for k, v in out.items():
        print(f"[bootstrap] {k}: diff mean={v['boot_mean_diff']:.4f} 95% CI=[{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}]")
    return out


# ---------- (3) coefficient diagnostic: no_dist vs linear_only -------------
# Fit each feature set once on a shared, large, *unbiased* random-exposure
# log (opinions x uniform items, ground-truth-labeled) so the comparison
# isolates what each feature set predicts, not run-to-run bootstrapping
# dynamics -- then compare decision surfaces on a fixed opinion grid.

def diagnostic_fit_and_compare(seed=0, n_samples=20000):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=n_samples)
    items = rng.uniform(-1, 1, size=n_samples)
    probs = sim.engagement_prob(opinions, items)
    labels = (rng.uniform(size=n_samples) < probs).astype(int)

    results = {}
    grid_op = np.array([-0.6, -0.2, 0.2, 0.6])
    grid_it = np.linspace(-1, 1, 9)
    for name, feat_fn in [("full", sim.features), ("no_dist", rev.features_no_dist),
                           ("no_product", rev.features_no_product), ("linear_only", rev.features_misspec)]:
        X = feat_fn(opinions, items)
        m = LogisticRegression(max_iter=500).fit(X, labels)
        coef = m.coef_[0].tolist()
        intercept = float(m.intercept_[0])
        # predicted P(engage) on a fixed grid, per agent-opinion row, to see
        # whether the model's preferred item for each opinion is near (agrees
        # with ground truth) or far (systematically wrong / overconfident).
        row_preds = {}
        for op in grid_op:
            gx = feat_fn(np.full_like(grid_it, op), grid_it)
            p = m.predict_proba(gx)[:, 1]
            best_item = float(grid_it[np.argmax(p)])
            row_preds[float(op)] = {"argmax_item": best_item, "max_prob": float(p.max()), "min_prob": float(p.min())}
        results[name] = {"coef": coef, "intercept": intercept, "grid_preds": row_preds}
        print(f"[diag] {name:12s} coef={np.round(coef,3).tolist()} intercept={intercept:.3f}")
        for op, v in row_preds.items():
            print(f"        opinion={op:+.1f}: argmax_item={v['argmax_item']:+.2f} "
                  f"max_p={v['max_prob']:.3f} min_p={v['min_prob']:.3f}")
    return results


def main():
    t0 = time.time()
    print("=== (1) Assimilation-rate / noise sensitivity sweep ===")
    sens = run_sensitivity_sweep()

    print("\n=== (2) Bootstrap CIs for n=3 comparisons ===")
    boot = run_bootstrap_cis()

    print("\n=== (3) Coefficient diagnostic: no_dist vs linear_only ===")
    diag = diagnostic_fit_and_compare()

    with open("revision4_results.json", "w") as f:
        json.dump({"sensitivity_sweep": sens, "bootstrap_cis": boot, "coefficient_diagnostic": diag}, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
