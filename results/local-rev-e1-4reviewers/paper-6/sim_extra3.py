"""
Round-3 revision experiments addressing round-3 reviewer concerns, several
raised independently by 2+ of the 4 reviewers:

(a) [Reviewers 2,3,4] The structural-constant sensitivity sweep in
    sim_extra2.py used only a single seed (seed 0) per perturbation, unlike
    the 20-seed protocol used for the main competence-crossing sweep, so its
    "robust across settings" claim was not itself variance-checked.
(b) [Reviewers 1,2] The sensitivity sweep only perturbed LR and the two
    reliance-logistic weights; the confidence means (0.85/0.55) and the
    0.3x non-reliance trust-decay factor were held fixed throughout.
(c) [Reviewers 1,3] The equal-competence residual accuracy cost (Sec 3.3) is
    described narratively but not decomposed; a minimal ablation splitting
    outcomes by whether the AI deferred that round would let the "deferred
    answers get conditioned on by calibrated confidence" story be tested
    directly rather than asserted.

This script: (1) re-runs the 5-setting structural sensitivity sweep across
10 seeds instead of 1; (2) adds two new perturbation axes (confidence means,
non-reliance decay factor) to that same multi-seed sweep; (3) decomposes the
equal-competence s=0-vs-s=1 accuracy gap into deferred-round vs
non-deferred-round contributions.
"""
import numpy as np
import json
import time
import sim as base

N_SEEDS = 10
COMBOS = [("AI>user (base)", 0.78, 0.55), ("equal competence", 0.65, 0.65)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_condition_full(s, rng, w_trust=3.0, w_conf=2.0, lr=None,
                        conf_hi=0.85, conf_lo=0.55, decay=0.3,
                        ai_c=None, user_c=None):
    """Generalized copy of sim.run_condition exposing every hand-set
    structural constant as an argument (confidence means, non-reliance decay
    factor, reliance weights, LR), used for the extended sensitivity sweep."""
    N_DYADS, N_ROUNDS = base.N_DYADS, base.N_ROUNDS
    AI_COMPETENCE = ai_c if ai_c is not None else base.AI_COMPETENCE
    USER_COMPETENCE = user_c if user_c is not None else base.USER_COMPETENCE
    LR = lr if lr is not None else base.LR
    T = np.full(N_DYADS, 0.5)
    trust_traj = np.zeros((N_DYADS, N_ROUNDS))
    acc_traj = np.zeros((N_DYADS, N_ROUNDS))
    for t in range(N_ROUNDS):
        ground_truth = rng.integers(0, 2, size=N_DYADS)
        user_correct_mask = rng.random(N_DYADS) < USER_COMPETENCE
        user_belief = np.where(user_correct_mask, ground_truth, 1 - ground_truth)
        ai_correct_mask = rng.random(N_DYADS) < AI_COMPETENCE
        ai_private = np.where(ai_correct_mask, ground_truth, 1 - ground_truth)
        disagree = ai_private != user_belief
        defer = disagree & (rng.random(N_DYADS) < s)
        ai_stated = np.where(defer, user_belief, ai_private)
        true_correct = (ai_stated == ground_truth).astype(float)
        conf = np.where(true_correct == 1, conf_hi, conf_lo) + rng.normal(0, 0.03, N_DYADS)
        conf = np.clip(conf, 0.0, 1.0)
        p_follow = sigmoid(w_trust * (T - 0.5) + w_conf * (conf - 0.5))
        follow = rng.random(N_DYADS) < p_follow
        final_answer = np.where(follow, ai_stated, user_belief)
        outcome_correct = (final_answer == ground_truth).astype(float)
        ai_was_right = (ai_stated == ground_truth).astype(float)
        T = T + LR * follow * (ai_was_right - T)
        T = T + LR * (1 - follow) * decay * (ai_was_right - T)
        T = np.clip(T, 0.0, 1.0)
        trust_traj[:, t] = T
        acc_traj[:, t] = outcome_correct
    return {
        "final_trust_mean": float(trust_traj[:, -10:].mean()),
        "late_accuracy": float(acc_traj[:, -10:].mean()),
    }


def multiseed_sensitivity_sweep():
    perturbations = [
        ("default", dict()),
        ("low LR (0.04)", dict(lr=0.04)),
        ("high LR (0.30)", dict(lr=0.30)),
        ("low reliance slope (1,1)", dict(w_trust=1.0, w_conf=1.0)),
        ("high reliance slope (6,4)", dict(w_trust=6.0, w_conf=4.0)),
        ("compressed confidence (0.70/0.60)", dict(conf_hi=0.70, conf_lo=0.60)),
        ("extreme confidence (0.95/0.50)", dict(conf_hi=0.95, conf_lo=0.50)),
        ("low non-reliance decay (0.1x)", dict(decay=0.1)),
        ("high non-reliance decay (0.6x)", dict(decay=0.6)),
    ]
    results = []
    for pname, kwargs in perturbations:
        for clabel, ai_c, user_c in COMBOS:
            trust_gaps, acc_gaps = [], []
            for seed in range(N_SEEDS):
                r0 = run_condition_full(0.0, np.random.default_rng(seed), ai_c=ai_c, user_c=user_c, **kwargs)
                r1 = run_condition_full(1.0, np.random.default_rng(seed), ai_c=ai_c, user_c=user_c, **kwargs)
                trust_gaps.append(r1["final_trust_mean"] - r0["final_trust_mean"])
                acc_gaps.append(r1["late_accuracy"] - r0["late_accuracy"])
            results.append({
                "perturbation": pname, "competence": clabel,
                "trust_gap_mean": float(np.mean(trust_gaps)), "trust_gap_std": float(np.std(trust_gaps)),
                "acc_gap_mean": float(np.mean(acc_gaps)), "acc_gap_std": float(np.std(acc_gaps)),
            })
    return results


def deferred_round_decomposition():
    """At equal competence, decompose the s=0-vs-s=1 late-accuracy gap into
    the contribution from rounds where the AI deferred vs. did not, to test
    the Sec 3.3 hypothesis that calibrated confidence's conditioning on the
    deferred answer's correctness (not a general accuracy shift) drives the
    residual."""
    ai_c, user_c = 0.65, 0.65
    N_DYADS, N_ROUNDS = base.N_DYADS, base.N_ROUNDS
    LATE = 10  # last 10 rounds, matching late_accuracy definition elsewhere

    def run(s, rng):
        T = np.full(N_DYADS, 0.5)
        late_acc_deferred, late_acc_nondeferred = [], []
        late_n_deferred, late_n_nondeferred = [], []
        for t in range(N_ROUNDS):
            ground_truth = rng.integers(0, 2, size=N_DYADS)
            user_correct_mask = rng.random(N_DYADS) < user_c
            user_belief = np.where(user_correct_mask, ground_truth, 1 - ground_truth)
            ai_correct_mask = rng.random(N_DYADS) < ai_c
            ai_private = np.where(ai_correct_mask, ground_truth, 1 - ground_truth)
            disagree = ai_private != user_belief
            defer = disagree & (rng.random(N_DYADS) < s)
            ai_stated = np.where(defer, user_belief, ai_private)
            true_correct = (ai_stated == ground_truth).astype(float)
            conf = np.where(true_correct == 1, 0.85, 0.55) + rng.normal(0, 0.03, N_DYADS)
            conf = np.clip(conf, 0.0, 1.0)
            p_follow = sigmoid(3.0 * (T - 0.5) + 2.0 * (conf - 0.5))
            follow = rng.random(N_DYADS) < p_follow
            final_answer = np.where(follow, ai_stated, user_belief)
            outcome_correct = (final_answer == ground_truth).astype(float)
            ai_was_right = (ai_stated == ground_truth).astype(float)
            T = T + base.LR * follow * (ai_was_right - T)
            T = T + base.LR * (1 - follow) * 0.3 * (ai_was_right - T)
            T = np.clip(T, 0.0, 1.0)
            if t >= N_ROUNDS - LATE:
                late_acc_deferred.append(outcome_correct[defer].sum())
                late_n_deferred.append(defer.sum())
                late_acc_nondeferred.append(outcome_correct[~defer].sum())
                late_n_nondeferred.append((~defer).sum())
        acc_def = float(np.sum(late_acc_deferred) / max(np.sum(late_n_deferred), 1))
        acc_nondef = float(np.sum(late_acc_nondeferred) / np.sum(late_n_nondeferred))
        share_deferred = float(np.sum(late_n_deferred) / (N_DYADS * LATE))
        return acc_def, acc_nondef, share_deferred

    out = {"per_seed": []}
    for seed in range(N_SEEDS):
        d0, nd0, share0 = run(0.0, np.random.default_rng(seed))
        d1, nd1, share1 = run(1.0, np.random.default_rng(seed))
        out["per_seed"].append({
            "seed": seed,
            "acc_deferred_s0": d0, "acc_nondeferred_s0": nd0, "share_deferred_s0": share0,
            "acc_deferred_s1": d1, "acc_nondeferred_s1": nd1, "share_deferred_s1": share1,
        })
    nd0s = [r["acc_nondeferred_s0"] for r in out["per_seed"]]
    nd1s = [r["acc_nondeferred_s1"] for r in out["per_seed"]]
    d1s = [r["acc_deferred_s1"] for r in out["per_seed"]]
    share1s = [r["share_deferred_s1"] for r in out["per_seed"]]
    out["summary"] = {
        "nondeferred_acc_s0_mean": float(np.mean(nd0s)),
        "nondeferred_acc_s1_mean": float(np.mean(nd1s)),
        "nondeferred_acc_gap": float(np.mean(nd1s) - np.mean(nd0s)),
        "deferred_acc_s1_mean": float(np.mean(d1s)),
        "share_deferred_at_s1_mean": float(np.mean(share1s)),
    }
    return out


def main():
    t0 = time.time()
    sens = multiseed_sensitivity_sweep()
    decomp = deferred_round_decomposition()
    out = {"multiseed_sensitivity_sweep": sens, "deferred_round_decomposition": decomp}
    with open("results_extra3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nExtra3 runtime: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
