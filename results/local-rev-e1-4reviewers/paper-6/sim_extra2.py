"""
Round-2 revision experiments addressing reviewer round-2 concerns, all four of
whom independently flagged: (a) the competence-crossing sweep (sim_extra.py)
was only run at a single seed with s in {0,1}, unlike the multi-seed base
gap check; (b) no sensitivity analysis on the hand-set structural constants
(LR, reliance-logistic weights, confidence means); (c) whether the small
residual harm at equal competence is fully explained by the confidence-
decoupling mechanism of Section 3.3.

1. Multi-seed (20 seeds) competence-crossing sweep, full s-sweep (not just
   s=0,1) at all three competence configurations.
2. Sensitivity sweep: re-run base vs equal-competence s=0-vs-s=1 gap under
   perturbed LR and reliance-logistic weights.
3. Equal-competence residual isolation: decoupled-confidence run at equal
   competence to check whether the -0.036 accuracy residual is explained by
   confidence decoupling alone.
Reuses run_condition from sim.py, monkeypatching module-level constants.
"""
import numpy as np
import json
import time
import sim as base

N_SEEDS = 20
S_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COMBOS = [
    ("AI>user (base)", 0.78, 0.55),
    ("equal competence", 0.65, 0.65),
    ("user>AI (reversed)", 0.55, 0.78),
]


def multi_seed_competence_crossing():
    """Full s-sweep x 3 competence configs x 20 seeds."""
    out = {}
    for label, ai_c, user_c in COMBOS:
        base.AI_COMPETENCE = ai_c
        base.USER_COMPETENCE = user_c
        per_s = {s: {"trust": [], "acc": []} for s in S_LEVELS}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            for s in S_LEVELS:
                res = base.run_condition(s, True, rng)
                per_s[s]["trust"].append(res["final_trust_mean"])
                per_s[s]["acc"].append(res["late_accuracy"])
        rows = []
        for s in S_LEVELS:
            rows.append({
                "s": s,
                "trust_mean": float(np.mean(per_s[s]["trust"])),
                "trust_std": float(np.std(per_s[s]["trust"])),
                "acc_mean": float(np.mean(per_s[s]["acc"])),
                "acc_std": float(np.std(per_s[s]["acc"])),
            })
        gap_trust = np.array(per_s[1.0]["trust"]) - np.array(per_s[0.0]["trust"])
        gap_acc = np.array(per_s[1.0]["acc"]) - np.array(per_s[0.0]["acc"])
        out[label] = {
            "AI_COMPETENCE": ai_c, "USER_COMPETENCE": user_c,
            "by_s": rows,
            "gap_trust_s1_minus_s0_mean": float(np.mean(gap_trust)),
            "gap_trust_s1_minus_s0_std": float(np.std(gap_trust)),
            "gap_acc_s1_minus_s0_mean": float(np.mean(gap_acc)),
            "gap_acc_s1_minus_s0_std": float(np.std(gap_acc)),
        }
    base.AI_COMPETENCE = 0.78
    base.USER_COMPETENCE = 0.55
    return out


def sensitivity_sweep():
    """Re-run base-vs-equal-competence s=0-vs-s=1 gap under perturbed
    structural constants (LR, reliance-logistic trust/confidence weights)."""
    orig_LR = base.LR
    perturbations = [
        ("default (LR=0.12, w_trust=3.0, w_conf=2.0)", 0.12, 3.0, 2.0),
        ("low LR (0.04, sticky trust)", 0.04, 3.0, 2.0),
        ("high LR (0.30, fast trust)", 0.30, 3.0, 2.0),
        ("low reliance slope (w_trust=1.0, w_conf=1.0)", 0.12, 1.0, 1.0),
        ("high reliance slope (w_trust=6.0, w_conf=4.0)", 0.12, 6.0, 4.0),
    ]
    results = []
    for pname, lr, w_t, w_c in perturbations:
        for clabel, ai_c, user_c in [("AI>user (base)", 0.78, 0.55),
                                       ("equal competence", 0.65, 0.65)]:
            base.LR = lr
            base.AI_COMPETENCE = ai_c
            base.USER_COMPETENCE = user_c
            orig_sigmoid_arg = None
            # monkeypatch run_condition's hardcoded weights via a wrapper:
            # sim.py hardcodes 3.0/2.0 inline, so we patch by temporarily
            # replacing sigmoid call semantics is not possible without
            # editing sim.py; instead we reimplement only the weight change
            # by scaling T and conf inputs equivalently is not exact, so
            # we directly vary LR (a true free parameter) and, for slope,
            # call a local modified copy of run_condition.
            res0 = run_condition_custom(0.0, True, np.random.default_rng(0), w_t, w_c)
            res1 = run_condition_custom(1.0, True, np.random.default_rng(0), w_t, w_c)
            results.append({
                "perturbation": pname, "competence": clabel,
                "AI_COMPETENCE": ai_c, "USER_COMPETENCE": user_c,
                "LR": lr, "w_trust": w_t, "w_conf": w_c,
                "trust_delta_s1_minus_s0": res1["final_trust_mean"] - res0["final_trust_mean"],
                "acc_delta_s1_minus_s0": res1["late_accuracy"] - res0["late_accuracy"],
            })
    base.LR = orig_LR
    base.AI_COMPETENCE = 0.78
    base.USER_COMPETENCE = 0.55
    return results


def run_condition_custom(s, calibrated_confidence, rng, w_trust, w_conf):
    """Copy of sim.run_condition with configurable reliance-logistic weights,
    used only for the sensitivity sweep (sim.py itself hardcodes 3.0/2.0)."""
    N_DYADS, N_ROUNDS = base.N_DYADS, base.N_ROUNDS
    AI_COMPETENCE, USER_COMPETENCE, LR = base.AI_COMPETENCE, base.USER_COMPETENCE, base.LR
    trust_traj = np.zeros((N_DYADS, N_ROUNDS))
    acc_traj = np.zeros((N_DYADS, N_ROUNDS))
    T = np.full(N_DYADS, 0.5)
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
        if calibrated_confidence:
            conf = np.where(true_correct == 1, 0.85, 0.55) + rng.normal(0, 0.03, N_DYADS)
        else:
            conf = np.full(N_DYADS, 0.85) + rng.normal(0, 0.03, N_DYADS)
        conf = np.clip(conf, 0.0, 1.0)
        p_follow = base.sigmoid(w_trust * (T - 0.5) + w_conf * (conf - 0.5))
        follow = rng.random(N_DYADS) < p_follow
        final_answer = np.where(follow, ai_stated, user_belief)
        outcome_correct = (final_answer == ground_truth).astype(float)
        ai_was_right = (ai_stated == ground_truth).astype(float)
        T = T + LR * follow * (ai_was_right - T)
        T = T + LR * (1 - follow) * 0.3 * (ai_was_right - T)
        T = np.clip(T, 0.0, 1.0)
        trust_traj[:, t] = T
        acc_traj[:, t] = outcome_correct
    return {
        "final_trust_mean": float(trust_traj[:, -10:].mean()),
        "late_accuracy": float(acc_traj[:, -10:].mean()),
    }


def equal_competence_decoupled_isolation():
    """At equal competence, compare calibrated vs decoupled confidence at
    s=0 and s=1 to check whether the -0.036 late-accuracy residual reported
    in Section 3.2 is explained by the confidence-decoupling mechanism."""
    base.AI_COMPETENCE = 0.65
    base.USER_COMPETENCE = 0.65
    out = {}
    for calibrated in [True, False]:
        rng = np.random.default_rng(0)
        res0 = base.run_condition(0.0, calibrated, rng)
        res1 = base.run_condition(1.0, calibrated, rng)
        out["calibrated" if calibrated else "decoupled"] = {
            "final_trust_s0": res0["final_trust_mean"], "final_trust_s1": res1["final_trust_mean"],
            "late_acc_s0": res0["late_accuracy"], "late_acc_s1": res1["late_accuracy"],
            "trust_delta": res1["final_trust_mean"] - res0["final_trust_mean"],
            "acc_delta": res1["late_accuracy"] - res0["late_accuracy"],
        }
    base.AI_COMPETENCE = 0.78
    base.USER_COMPETENCE = 0.55
    return out


def main():
    t0 = time.time()
    mscc = multi_seed_competence_crossing()
    sens = sensitivity_sweep()
    iso = equal_competence_decoupled_isolation()
    out = {
        "multi_seed_competence_crossing": mscc,
        "sensitivity_sweep": sens,
        "equal_competence_decoupled_isolation": iso,
    }
    with open("results_extra2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nExtra2 runtime: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
