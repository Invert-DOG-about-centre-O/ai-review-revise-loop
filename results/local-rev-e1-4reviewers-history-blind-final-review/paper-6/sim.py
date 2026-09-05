import numpy as np

N_ROUNDS = 60
N_DYADS = 400
AI_COMPETENCE = 0.78
USER_COMPETENCE = 0.55
LR = 0.12
WEAK_MULT = 0.3
CONF_CORRECT_MEAN = 0.85
CONF_WRONG_MEAN = 0.55
CONF_STD = 0.08
INIT_TRUST = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_condition(s, decoupled_conf, seed, ai_comp=AI_COMPETENCE, user_comp=USER_COMPETENCE,
                   lr=LR, weak_mult=WEAK_MULT, slope_t=3.0, slope_c=2.0,
                   n_dyads=N_DYADS, n_rounds=N_ROUNDS):
    rng = np.random.default_rng(seed)
    T = np.full(n_dyads, INIT_TRUST)
    trust_traj = np.zeros((n_rounds, n_dyads))
    acc = np.zeros((n_rounds, n_dyads))
    overrel = np.zeros((n_rounds, n_dyads), dtype=bool)
    underrel = np.zeros((n_rounds, n_dyads), dtype=bool)

    for r in range(n_rounds):
        truth = rng.integers(0, 2, n_dyads)
        user_correct = rng.random(n_dyads) < user_comp
        ai_correct = rng.random(n_dyads) < ai_comp
        user_belief = np.where(user_correct, truth, 1 - truth)
        ai_private = np.where(ai_correct, truth, 1 - truth)

        disagree = ai_private != user_belief
        defer = disagree & (rng.random(n_dyads) < s)
        ai_stated = np.where(defer, user_belief, ai_private)
        ai_stated_correct = (ai_stated == truth)

        if decoupled_conf:
            conf = np.clip(rng.normal(CONF_CORRECT_MEAN, CONF_STD, n_dyads), 0, 1)
        else:
            conf = np.where(
                ai_stated_correct,
                np.clip(rng.normal(CONF_CORRECT_MEAN, CONF_STD, n_dyads), 0, 1),
                np.clip(rng.normal(CONF_WRONG_MEAN, CONF_STD, n_dyads), 0, 1),
            )

        p_follow = sigmoid(slope_t * (T - 0.5) + slope_c * (conf - 0.5))
        follow = rng.random(n_dyads) < p_follow

        final_answer = np.where(follow, ai_stated, user_belief)
        correct = (final_answer == truth)
        acc[r] = correct

        overrel[r] = follow & (~ai_stated_correct)
        underrel[r] = (~follow) & ai_stated_correct & (~user_correct)

        target = ai_stated_correct.astype(float)
        eff_lr = np.where(follow, lr, lr * weak_mult)
        T = T + eff_lr * (target - T)
        trust_traj[r] = T

    return {
        "trust_traj": trust_traj.mean(axis=1),
        "final_trust_mean": float(trust_traj[-1].mean()),
        "final_trust_std": float(trust_traj[-1].std()),
        "late_acc_mean": float(acc[-10:].mean()),
        "late_acc_std": float(acc[-10:].mean(axis=0).std()),
        "overreliance": float(overrel.mean()),
        "underreliance": float(underrel.mean()),
    }


if __name__ == "__main__":
    import json
    results = {}
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for decoupled in [False, True]:
            key = f"s={s}_decoupled={decoupled}"
            results[key] = run_condition(s, decoupled, seed=0)
    with open("results_repro.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    for s in [0.0, 0.4, 1.0]:
        r = results[f"s={s}_decoupled=False"]
        print(s, r["final_trust_mean"], r["late_acc_mean"], r["overreliance"], r["underreliance"])
