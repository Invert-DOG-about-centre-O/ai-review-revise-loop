"""
Round-2 review follow-up: does the crossover-point finding survive when the
reward itself is *learned* from noisy simulated preference comparisons,
rather than being the closed-form oracle formula used in experiment.py?

Pipeline (mirrors a simplified RLHF reward-model step):
  1. For each trial, compute the TRUE closed-form approval-confidence-linear
     rewards r1 (defer) and r0 (honest), as in experiment.py.
  2. Simulate a noisy annotator: prefers defer=1 with probability
     sigmoid((r1 - r0) / tau), i.e. a Bradley-Terry-style noisy comparison
     around the true reward gap, tau controlling annotator noise.
  3. Fit a REWARD MODEL: logistic regression predicting the noisy preference
     label from features (d, c, u, e_hat, agree) -- this is the learned
     analogue of the closed-form reward.
  4. Take the reward model's argmax action as the training label and fit a
     downstream POLICY (same logistic-regression distillation step as the
     main paper) on a fresh batch of trials.
  5. Evaluate the resulting policy exactly as in experiment.py and check
     whether it reproduces the monotonic confidence-quartile rise and a
     crossover near c=0.5.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment import simulate, make_features, evaluate

N_PREF = 40000     # trials used to fit the noisy-preference reward model
N_POLICY_TRAIN = 40000
N_TEST = 20000
TAU_VALUES = [0.05, 0.15, 0.30]   # annotator-noise temperatures (low->high noise)
N_SEEDS = 10


def true_rewards_confidence_linear(t, e_hat, u, c):
    b1, b0 = u, e_hat
    r1 = c * (b1 == u).astype(float) + (1 - c) * (b1 == t).astype(float)
    r0 = c * (b0 == u).astype(float) + (1 - c) * (b0 == t).astype(float)
    return r1, r0


def fit_reward_model(d, t, e_hat, u, c, tau, rng):
    r1, r0 = true_rewards_confidence_linear(t, e_hat, u, c)
    p_prefer_defer = 1.0 / (1.0 + np.exp(-(r1 - r0) / tau))
    noisy_pref = (rng.uniform(0, 1, size=len(p_prefer_defer)) < p_prefer_defer).astype(int)

    X = make_features(d, c, u, e_hat)
    if noisy_pref.min() == noisy_pref.max():
        const = int(noisy_pref[0])

        class ConstRM:
            def predict(self, X):
                return np.full(X.shape[0], const)

        return ConstRM()

    rm = LogisticRegression(max_iter=1000)
    rm.fit(X, noisy_pref)
    return rm


def fit_policy_from_reward_model(rm, d, c, u, e_hat):
    X = make_features(d, c, u, e_hat)
    y = rm.predict(X)  # reward-model-preferred action becomes the training label
    if y.min() == y.max():
        const = int(y[0])

        class ConstPolicy:
            def predict(self, X):
                return np.full(X.shape[0], const)

        return ConstPolicy(), float("nan")

    policy = LogisticRegression(max_iter=1000)
    policy.fit(X, y)
    return policy, float(policy.coef_[0][1])


def run_one(tau, seed):
    rng = np.random.default_rng(seed)
    d_pref, t_pref, e_pref, u_pref, c_pref = simulate(N_PREF, rng)
    rm = fit_reward_model(d_pref, t_pref, e_pref, u_pref, c_pref, tau, rng)

    d_tr, t_tr, e_tr, u_tr, c_tr = simulate(N_POLICY_TRAIN, rng)
    policy, coef_c = fit_policy_from_reward_model(rm, d_tr, c_tr, u_tr, e_tr)

    d_te, t_te, e_te, u_te, c_te = simulate(N_TEST, rng)
    res = evaluate(policy, d_te, t_te, e_te, u_te, c_te)
    q_rates = [q["sycophancy_rate"] for q in res["by_confidence_quartile"]]
    return res["accuracy"], res["sycophancy_rate"], coef_c, q_rates


def main():
    summary = {}
    for tau in TAU_VALUES:
        accs, sycs, coefs, qs = [], [], [], []
        for seed in range(N_SEEDS):
            acc, syc, coef_c, q_rates = run_one(tau, seed)
            accs.append(acc)
            sycs.append(syc)
            coefs.append(coef_c)
            qs.append(q_rates)
        accs, sycs, coefs, qs = map(np.array, (accs, sycs, coefs, qs))
        summary[f"tau_{tau}"] = {
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std()),
            "sycophancy_rate_mean": float(sycs.mean()),
            "sycophancy_rate_std": float(sycs.std()),
            "coef_c_mean": float(np.nanmean(coefs)),
            "coef_c_std": float(np.nanstd(coefs)),
            "quartile_mean": qs.mean(axis=0).tolist(),
            "quartile_std": qs.std(axis=0).tolist(),
        }

    out = {"n_seeds": N_SEEDS, "n_pref": N_PREF, "n_policy_train": N_POLICY_TRAIN,
           "n_test": N_TEST, "tau_values": TAU_VALUES, "summary": summary}
    with open("learned_reward_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
