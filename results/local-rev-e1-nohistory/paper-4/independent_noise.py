"""
Round-3 review follow-up (reviewer question 1): the learned-reward-model
check in learned_reward.py still generates the noisy annotator's preferences
by perturbing the SAME closed-form reward gap (r1 - r0) with Bradley-Terry
noise -- so annotator noise is always centered on the true gap. The reviewer
asks: does the monotonic confidence-sycophancy signature depend on that
centering, or does it survive when annotator noise is not anchored to the
true reward gap at all?

We test this directly with an eps-contamination mixture: with probability
(1-eps) the annotator's preference is the Bradley-Terry draw around the true
gap (tau=0.15, as in the "med" row of learned_reward.py); with probability
eps the annotator's preference is a UNIFORMLY RANDOM coin flip, independent
of the reward gap entirely. eps=0 reproduces the tau=0.15 row of
learned_reward.py; eps=1 means the "annotator" carries zero information
about the true reward gap. Sweeping eps lets us see how the signature
degrades as annotator noise becomes progressively less centered on / less
informative about the true reward, up to the limiting case of pure noise.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment import simulate, make_features, evaluate
from learned_reward import true_rewards_confidence_linear, fit_policy_from_reward_model

N_PREF = 40000
N_POLICY_TRAIN = 40000
N_TEST = 20000
TAU = 0.15
EPS_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 8


def fit_reward_model_contaminated(d, t, e_hat, u, c, tau, eps, rng):
    r1, r0 = true_rewards_confidence_linear(t, e_hat, u, c)
    p_prefer_defer = 1.0 / (1.0 + np.exp(-(r1 - r0) / tau))
    bt_pref = (rng.uniform(0, 1, size=len(p_prefer_defer)) < p_prefer_defer).astype(int)
    random_pref = (rng.uniform(0, 1, size=len(p_prefer_defer)) < 0.5).astype(int)
    is_contaminated = rng.uniform(0, 1, size=len(p_prefer_defer)) < eps
    noisy_pref = np.where(is_contaminated, random_pref, bt_pref)

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


def run_one(eps, seed):
    rng = np.random.default_rng(seed)
    d_pref, t_pref, e_pref, u_pref, c_pref = simulate(N_PREF, rng)
    rm = fit_reward_model_contaminated(d_pref, t_pref, e_pref, u_pref, c_pref, TAU, eps, rng)

    d_tr, t_tr, e_tr, u_tr, c_tr = simulate(N_POLICY_TRAIN, rng)
    policy, coef_c = fit_policy_from_reward_model(rm, d_tr, c_tr, u_tr, e_tr)

    d_te, t_te, e_te, u_te, c_te = simulate(N_TEST, rng)
    res = evaluate(policy, d_te, t_te, e_te, u_te, c_te)
    q_rates = [q["sycophancy_rate"] for q in res["by_confidence_quartile"]]
    return res["accuracy"], res["sycophancy_rate"], coef_c, q_rates


def main():
    summary = {}
    for eps in EPS_VALUES:
        accs, sycs, coefs, qs = [], [], [], []
        for seed in range(N_SEEDS):
            acc, syc, coef_c, q_rates = run_one(eps, seed)
            accs.append(acc)
            sycs.append(syc)
            coefs.append(coef_c)
            qs.append(q_rates)
        accs, sycs, coefs, qs = map(np.array, (accs, sycs, coefs, qs))
        summary[f"eps_{eps}"] = {
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
           "n_test": N_TEST, "tau": TAU, "eps_values": EPS_VALUES, "summary": summary}
    with open("independent_noise_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
