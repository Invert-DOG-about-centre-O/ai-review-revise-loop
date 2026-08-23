"""
Robustness checks for the confidence-modulated-sycophancy simulation,
addressing peer review concerns:
  (1) single-seed point estimates -> report mean +/- std over many seeds
  (2) exact linear interpolation in c -> also test a nonlinear (quadratic
      and sigmoid-sharpened) confidence-modulation functional form to see
      whether the qualitative signature (monotonic rise, large positive c
      coefficient) survives changes to the reward's functional form.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment import simulate, make_features, evaluate

N_TRAIN = 40000
N_TEST = 20000
N_SEEDS = 20


def optimal_defer_labels_variant(scheme, t, e_hat, u, c):
    b_defer1 = u
    b_defer0 = e_hat

    if scheme == "accuracy_oracle":
        r1 = (b_defer1 == t).astype(float)
        r0 = (b_defer0 == t).astype(float)
    elif scheme == "approval_flat":
        r1 = (b_defer1 == u).astype(float)
        r0 = (b_defer0 == u).astype(float)
    elif scheme == "approval_confidence_linear":
        w = c
        r1 = w * (b_defer1 == u) + (1 - w) * (b_defer1 == t)
        r0 = w * (b_defer0 == u) + (1 - w) * (b_defer0 == t)
    elif scheme == "approval_confidence_quadratic":
        w = c ** 2
        r1 = w * (b_defer1 == u) + (1 - w) * (b_defer1 == t)
        r0 = w * (b_defer0 == u) + (1 - w) * (b_defer0 == t)
    elif scheme == "approval_confidence_sigmoid":
        w = 1.0 / (1.0 + np.exp(-8 * (c - 0.5)))
        r1 = w * (b_defer1 == u) + (1 - w) * (b_defer1 == t)
        r0 = w * (b_defer0 == u) + (1 - w) * (b_defer0 == t)
    else:
        raise ValueError(scheme)

    r1 = np.asarray(r1, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    return (r1 > r0).astype(int)


def fit_and_eval(scheme, d_tr, t_tr, e_tr, u_tr, c_tr, d_te, t_te, e_te, u_te, c_te):
    y_tr = optimal_defer_labels_variant(scheme, t_tr, e_tr, u_tr, c_tr)
    X_tr = make_features(d_tr, c_tr, u_tr, e_tr)

    if y_tr.min() == y_tr.max():
        const_label = int(y_tr[0])

        class ConstPolicy:
            def __init__(self, label):
                self.label = label

            def predict(self, X):
                return np.full(X.shape[0], self.label)

        policy = ConstPolicy(const_label)
        coef_c = float("nan")
    else:
        policy = LogisticRegression(max_iter=1000)
        policy.fit(X_tr, y_tr)
        coef_c = float(policy.coef_[0][1])

    res = evaluate(policy, d_te, t_te, e_te, u_te, c_te)
    q_rates = [q["sycophancy_rate"] for q in res["by_confidence_quartile"]]
    return res["accuracy"], res["sycophancy_rate"], coef_c, q_rates


def main():
    schemes = [
        "accuracy_oracle",
        "approval_flat",
        "approval_confidence_linear",
        "approval_confidence_quadratic",
        "approval_confidence_sigmoid",
    ]

    per_scheme = {s: {"accuracy": [], "sycophancy_rate": [], "coef_c": [], "q_rates": []} for s in schemes}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        d_tr, t_tr, e_tr, u_tr, c_tr = simulate(N_TRAIN, rng)
        d_te, t_te, e_te, u_te, c_te = simulate(N_TEST, rng)

        for s in schemes:
            acc, syc, coef_c, q_rates = fit_and_eval(
                s, d_tr, t_tr, e_tr, u_tr, c_tr, d_te, t_te, e_te, u_te, c_te
            )
            per_scheme[s]["accuracy"].append(acc)
            per_scheme[s]["sycophancy_rate"].append(syc)
            per_scheme[s]["coef_c"].append(coef_c)
            per_scheme[s]["q_rates"].append(q_rates)

    summary = {}
    for s in schemes:
        acc = np.array(per_scheme[s]["accuracy"])
        syc = np.array(per_scheme[s]["sycophancy_rate"])
        coef_c = np.array(per_scheme[s]["coef_c"], dtype=float)
        q = np.array(per_scheme[s]["q_rates"], dtype=float)  # (seeds, 4)
        summary[s] = {
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std()),
            "sycophancy_rate_mean": float(syc.mean()),
            "sycophancy_rate_std": float(syc.std()),
            "coef_c_mean": float(np.nanmean(coef_c)),
            "coef_c_std": float(np.nanstd(coef_c)),
            "quartile_mean": q.mean(axis=0).tolist(),
            "quartile_std": q.std(axis=0).tolist(),
        }

    out = {"n_seeds": N_SEEDS, "n_train": N_TRAIN, "n_test": N_TEST, "summary": summary}
    with open("robustness_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
