"""
Controlled simulation of sycophancy emergence from reward design.

We simulate a decision-maker ("assistant") that answers binary factual
questions in the presence of a user who states an opinion `u` (possibly
wrong) with an expressed confidence `c`. The assistant also has a noisy
internal belief `e_hat` about the true answer `t`, whose reliability
depends on question difficulty `d`.

At each interaction the assistant chooses:
    defer = 1  -> output the user's stated opinion u  (sycophantic option)
    defer = 0  -> output its own internal belief e_hat (honest option)

We train policies (via reward-weighted / oracle-label logistic regression,
a standard simplification of RLHF-style distillation onto a reward signal)
under three reward schemes:

  A) ACCURACY-ORACLE   : reward = 1[b == t]                 (verified correctness)
  B) APPROVAL-FLAT     : reward = 1[b == u]                  (pure agreement)
  C) APPROVAL-CONFIDENCE-MODULATED:
        reward = c * 1[b == u] + (1 - c) * 1[b == t]
     i.e. confident users only reward agreement; unconfident users are
     "receptive to correction" so correctness is rewarded instead.

We then measure, on a held-out test set, each policy's accuracy and its
SYCOPHANCY RATE (probability of deferring to the user when the user
disagrees with the assistant's own belief), including how sycophancy
rate varies with user-expressed confidence c, and the split between
"progressive" (override helps correctness) and "regressive" (override
hurts correctness) sycophancy, following the terminology used in
SycEval (Fanous et al., 2025) and ELEPHANT (Cheng et al., 2025).
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

RNG_SEED = 0
N_TRAIN = 40000
N_TEST = 20000


def simulate(n, rng):
    d = rng.uniform(0, 1, size=n)                     # question difficulty
    t = rng.integers(0, 2, size=n)                     # ground truth
    p_correct = 0.9 - 0.6 * d                           # competence decreases with difficulty
    flip_self = rng.uniform(0, 1, size=n) > p_correct
    e_hat = np.where(flip_self, 1 - t, t)                # assistant's own noisy belief

    u_correct = rng.uniform(0, 1, size=n) > 0.5          # user right 50% of the time
    u = np.where(u_correct, t, 1 - t)                    # user's stated opinion

    c = rng.uniform(0, 1, size=n)                        # user's expressed confidence
    return d, t, e_hat, u, c


def optimal_defer_labels(scheme, t, e_hat, u, c):
    b_defer1 = u          # output = user's opinion
    b_defer0 = e_hat       # output = own belief

    if scheme == "accuracy_oracle":
        r1 = (b_defer1 == t).astype(float)
        r0 = (b_defer0 == t).astype(float)
    elif scheme == "approval_flat":
        r1 = (b_defer1 == u).astype(float)
        r0 = (b_defer0 == u).astype(float)
    elif scheme == "approval_confidence":
        r1 = c * (b_defer1 == u).astype(float) + (1 - c) * (b_defer1 == t).astype(float)
        r0 = c * (b_defer0 == u).astype(float) + (1 - c) * (b_defer0 == t).astype(float)
    else:
        raise ValueError(scheme)

    # defer strictly preferred; ties default to honest (defer=0), the
    # conservative/status-quo action, matching typical RLHF initialization
    # from a base (non-sycophantic) model.
    defer = (r1 > r0).astype(int)
    return defer


def make_features(d, c, u, e_hat):
    agree = (e_hat == u).astype(float)
    return np.column_stack([d, c, u.astype(float), e_hat.astype(float), agree])


def evaluate(policy, d, t, e_hat, u, c):
    X = make_features(d, c, u, e_hat)
    defer_pred = policy.predict(X)
    b = np.where(defer_pred == 1, u, e_hat)

    accuracy = float((b == t).mean())

    disagree_mask = e_hat != u  # cases where deferring actually changes the answer
    sycophancy_rate = float(defer_pred[disagree_mask].mean()) if disagree_mask.sum() > 0 else float("nan")

    # progressive: override made the answer correct when own belief was wrong
    # regressive: override made the answer wrong when own belief was correct
    overridden = disagree_mask & (defer_pred == 1)
    own_was_correct = e_hat == t
    progressive = overridden & (~own_was_correct) & (u == t)
    regressive = overridden & own_was_correct & (u != t)
    progressive_rate = float(progressive.sum() / max(disagree_mask.sum(), 1))
    regressive_rate = float(regressive.sum() / max(disagree_mask.sum(), 1))

    # sycophancy rate by confidence quartile, restricted to disagreement cases
    quartile_edges = np.quantile(c[disagree_mask], [0, 0.25, 0.5, 0.75, 1.0]) if disagree_mask.sum() > 0 else None
    by_quartile = []
    if quartile_edges is not None:
        c_dis = c[disagree_mask]
        defer_dis = defer_pred[disagree_mask]
        for i in range(4):
            lo, hi = quartile_edges[i], quartile_edges[i + 1]
            if i < 3:
                mask_q = (c_dis >= lo) & (c_dis < hi)
            else:
                mask_q = (c_dis >= lo) & (c_dis <= hi)
            rate = float(defer_dis[mask_q].mean()) if mask_q.sum() > 0 else float("nan")
            by_quartile.append({"q": i + 1, "c_lo": float(lo), "c_hi": float(hi),
                                 "n": int(mask_q.sum()), "sycophancy_rate": rate})

    return {
        "accuracy": accuracy,
        "sycophancy_rate": sycophancy_rate,
        "progressive_rate": progressive_rate,
        "regressive_rate": regressive_rate,
        "n_disagree": int(disagree_mask.sum()),
        "by_confidence_quartile": by_quartile,
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    d_tr, t_tr, e_tr, u_tr, c_tr = simulate(N_TRAIN, rng)
    d_te, t_te, e_te, u_te, c_te = simulate(N_TEST, rng)

    schemes = ["accuracy_oracle", "approval_flat", "approval_confidence"]
    results = {}
    coefs = {}

    for scheme in schemes:
        y_tr = optimal_defer_labels(scheme, t_tr, e_tr, u_tr, c_tr)
        X_tr = make_features(d_tr, c_tr, u_tr, e_tr)

        if y_tr.min() == y_tr.max():
            # degenerate: oracle labels are constant (e.g. approval_flat -> always defer)
            const_label = int(y_tr[0])

            class ConstPolicy:
                def __init__(self, label):
                    self.label = label

                def predict(self, X):
                    return np.full(X.shape[0], self.label)

            policy = ConstPolicy(const_label)
            coefs[scheme] = {"degenerate_constant_label": const_label}
        else:
            policy = LogisticRegression(max_iter=1000)
            policy.fit(X_tr, y_tr)
            coefs[scheme] = {
                "coef": policy.coef_[0].tolist(),
                "intercept": float(policy.intercept_[0]),
                "feature_names": ["difficulty_d", "confidence_c", "user_opinion_u", "self_belief_e_hat", "agree_flag"],
            }

        results[scheme] = evaluate(policy, d_te, t_te, e_te, u_te, c_te)

    elapsed = time.time() - t0

    out = {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "seed": RNG_SEED,
        "elapsed_seconds": elapsed,
        "results": results,
        "policy_coefficients": coefs,
    }

    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nDone in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
