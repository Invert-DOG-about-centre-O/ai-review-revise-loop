"""
Revision follow-up: test whether a simple non-naive aggregator (per-candidate
median rater score, instead of naive score-summing) shifts the correlated-error
(rho) failure boundary found in experiment_v3_graded_rho.py. Directly answers
the question raised independently by all four round-3 reviewers: does anything
cheap beat naive aggregation once rater errors are correlated?

Median is a natural first non-naive baseline: it is robust to a captured
minority of raters (per item, per round) as long as fewer than half of the
n_raters=9 panel adopts the shared wrong belief on that item; naive summing
has no such protection.

Same task/policy/rater model as experiment.py; only the cross-rater
aggregation rule (sum vs. median) differs. alpha=0.75, n_raters=9 throughout
(the fully-mitigating panel size for independent error in Sec 3.2).
"""
import numpy as np
import json
import time
from experiment import gen_data, gen_beliefs, evaluate, rater_score, sigmoid, RATER_Q
from experiment_v2 import gen_beliefs_correlated

d = 10
n_train = 4000
n_eval = 2000
n_rounds = 60
lr = 0.5
K = 4
n_raters = 9
alpha = 0.75
n_seeds = 30


def train_policy_robust(X_tr, y_tr, agg, n_rounds, K, lr, n_raters, rho, rng,
                         X_eval, y_eval, b_eval_indep):
    """agg: 'sum' (naive, as in the paper so far) or 'median' (non-naive)."""
    w = np.zeros(d)
    n = X_tr.shape[0]
    shared_wrong_tr = 1 - y_tr
    for r in range(n_rounds):
        idx = rng.integers(0, n, size=min(256, n))
        Xb, yb = X_tr[idx], y_tr[idx]
        sw = shared_wrong_tr[idx]
        p = sigmoid(Xb @ w)
        cand = (rng.uniform(size=(K, len(idx))) < p[None, :]).astype(int)
        rater_scores = np.zeros((n_raters, K, len(idx)), dtype=float)
        for ri in range(n_raters):
            b_use = gen_beliefs_correlated(yb, RATER_Q, rho, rng, sw)
            for k in range(K):
                rater_scores[ri, k] = rater_score(cand[k], yb, b_use, alpha, rng)
        if agg == "sum":
            total_score = rater_scores.sum(axis=0)
        elif agg == "median":
            total_score = np.median(rater_scores, axis=0)
        else:
            raise ValueError(agg)
        best_k = np.argmax(total_score, axis=0)
        chosen = cand[best_k, np.arange(len(idx))]
        p_now = sigmoid(Xb @ w)
        grad = Xb.T @ (chosen - p_now) / len(idx)
        w += lr * grad
    return evaluate(w, X_eval, y_eval, b_eval_indep)


def main():
    t0 = time.time()
    out = {}
    rhos = [0.0, 0.3, 0.4, 0.44, 0.48, 0.5]
    for agg in ["sum", "median"]:
        for rho in rhos:
            accs, sycos = [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng(13000 * seed + int(rho * 1000) + (1 if agg == "median" else 0))
                w_true = rng.normal(size=d)
                w_true /= np.linalg.norm(w_true)
                X_tr, y_tr = gen_data(n_train, d, w_true, rng)
                X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
                b_ev = gen_beliefs(y_ev, RATER_Q, rng)
                acc, syco = train_policy_robust(X_tr, y_tr, agg, n_rounds, K, lr,
                                                 n_raters=n_raters, rho=rho, rng=rng,
                                                 X_eval=X_ev, y_eval=y_ev, b_eval_indep=b_ev)
                accs.append(acc)
                sycos.append(syco)
            accs, sycos = np.array(accs), np.array(sycos)
            out[f"{agg}_rho_{rho}"] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
                                        "syco_mean": float(sycos.mean()), "syco_std": float(sycos.std())}
            print(f"agg={agg} rho={rho} acc={accs.mean():.3f}+-{accs.std():.3f} "
                  f"syco={sycos.mean():.3f}+-{sycos.std():.3f}  t={time.time()-t0:.1f}s")

    with open("results_v4_robust_agg.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
