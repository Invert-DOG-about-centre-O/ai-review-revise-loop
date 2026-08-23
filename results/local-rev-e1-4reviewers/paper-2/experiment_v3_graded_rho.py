"""
Revision follow-up: graded rho sweep to check whether the aggregation
collapse under correlated rater error is a smooth degradation or a sharp
discontinuity, addressing reviewer requests for rho values between 0 and 0.5.
Fixed alpha=0.75, n_raters=9 (the panel size that fully mitigates independent
error), rho in {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, 20 seeds each.
"""
import numpy as np
import json
import time
from experiment import gen_data, gen_beliefs, evaluate, RATER_Q
from experiment_v2 import train_policy_corr

d = 10
n_train = 4000
n_eval = 2000
n_rounds = 60
lr = 0.5
K = 4
n_seeds = 30


def main():
    t0 = time.time()
    out = {}
    for rho in [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]:
        accs, sycos = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(11000 * seed + int(rho * 1000) + 9)
            w_true = rng.normal(size=d)
            w_true /= np.linalg.norm(w_true)
            X_tr, y_tr = gen_data(n_train, d, w_true, rng)
            X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
            b_ev = gen_beliefs(y_ev, RATER_Q, rng)
            acc, syco = train_policy_corr(X_tr, y_tr, 0.75, n_rounds, K, lr,
                                           n_raters=9, rho=rho, rng=rng,
                                           X_eval=X_ev, y_eval=y_ev, b_eval_indep=b_ev)
            accs.append(acc)
            sycos.append(syco)
        accs, sycos = np.array(accs), np.array(sycos)
        out[f"rho_{rho}"] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
                              "syco_mean": float(sycos.mean()), "syco_std": float(sycos.std())}
        print(f"rho={rho} acc={accs.mean():.3f}+-{accs.std():.3f} syco={sycos.mean():.3f}+-{sycos.std():.3f}  t={time.time()-t0:.1f}s")

    with open("results_v3_graded_rho.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
