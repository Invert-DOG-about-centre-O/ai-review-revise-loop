"""
Follow-up experiments for the revision: (1) significance tests on the
alpha-sweep and rater-aggregation comparisons, (2) a correlated-rater-error
condition (raters share a common wrong belief with probability rho) to test
whether aggregation degrades gracefully, (3) a hyperparameter sensitivity
check (K, rater-score noise scale) on the threshold location.

Reuses gen_data / gen_beliefs / rater_score / evaluate from experiment.py.
"""
import numpy as np
import json
import time
from scipy import stats
from experiment import gen_data, gen_beliefs, rater_score, evaluate, sigmoid, RATER_Q

d = 10
n_train = 4000
n_eval = 2000
n_rounds = 60
lr = 0.5
n_seeds = 30


def gen_beliefs_correlated(y, q, rho, rng, shared_wrong):
    """With probability rho, a rater adopts a shared common (wrong) belief
    instead of drawing its own independent belief. shared_wrong is a fixed
    array of wrong labels (1-y) representing "the same wrong claim online"."""
    b = gen_beliefs(y, q, rng)
    use_shared = rng.uniform(size=len(y)) < rho
    b = b.copy()
    b[use_shared] = shared_wrong[use_shared]
    return b


def train_policy_corr(X_tr, y_tr, alpha, n_rounds, K, lr, n_raters, rho, rng,
                       X_eval, y_eval, b_eval_indep):
    w = np.zeros(d)
    n = X_tr.shape[0]
    shared_wrong_tr = 1 - y_tr  # the single shared misinformation claim
    for r in range(n_rounds):
        idx = rng.integers(0, n, size=min(256, n))
        Xb, yb = X_tr[idx], y_tr[idx]
        sw = shared_wrong_tr[idx]
        p = sigmoid(Xb @ w)
        cand = (rng.uniform(size=(K, len(idx))) < p[None, :]).astype(int)
        total_score = np.zeros_like(cand, dtype=float)
        for _ in range(n_raters):
            b_use = gen_beliefs_correlated(yb, RATER_Q, rho, rng, sw)
            for k in range(K):
                total_score[k] += rater_score(cand[k], yb, b_use, alpha, rng)
        best_k = np.argmax(total_score, axis=0)
        chosen = cand[best_k, np.arange(len(idx))]
        p_now = sigmoid(Xb @ w)
        grad = Xb.T @ (chosen - p_now) / len(idx)
        w += lr * grad
    acc, syco = evaluate(w, X_eval, y_eval, b_eval_indep)
    return acc, syco


def train_policy_hp(X_tr, y_tr, b_tr, alpha, n_rounds, K, lr, noise_scale, rng,
                     X_eval, y_eval, b_eval):
    w = np.zeros(d)
    n = X_tr.shape[0]
    for r in range(n_rounds):
        idx = rng.integers(0, n, size=min(256, n))
        Xb, yb, bb = X_tr[idx], y_tr[idx], b_tr[idx]
        p = sigmoid(Xb @ w)
        cand = (rng.uniform(size=(K, len(idx))) < p[None, :]).astype(int)
        total_score = np.zeros_like(cand, dtype=float)
        for k in range(K):
            noise = rng.normal(scale=noise_scale, size=cand[k].shape)
            total_score[k] = (alpha * (cand[k] == bb).astype(float) +
                               (1 - alpha) * (cand[k] == yb).astype(float) + noise)
        best_k = np.argmax(total_score, axis=0)
        chosen = cand[best_k, np.arange(len(idx))]
        p_now = sigmoid(Xb @ w)
        grad = Xb.T @ (chosen - p_now) / len(idx)
        w += lr * grad
    return evaluate(w, X_eval, y_eval, b_eval)


def run_condition(alpha, n_raters, K=4, seeds=n_seeds, seed_offset=5000):
    from experiment import train_policy
    accs, sycos = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed_offset * seed + int(alpha * 1000) + n_raters)
        w_true = rng.normal(size=d)
        w_true /= np.linalg.norm(w_true)
        X_tr, y_tr = gen_data(n_train, d, w_true, rng)
        b_tr = gen_beliefs(y_tr, RATER_Q, rng)
        X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
        b_ev = gen_beliefs(y_ev, RATER_Q, rng)
        w, _ = train_policy(X_tr, y_tr, b_tr, alpha, n_rounds, K, lr,
                             n_raters=n_raters, rng=rng,
                             X_eval=X_ev, y_eval=y_ev, b_eval=b_ev, eval_every=1000)
        acc, syco = evaluate(w, X_ev, y_ev, b_ev)
        accs.append(acc)
        sycos.append(syco)
    return np.array(accs), np.array(sycos)


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def main():
    t0 = time.time()
    out = {}

    # --- Part 1: significance tests on key comparisons (fresh per-seed data) ---
    print("Part 1: significance tests")
    conds = {}
    for alpha in [0.0, 0.5, 0.75, 1.0]:
        conds[f"alpha_{alpha}"] = run_condition(alpha, n_raters=1)
        a, s = conds[f"alpha_{alpha}"]
        print(f"  alpha={alpha} acc={a.mean():.3f}+-{a.std():.3f} syco={s.mean():.3f}+-{s.std():.3f}  t={time.time()-t0:.1f}s")
    for nr in [1, 3, 9]:
        conds[f"nraters_{nr}"] = run_condition(0.75, n_raters=nr)
        a, s = conds[f"nraters_{nr}"]
        print(f"  n_raters={nr} acc={a.mean():.3f}+-{a.std():.3f} syco={s.mean():.3f}+-{s.std():.3f}  t={time.time()-t0:.1f}s")

    sig = {}
    pairs = [("alpha_0.0", "alpha_0.5"), ("alpha_0.5", "alpha_0.75"),
             ("alpha_0.75", "alpha_1.0"), ("nraters_1", "nraters_3"), ("nraters_1", "nraters_9")]
    for k1, k2 in pairs:
        a1, s1 = conds[k1]
        a2, s2 = conds[k2]
        t_acc, p_acc = welch(a1, a2)
        t_syc, p_syc = welch(s1, s2)
        sig[f"{k1}_vs_{k2}"] = {"acc_t": t_acc, "acc_p": p_acc, "syco_t": t_syc, "syco_p": p_syc}
        print(f"  {k1} vs {k2}: acc p={p_acc:.2e}, syco p={p_syc:.2e}")
    out["significance"] = sig
    out["conds_summary"] = {
        name: {"acc_mean": float(a.mean()), "acc_std": float(a.std()),
               "syco_mean": float(s.mean()), "syco_std": float(s.std())}
        for name, (a, s) in conds.items()}

    # --- Part 2: correlated rater errors ---
    print("Part 2: correlated rater errors (alpha=0.75)")
    corr = {}
    for rho in [0.0, 0.5, 1.0]:
        for nr in [1, 3, 9]:
            accs, sycos = [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng(7000 * seed + int(rho * 100) + nr)
                w_true = rng.normal(size=d)
                w_true /= np.linalg.norm(w_true)
                X_tr, y_tr = gen_data(n_train, d, w_true, rng)
                X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
                b_ev = gen_beliefs(y_ev, RATER_Q, rng)
                acc, syco = train_policy_corr(X_tr, y_tr, 0.75, n_rounds, 4, lr,
                                               n_raters=nr, rho=rho, rng=rng,
                                               X_eval=X_ev, y_eval=y_ev, b_eval_indep=b_ev)
                accs.append(acc)
                sycos.append(syco)
            accs, sycos = np.array(accs), np.array(sycos)
            corr[f"rho_{rho}_nraters_{nr}"] = {
                "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
                "syco_mean": float(sycos.mean()), "syco_std": float(sycos.std())}
            print(f"  rho={rho} n_raters={nr} acc={accs.mean():.3f}+-{accs.std():.3f} "
                  f"syco={sycos.mean():.3f}+-{sycos.std():.3f}  t={time.time()-t0:.1f}s")
    out["correlated_raters"] = corr

    # --- Part 3: hyperparameter sensitivity (K, noise scale) on threshold location ---
    print("Part 3: hyperparameter sensitivity")
    hp = {}
    for K in [2, 4, 8]:
        for noise_scale in [0.05, 0.2]:
            for alpha in [0.5, 0.75]:
                accs, sycos = [], []
                for seed in range(15):
                    rng = np.random.default_rng(9000 * seed + K * 1000 + int(noise_scale * 1000) + int(alpha * 100))
                    w_true = rng.normal(size=d)
                    w_true /= np.linalg.norm(w_true)
                    X_tr, y_tr = gen_data(n_train, d, w_true, rng)
                    b_tr = gen_beliefs(y_tr, RATER_Q, rng)
                    X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
                    b_ev = gen_beliefs(y_ev, RATER_Q, rng)
                    acc, syco = train_policy_hp(X_tr, y_tr, b_tr, alpha, n_rounds, K, lr,
                                                 noise_scale, rng, X_ev, y_ev, b_ev)
                    accs.append(acc)
                    sycos.append(syco)
                accs, sycos = np.array(accs), np.array(sycos)
                hp[f"K{K}_noise{noise_scale}_alpha{alpha}"] = {
                    "acc_mean": float(accs.mean()), "syco_mean": float(sycos.mean())}
        print(f"  K={K} done  t={time.time()-t0:.1f}s")
    out["hyperparam_sensitivity"] = hp

    with open("results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
