"""
Simulation of sycophancy amplification under RLHF-style rejection-sampling
fine-tuning with biased human raters, and a rater-aggregation mitigation.

Pure numpy, CPU-only, small synthetic data. Deterministic given seed.
"""
import numpy as np
import json
import time

RNG_SEED = 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gen_data(n, d, w_true, rng):
    X = rng.normal(size=(n, d))
    p_true = sigmoid(X @ w_true)
    y = (rng.uniform(size=n) < p_true).astype(int)
    return X, y


def gen_beliefs(y, q, rng):
    """User's stated belief about the label: correct w.p. q, else flipped."""
    flip = rng.uniform(size=len(y)) >= q
    b = y.copy()
    b[flip] = 1 - b[flip]
    return b


def rater_score(candidate, y, b, alpha, rng):
    """Score a candidate label from one rater.
    alpha: probability mass the rater's judgment is driven by agreement with
    the user's stated belief b (sycophancy) rather than fact-checking against
    true label y (which a careful/expert rater could verify)."""
    noise = rng.normal(scale=0.05, size=candidate.shape)
    return alpha * (candidate == b).astype(float) + (1 - alpha) * (candidate == y).astype(float) + noise


def train_policy(X_tr, y_tr, b_tr, alpha, n_rounds, K, lr, n_raters, rng,
                  X_eval, y_eval, b_eval, eval_every=1):
    d = X_tr.shape[1]
    w = np.zeros(d)
    n = X_tr.shape[0]
    history = []
    for r in range(n_rounds):
        idx = rng.integers(0, n, size=min(256, n))
        Xb, yb, bb = X_tr[idx], y_tr[idx], b_tr[idx]
        p = sigmoid(Xb @ w)
        # sample K candidate labels per example from current policy
        cand = (rng.uniform(size=(K, len(idx))) < p[None, :]).astype(int)
        # aggregate scores across n_raters independent raters (each with own belief draw
        # when n_raters>1, to model rater diversity mitigation)
        total_score = np.zeros_like(cand, dtype=float)
        for _ in range(n_raters):
            if n_raters == 1:
                b_use = bb
            else:
                b_use = gen_beliefs(yb, q=RATER_Q, rng=rng)
            for k in range(K):
                total_score[k] += rater_score(cand[k], yb, b_use, alpha, rng)
        best_k = np.argmax(total_score, axis=0)
        chosen = cand[best_k, np.arange(len(idx))]
        # standard logistic regression gradient step toward chosen label
        p_now = sigmoid(Xb @ w)
        grad = Xb.T @ (chosen - p_now) / len(idx)
        w += lr * grad

        if r % eval_every == 0 or r == n_rounds - 1:
            acc, syco = evaluate(w, X_eval, y_eval, b_eval)
            history.append({"round": r, "accuracy": acc, "sycophancy_rate": syco})
    return w, history


def evaluate(w, X, y, b):
    p = sigmoid(X @ w)
    pred = (p >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    wrong_belief_mask = b != y
    if wrong_belief_mask.sum() > 0:
        syco = float(np.mean(pred[wrong_belief_mask] == b[wrong_belief_mask]))
    else:
        syco = float("nan")
    return acc, syco


RATER_Q = 0.6  # baseline correctness probability of a single rater's stated belief


def main():
    t0 = time.time()
    rng_master = np.random.default_rng(RNG_SEED)
    d = 10
    n_train = 4000
    n_eval = 2000
    n_rounds = 60
    K = 4
    lr = 0.5
    n_seeds = 30

    results = {"alpha_sweep": {}, "rater_aggregation": {}}

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        accs, sycos, trajs = [], [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 * seed + int(alpha * 100))
            w_true = rng.normal(size=d)
            w_true /= np.linalg.norm(w_true)
            X_tr, y_tr = gen_data(n_train, d, w_true, rng)
            b_tr = gen_beliefs(y_tr, RATER_Q, rng)
            X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
            b_ev = gen_beliefs(y_ev, RATER_Q, rng)

            w, hist = train_policy(X_tr, y_tr, b_tr, alpha, n_rounds, K, lr,
                                    n_raters=1, rng=rng,
                                    X_eval=X_ev, y_eval=y_ev, b_eval=b_ev,
                                    eval_every=5)
            acc, syco = evaluate(w, X_ev, y_ev, b_ev)
            accs.append(acc)
            sycos.append(syco)
            trajs.append(hist)
        results["alpha_sweep"][str(alpha)] = {
            "final_accuracy_mean": float(np.mean(accs)),
            "final_accuracy_std": float(np.std(accs)),
            "final_sycophancy_mean": float(np.mean(sycos)),
            "final_sycophancy_std": float(np.std(sycos)),
            "trajectories": trajs,
        }
        print(f"alpha={alpha:.2f} acc={np.mean(accs):.3f}+-{np.std(accs):.3f} "
              f"syco={np.mean(sycos):.3f}+-{np.std(sycos):.3f}  "
              f"elapsed={time.time()-t0:.1f}s")

    # Rater aggregation mitigation experiment at fixed high alpha
    alpha_fixed = 0.75
    for n_raters in [1, 3, 5, 9]:
        accs, sycos = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(2000 * seed + n_raters)
            w_true = rng.normal(size=d)
            w_true /= np.linalg.norm(w_true)
            X_tr, y_tr = gen_data(n_train, d, w_true, rng)
            b_tr = gen_beliefs(y_tr, RATER_Q, rng)
            X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
            b_ev = gen_beliefs(y_ev, RATER_Q, rng)

            w, hist = train_policy(X_tr, y_tr, b_tr, alpha_fixed, n_rounds, K, lr,
                                    n_raters=n_raters, rng=rng,
                                    X_eval=X_ev, y_eval=y_ev, b_eval=b_ev,
                                    eval_every=10)
            acc, syco = evaluate(w, X_ev, y_ev, b_ev)
            accs.append(acc)
            sycos.append(syco)
        results["rater_aggregation"][str(n_raters)] = {
            "final_accuracy_mean": float(np.mean(accs)),
            "final_accuracy_std": float(np.std(accs)),
            "final_sycophancy_mean": float(np.mean(sycos)),
            "final_sycophancy_std": float(np.std(sycos)),
        }
        print(f"n_raters={n_raters} acc={np.mean(accs):.3f}+-{np.std(accs):.3f} "
              f"syco={np.mean(sycos):.3f}+-{np.std(sycos):.3f}  "
              f"elapsed={time.time()-t0:.1f}s")

    # Baseline: fully fact-checked oracle training (alpha=0) already included above.
    # Baseline: pure supervised training on true labels (upper bound reference)
    accs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(3000 * seed)
        w_true = rng.normal(size=d)
        w_true /= np.linalg.norm(w_true)
        X_tr, y_tr = gen_data(n_train, d, w_true, rng)
        X_ev, y_ev = gen_data(n_eval, d, w_true, rng)
        b_ev = gen_beliefs(y_ev, RATER_Q, rng)
        w = np.zeros(d)
        for r in range(n_rounds):
            idx = rng.integers(0, n_train, size=256)
            p = sigmoid(X_tr[idx] @ w)
            grad = X_tr[idx].T @ (y_tr[idx] - p) / 256
            w += lr * grad
        acc, syco = evaluate(w, X_ev, y_ev, b_ev)
        accs.append(acc)
    results["supervised_oracle_baseline"] = {"final_accuracy_mean": float(np.mean(accs))}
    print(f"supervised oracle baseline acc={np.mean(accs):.3f}  elapsed={time.time()-t0:.1f}s")

    results["config"] = dict(d=d, n_train=n_train, n_eval=n_eval, n_rounds=n_rounds,
                              K=K, lr=lr, n_seeds=n_seeds, RATER_Q=RATER_Q,
                              alpha_fixed_for_aggregation=alpha_fixed)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
