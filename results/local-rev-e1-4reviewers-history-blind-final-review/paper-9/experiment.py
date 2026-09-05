"""
Calibration vs conformal-score-choice study on a synthetic 6-way classifier.
Runs the full pipeline (data gen -> train -> calibrate -> conformal eval)
across multiple seeds so headline numbers can be reported as mean +/- std.
"""
import json
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

N_CLASSES = 6
N_FEATURES = 20
N_TRAIN, N_CAL, N_TEST = 4000, 1500, 1500
NOISE_SIGMA = 1.8
LABEL_NOISE = 0.12
ALPHA = 0.10
N_SEEDS = 10


def make_data(seed):
    rng = np.random.RandomState(seed)
    # Fixed class means shared across seeds (same DGP "shape"), only draws differ.
    mean_rng = np.random.RandomState(12345)
    class_means = mean_rng.randn(N_CLASSES, N_FEATURES) * 2.5

    def sample(n, rng_local, noisy_labels):
        y = rng_local.randint(0, N_CLASSES, size=n)
        X = class_means[y] + rng_local.randn(n, N_FEATURES) * NOISE_SIGMA
        if noisy_labels:
            flip = rng_local.rand(n) < LABEL_NOISE
            y = y.copy()
            y[flip] = rng_local.randint(0, N_CLASSES, size=flip.sum())
        return X, y

    X_train, y_train = sample(N_TRAIN, rng, noisy_labels=True)
    X_cal, y_cal = sample(N_CAL, rng, noisy_labels=False)
    X_test, y_test = sample(N_TEST, rng, noisy_labels=False)
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def fit_temperature(logits, y, iters=200, lr=0.05):
    # 1-parameter NLL minimization via simple gradient descent (no external deps).
    T = 1.0
    y = np.asarray(y)
    for _ in range(iters):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        # d NLL / d T
        true_p = p[np.arange(len(y)), y]
        grad = -np.mean((logits[np.arange(len(y)), y] - (p * logits).sum(axis=1)) * (-1.0 / T**2))
        T -= lr * np.sign(grad) * min(abs(grad), 1.0) * 0.05
        T = max(T, 1e-3)
    return T


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_logits(proba, eps=1e-9):
    p = np.clip(proba, eps, 1 - eps)
    return np.log(p)


def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        e += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return e


def nll(probs, y, eps=1e-9):
    p = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def brier(probs, y):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def conformal_lac(cal_probs, y_cal, test_probs, alpha):
    scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(y_cal)
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    pred_sets = test_probs >= (1 - qhat)
    return qhat, pred_sets


def conformal_aps(cal_probs, y_cal, test_probs, alpha):
    def cum_mass_to_true(probs, y):
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])
        return cum[np.arange(len(y)), rank]

    scores = cum_mass_to_true(cal_probs, y_cal)
    n = len(y_cal)
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")

    order = np.argsort(-test_probs, axis=1)
    sorted_p = np.take_along_axis(test_probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    include = cum <= qhat
    include[:, 0] = True  # top class always included
    pred_sets = np.zeros_like(test_probs, dtype=bool)
    for i in range(len(test_probs)):
        pred_sets[i, order[i][include[i]]] = True
    return qhat, pred_sets


def eval_conformal(pred_sets, y_test):
    covered = pred_sets[np.arange(len(y_test)), y_test]
    set_sizes = pred_sets.sum(axis=1)
    empty_frac = (set_sizes == 0).mean()
    return covered.mean(), set_sizes.mean(), empty_frac


def run_seed(seed):
    X_train, y_train, X_cal, y_cal, X_test, y_test = make_data(seed)

    base = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                          solver="adam", max_iter=80, random_state=seed,
                          batch_size=len(X_train))
    base.fit(X_train, y_train)

    def logits_of(model, X):
        proba = model.predict_proba(X)
        return to_logits(proba), proba

    ens_models = []
    for k in range(5):
        Xb, yb = resample(X_train, y_train, random_state=seed * 100 + k)
        m = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                           solver="adam", max_iter=80,
                           random_state=seed * 100 + k, batch_size=len(Xb))
        m.fit(Xb, yb)
        ens_models.append(m)

    def ensemble_proba(X):
        return np.mean([m.predict_proba(X) for m in ens_models], axis=0)

    results = {}

    def make_variant(name, cal_proba_fn, test_proba_fn, temp_scale):
        cal_proba = cal_proba_fn(X_cal)
        test_proba = test_proba_fn(X_test)
        if temp_scale:
            cal_logits = to_logits(cal_proba)
            T = fit_temperature(cal_logits, y_cal)
            cal_proba = softmax(cal_logits / T)
            test_proba = softmax(to_logits(test_proba) / T)
        else:
            T = 1.0
        m = {
            "T": T,
            "accuracy": float((test_proba.argmax(axis=1) == y_test).mean()),
            "nll": float(nll(test_proba, y_test)),
            "brier": float(brier(test_proba, y_test)),
            "ece": float(ece(test_proba, y_test)),
            "mean_confidence": float(test_proba.max(axis=1).mean()),
        }
        for score_name, fn in [("LAC", conformal_lac), ("APS", conformal_aps)]:
            qhat, pred_sets = fn(cal_proba, y_cal, test_proba, ALPHA)
            cov, size, empty = eval_conformal(pred_sets, y_test)
            m[score_name] = {"qhat": float(qhat), "coverage": float(cov),
                              "avg_set_size": float(size), "empty_frac": float(empty)}
        results[name] = m

    make_variant("raw_softmax", lambda X: base.predict_proba(X), lambda X: base.predict_proba(X), False)
    make_variant("temp_scaled", lambda X: base.predict_proba(X), lambda X: base.predict_proba(X), True)
    make_variant("ensemble", ensemble_proba, ensemble_proba, False)
    make_variant("ensemble_temp_scaled", ensemble_proba, ensemble_proba, True)

    return results


def main():
    t0 = time.time()
    all_runs = [run_seed(seed) for seed in range(N_SEEDS)]

    methods = ["raw_softmax", "temp_scaled", "ensemble", "ensemble_temp_scaled"]
    summary = {}
    for m in methods:
        summary[m] = {}
        for key in ["ece", "accuracy", "nll", "brier"]:
            vals = [r[m][key] for r in all_runs]
            summary[m][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for score in ["LAC", "APS"]:
            summary[m][score] = {}
            for key in ["coverage", "avg_set_size", "empty_frac"]:
                vals = [r[m][score][key] for r in all_runs]
                summary[m][score][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out = {
        "n_seeds": N_SEEDS,
        "per_seed": all_runs,
        "summary": summary,
        "wallclock_sec": time.time() - t0,
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Done in {out['wallclock_sec']:.1f}s over {N_SEEDS} seeds")
    for m in methods:
        s = summary[m]
        print(f"{m}: ECE={s['ece']['mean']:.4f}+/-{s['ece']['std']:.4f} "
              f"LAC_size={s['LAC']['avg_set_size']['mean']:.4f}+/-{s['LAC']['avg_set_size']['std']:.4f} "
              f"APS_size={s['APS']['avg_set_size']['mean']:.4f}+/-{s['APS']['avg_set_size']['std']:.4f}")


if __name__ == "__main__":
    main()
