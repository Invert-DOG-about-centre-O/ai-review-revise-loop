"""
Mechanism check v2: extends experiment_mechanism.py to use 5 independently
trained models per width (instead of 1), to separate calibration-set-size
effects from idiosyncrasies of a single trained model (raised independently
by two reviewers of v1).
"""
import json
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize_scalar

T0 = time.time()
BASE_SEED = 20260827
WIDTHS = [4, 16, 64]
CAL_SIZES = [30, 60, 120, 240]
N_MODEL_SEEDS = 5
N_BOOT = 100
N_BINS = 15
EPS = 1e-12


def ece(probs, labels, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def probs_to_pseudologits(p):
    return np.log(np.clip(p, EPS, 1.0))


def softmax_with_temp(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_probs, val_labels):
    logits = probs_to_pseudologits(val_probs)

    def nll(T):
        p = softmax_with_temp(logits, T)
        p_true = p[np.arange(len(val_labels)), val_labels]
        return -np.log(np.clip(p_true, EPS, 1.0)).mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                           options={"xatol": 1e-4})
    return float(res.x)


def main():
    d = load_digits()
    X, y = d.data.astype(np.float64), d.target
    results = []

    for width in WIDTHS:
        for model_idx in range(N_MODEL_SEEDS):
            seed = BASE_SEED + 6_000_000 + width * 1000 + model_idx
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.5, random_state=seed, stratify=y
            )
            X_valpool, X_test, y_valpool, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
            )
            mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
            X_train = (X_train - mu) / sd
            X_valpool = (X_valpool - mu) / sd
            X_test = (X_test - mu) / sd

            clf = MLPClassifier(hidden_layer_sizes=(width,), activation="relu",
                                 solver="adam", alpha=1e-4, max_iter=300,
                                 random_state=seed)
            clf.fit(X_train, y_train)
            test_probs = clf.predict_proba(X_test)
            pre_ece = ece(test_probs, y_test)
            pool_probs = clf.predict_proba(X_valpool)
            pool_n = len(y_valpool)
            test_logits = probs_to_pseudologits(test_probs)

            for cal_size in CAL_SIZES:
                if cal_size > pool_n:
                    continue
                T_stars, post_eces = [], []
                rng = np.random.RandomState(
                    BASE_SEED + 9_500_000 + width * 10000 + model_idx * 1000 + cal_size)
                for b in range(N_BOOT):
                    idx = rng.choice(pool_n, size=cal_size, replace=False)
                    T_star = fit_temperature(pool_probs[idx], y_valpool[idx])
                    calibrated = softmax_with_temp(test_logits, T_star)
                    post_eces.append(ece(calibrated, y_test))
                    T_stars.append(T_star)
                results.append(dict(
                    width=width, model_idx=model_idx, cal_size=cal_size,
                    pre_ece=pre_ece,
                    T_star_mean=float(np.mean(T_stars)),
                    T_star_sd=float(np.std(T_stars, ddof=1)),
                    post_ece_mean=float(np.mean(post_eces)),
                    post_ece_sd=float(np.std(post_eces, ddof=1)),
                    frac_worse_than_pre=float(np.mean(np.array(post_eces) > pre_ece)),
                ))
        print(f"width={width} done at t={time.time()-T0:.1f}s")

    with open("results_mechanism_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    # aggregate across the 5 models per (width, cal_size)
    agg = {}
    for width in WIDTHS:
        for cal_size in CAL_SIZES:
            rows = [r for r in results if r["width"] == width and r["cal_size"] == cal_size]
            if not rows:
                continue
            key = f"w{width}_cs{cal_size}"
            agg[key] = dict(
                n_models=len(rows),
                mean_pre_ece=float(np.mean([r["pre_ece"] for r in rows])),
                mean_T_star=float(np.mean([r["T_star_mean"] for r in rows])),
                mean_frac_worse=float(np.mean([r["frac_worse_than_pre"] for r in rows])),
                sd_frac_worse_across_models=float(np.std([r["frac_worse_than_pre"] for r in rows], ddof=1)),
            )
    with open("results_mechanism_v2_agg.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"TOTAL ELAPSED: {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
