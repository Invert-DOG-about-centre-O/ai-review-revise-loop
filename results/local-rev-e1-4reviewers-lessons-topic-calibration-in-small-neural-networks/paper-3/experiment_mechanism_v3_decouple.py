"""
Mechanism check v3: decouple pre-ECE from width (requested independently by
three of four round-3 reviewers, who noted the v2 harm-rate check compares
different fixed widths and so cannot separate "already well-calibrated" from
"is width 16/64" as the driver).

Design: hold width FIXED at 16 (digits). Induce a spread of pre-ECE *within*
that single width by varying max_iter (training length) instead of width:
early-stopped models are typically less confident/less calibrated than fully
trained ones at the same width. Train 6 models per max_iter value x 5 seeds
= 30 models, all width=16. Split into low- vs high-pre-ECE halves (median
split within width=16) and run the same bootstrap harm-rate check on each
half at a fixed calibration-set size. If pre-ECE level (not width) drives the
harm rate, the low-pre-ECE half should show a high harm rate and the
high-pre-ECE half a lower one, matching the qualitative pattern in Table 3.3
despite width being held constant throughout.
"""
import json, time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize_scalar

T0 = time.time()
BASE_SEED = 20260827
WIDTH = 16
MAX_ITERS = [10, 20, 40, 80, 150, 300]
N_SEEDS_PER = 5
CAL_SIZE = 120
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
    models = []

    for max_iter in MAX_ITERS:
        for s in range(N_SEEDS_PER):
            seed = BASE_SEED + 7_000_000 + max_iter * 1000 + s
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

            clf = MLPClassifier(hidden_layer_sizes=(WIDTH,), activation="relu",
                                 solver="adam", alpha=1e-4, max_iter=max_iter,
                                 random_state=seed)
            clf.fit(X_train, y_train)
            test_probs = clf.predict_proba(X_test)
            pre_ece = ece(test_probs, y_test)
            pool_probs = clf.predict_proba(X_valpool)
            pool_n = len(y_valpool)
            test_logits = probs_to_pseudologits(test_probs)

            if pool_n < CAL_SIZE:
                continue
            rng = np.random.RandomState(BASE_SEED + 9_800_000 + max_iter * 10000 + s)
            post_eces = []
            for b in range(N_BOOT):
                idx = rng.choice(pool_n, size=CAL_SIZE, replace=False)
                T_star = fit_temperature(pool_probs[idx], y_valpool[idx])
                calibrated = softmax_with_temp(test_logits, T_star)
                post_eces.append(ece(calibrated, y_test))
            frac_worse = float(np.mean(np.array(post_eces) > pre_ece))
            models.append(dict(max_iter=max_iter, seed_idx=s, pre_ece=pre_ece,
                                frac_worse=frac_worse))
        print(f"max_iter={max_iter} done at t={time.time()-T0:.1f}s")

    with open("results_mechanism_v3_decouple.json", "w") as f:
        json.dump(models, f, indent=2)

    pre_eces = np.array([m["pre_ece"] for m in models])
    med = float(np.median(pre_eces))
    low = [m for m in models if m["pre_ece"] <= med]
    high = [m for m in models if m["pre_ece"] > med]
    summary = dict(
        width=WIDTH, n_models=len(models), median_pre_ece=med,
        low_group=dict(n=len(low), mean_pre_ece=float(np.mean([m["pre_ece"] for m in low])),
                        mean_frac_worse=float(np.mean([m["frac_worse"] for m in low])),
                        sd_frac_worse=float(np.std([m["frac_worse"] for m in low], ddof=1))),
        high_group=dict(n=len(high), mean_pre_ece=float(np.mean([m["pre_ece"] for m in high])),
                         mean_frac_worse=float(np.mean([m["frac_worse"] for m in high])),
                         sd_frac_worse=float(np.std([m["frac_worse"] for m in high], ddof=1))),
    )
    with open("results_mechanism_v3_decouple_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"TOTAL ELAPSED: {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
