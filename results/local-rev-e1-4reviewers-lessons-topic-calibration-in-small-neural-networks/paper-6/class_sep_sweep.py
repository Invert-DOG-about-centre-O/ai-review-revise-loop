"""
Orthogonal manipulation: does the width-ECE sign flip on the synthetic dataset
track class separability? Sweeps class_sep at fixed width grid, fewer seeds
than the main experiment (this is a follow-up robustness check, not a
replacement for the pre-registered primary analysis).
"""
import json
import sys
import time
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
N_SEEDS = 10
MAX_ITER = 1500
N_BINS = 15
CLASS_SEPS = [0.5, 1.0, 1.5, 2.0, 3.0]
BASE_SEED = 30_000


def relu(x):
    return np.maximum(0, x)


def forward_logits(clf, X):
    a = X
    n_layers = len(clf.coefs_)
    for i in range(n_layers):
        z = a @ clf.coefs_[i] + clf.intercepts_[i]
        a = relu(z) if i < n_layers - 1 else z
    return a


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, y, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return total


def main():
    t0 = time.time()
    results = []
    for cs_idx, class_sep in enumerate(CLASS_SEPS):
        Xs, ys = make_classification(
            n_samples=1800, n_features=20, n_informative=10, n_redundant=5,
            n_classes=4, n_clusters_per_class=2, class_sep=class_sep, flip_y=0.03,
            random_state=42,
        )
        Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8)

        for width_idx, width in enumerate(WIDTHS):
            for seed_idx in range(N_SEEDS):
                seed = BASE_SEED + cs_idx * 100_000 + width_idx * 1000 + seed_idx
                X_train, X_rest, y_train, y_rest = train_test_split(
                    Xs, ys, test_size=0.5, random_state=seed, stratify=ys
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
                )
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always", ConvergenceWarning)
                    clf = MLPClassifier(
                        hidden_layer_sizes=(width,), activation="relu", solver="lbfgs",
                        max_iter=MAX_ITER, random_state=seed, early_stopping=False,
                    )
                    clf.fit(X_train, y_train)
                    conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in wlist)

                logits_test = forward_logits(clf, X_test)
                probs_test = softmax(logits_test)
                ece_pre = ece(probs_test, y_test)
                acc = float((probs_test.argmax(axis=1) == y_test).mean())

                results.append({
                    "class_sep": class_sep, "width": width, "seed_idx": seed_idx,
                    "acc": acc, "ece_pre": ece_pre, "converged": not conv_warn,
                })
        print(f"class_sep={class_sep} done at {time.time()-t0:.1f}s", file=sys.stderr)

    with open("class_sep_sweep_results.json", "w") as f:
        json.dump({"results": results, "config": {"widths": WIDTHS, "n_seeds": N_SEEDS,
                    "class_seps": CLASS_SEPS, "max_iter": MAX_ITER}}, f, indent=2)

    # Per-class_sep H1 (width vs ece_pre)
    summary = []
    for class_sep in CLASS_SEPS:
        rows = [r for r in results if r["class_sep"] == class_sep]
        logw = [np.log2(r["width"]) for r in rows]
        e = [r["ece_pre"] for r in rows]
        rho, p = spearmanr(logw, e)
        mean_acc = np.mean([r["acc"] for r in rows])
        summary.append({"class_sep": class_sep, "rho": rho, "p": p, "n": len(rows), "mean_acc": mean_acc})
        print(f"class_sep={class_sep}: rho={rho:.3f} p={p:.2e} mean_acc={mean_acc:.3f}", file=sys.stderr)

    with open("class_sep_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"TOTAL elapsed: {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
