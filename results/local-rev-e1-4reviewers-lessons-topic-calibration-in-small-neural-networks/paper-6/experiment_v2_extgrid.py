"""
Round-3 review response: all four reviewers independently asked why the
grid-ceiling fix (grid_ceiling_check.py) was applied only to a 100-run
diagnostic instead of being propagated through the primary 320-run pipeline.
This script re-runs the FULL primary sweep (both datasets, all 8 widths,
20 seeds/cell) with the temperature grid extended from [0.1, 20] to
[0.1, 500], identical in every other respect to experiment.py (same seed
table, same splits, same model config). It also records, per run, whether
the ORIGINAL [0.1, 20] grid would have been pinned at its ceiling, so we
can report pinning stats for `digits` too (previously only argued to be
unlikely, not checked).
"""
import json
import sys
import time
import warnings

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
N_SEEDS = 20
MAX_ITER = 1500
N_BINS = 15
BASE_SEEDS = {"digits": 10_000, "synthetic": 20_000}


def get_seed(dataset_name, width_idx, seed_idx):
    return BASE_SEEDS[dataset_name] + width_idx * 1000 + seed_idx


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


def nll_from_logits(logits, y, T=1.0):
    probs = softmax(logits / T)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(probs[np.arange(len(y)), y]))


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
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        total += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return total


def brier_multiclass(probs, y, n_classes):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def fit_temperature(logits_val, y_val, grid_max):
    # coarse grid to grid_max, then local refine -- same method as experiment.py,
    # just with the ceiling raised from 20.0 to 500.0
    n_hi = 30 if grid_max <= 20.0 else 90
    grid = np.concatenate([np.linspace(0.1, 5.0, 50), np.linspace(5.0, grid_max, n_hi)])
    losses = [nll_from_logits(logits_val, y_val, T) for T in grid]
    best_T = grid[int(np.argmin(losses))]
    fine = np.linspace(max(0.05, best_T - 0.2), best_T + 0.2, 41)
    losses_fine = [nll_from_logits(logits_val, y_val, T) for T in fine]
    return float(fine[int(np.argmin(losses_fine))])


def run_dataset(dataset_name, X, y, n_classes, results, warn_log):
    for width_idx, width in enumerate(WIDTHS):
        for seed_idx in range(N_SEEDS):
            seed = get_seed(dataset_name, width_idx, seed_idx)
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.5, random_state=seed, stratify=y
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

            if conv_warn:
                warn_log.append({"dataset": dataset_name, "width": width, "seed_idx": seed_idx})

            logits_val = forward_logits(clf, X_val)
            logits_test = forward_logits(clf, X_test)

            probs_test_pre = softmax(logits_test)
            ece_pre = ece(probs_test_pre, y_test)
            nll_pre = nll_from_logits(logits_test, y_test, T=1.0)
            brier_pre = brier_multiclass(probs_test_pre, y_test, n_classes)
            acc = float((probs_test_pre.argmax(axis=1) == y_test).mean())

            T_star_orig = fit_temperature(logits_val, y_val, grid_max=20.0)
            T_star_ext = fit_temperature(logits_val, y_val, grid_max=500.0)
            pinned_orig = bool(T_star_orig >= 19.9)

            probs_test_post = softmax(logits_test / T_star_ext)
            ece_post = ece(probs_test_post, y_test)
            nll_post = nll_from_logits(logits_test, y_test, T=T_star_ext)
            brier_post = brier_multiclass(probs_test_post, y_test, n_classes)

            results.append({
                "dataset": dataset_name, "width": width, "width_idx": width_idx,
                "seed_idx": seed_idx, "seed": seed,
                "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
                "acc": acc, "ece_pre": ece_pre, "nll_pre": nll_pre, "brier_pre": brier_pre,
                "T_star": T_star_ext, "T_star_orig_grid20": T_star_orig, "pinned_orig_grid20": pinned_orig,
                "ece_post": ece_post, "nll_post": nll_post, "brier_post": brier_post,
                "converged": not conv_warn, "n_iter": int(clf.n_iter_),
            })


def main():
    t0 = time.time()
    results = []
    warn_log = []

    digits = load_digits()
    Xd, yd = digits.data.astype(np.float64), digits.target.astype(int)
    Xd = (Xd - Xd.mean(axis=0)) / (Xd.std(axis=0) + 1e-8)

    Xs, ys = make_classification(
        n_samples=1800, n_features=20, n_informative=10, n_redundant=5,
        n_classes=4, n_clusters_per_class=2, class_sep=1.0, flip_y=0.03,
        random_state=42,
    )
    Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8)

    run_dataset("digits", Xd, yd, n_classes=10, results=results, warn_log=warn_log)
    print(f"digits done at {time.time() - t0:.1f}s", file=sys.stderr)
    run_dataset("synthetic", Xs, ys, n_classes=4, results=results, warn_log=warn_log)
    print(f"synthetic done at {time.time() - t0:.1f}s", file=sys.stderr)

    with open("raw_results_extgrid.json", "w") as f:
        json.dump({"results": results, "warn_log": warn_log,
                   "config": {"widths": WIDTHS, "n_seeds": N_SEEDS, "max_iter": MAX_ITER,
                              "n_bins": N_BINS, "base_seeds": BASE_SEEDS,
                              "T_grid": "[0.1,500] extended; T_star_orig_grid20 also recorded"}}, f, indent=2)

    print(f"TOTAL runs: {len(results)}, non-converged: {len(warn_log)}", file=sys.stderr)
    print(f"Elapsed: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
