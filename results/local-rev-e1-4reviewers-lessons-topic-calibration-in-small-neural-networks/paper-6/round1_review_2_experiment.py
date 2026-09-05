"""
Calibration vs. width in small MLPs, with post-hoc temperature scaling.

Design:
  - Two datasets: sklearn `digits` (real, 10-class, 8x8 images) and a synthetic
    multiclass `make_classification` set (4 classes). Held fixed across widths
    so width is the only lever varied (per-run design note: architecture family
    and epoch budget are held fixed; only hidden-layer width changes).
  - Width grid: [2, 4, 8, 16, 32, 64, 128, 256] hidden units (single hidden
    layer MLP), fixed max_iter (training-length budget) across all widths so
    training length is not a confound.
  - For each (dataset, width) cell: N_SEEDS independent seeds. Each seed gets
    its own explicit integer seed (no hash()-derived seeds) used for both the
    train/val/test split and the MLP initialization/SGD, and re-seeding
    happens immediately before that unit of training (so no run's model
    depends on RNG state left over from a previous run in the loop).
  - Splits: 50% train / 25% val (temperature fit) / 25% test (evaluation).
  - Metrics on test set: ECE (15-bin), Brier score (multiclass), NLL, before
    and after temperature scaling. Optimal temperature T* fit on val by
    minimizing NLL via 1-D grid + refinement.
  - Logits are recovered by manually replaying the trained MLPClassifier's
    forward pass (coefs_, intercepts_, relu) to get pre-softmax scores, since
    sklearn's predict_proba only exposes post-softmax probabilities.
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

# Explicit fixed seed table -- never derived from hash(). Each (dataset, width, seed_idx)
# maps to one deterministic integer seed used for both splitting and model init/training.
BASE_SEEDS = {"digits": 10_000, "synthetic": 20_000}


def get_seed(dataset_name, width_idx, seed_idx):
    # deterministic integer combination, fixed mapping, no hashing of strings
    return BASE_SEEDS[dataset_name] + width_idx * 1000 + seed_idx


def relu(x):
    return np.maximum(0, x)


def forward_logits(clf, X):
    """Manually replay the trained MLPClassifier's forward pass to recover
    pre-softmax logits (sklearn only exposes post-softmax probabilities)."""
    a = X
    n_layers = len(clf.coefs_)
    for i in range(n_layers):
        z = a @ clf.coefs_[i] + clf.intercepts_[i]
        if i < n_layers - 1:
            a = relu(z)
        else:
            a = z  # final layer: pre-softmax logits
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


def fit_temperature(logits_val, y_val):
    # 1-D search over T minimizing NLL on val set: coarse grid then refine.
    grid = np.concatenate([np.linspace(0.1, 5.0, 50), np.linspace(5.0, 20.0, 30)])
    losses = [nll_from_logits(logits_val, y_val, T) for T in grid]
    best_T = grid[int(np.argmin(losses))]
    fine = np.linspace(max(0.05, best_T - 0.2), best_T + 0.2, 41)
    losses_fine = [nll_from_logits(logits_val, y_val, T) for T in fine]
    return float(fine[int(np.argmin(losses_fine))])


def run_dataset(dataset_name, X, y, n_classes, results, warn_log):
    for width_idx, width in enumerate(WIDTHS):
        for seed_idx in range(N_SEEDS):
            seed = get_seed(dataset_name, width_idx, seed_idx)
            rng_split = np.random.RandomState(seed)
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.5, random_state=seed, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
            )

            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = MLPClassifier(
                    hidden_layer_sizes=(width,),
                    activation="relu",
                    solver="lbfgs",
                    max_iter=MAX_ITER,
                    random_state=seed,
                    early_stopping=False,
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

            T_star = fit_temperature(logits_val, y_val)
            probs_test_post = softmax(logits_test / T_star)
            ece_post = ece(probs_test_post, y_test)
            nll_post = nll_from_logits(logits_test, y_test, T=T_star)
            brier_post = brier_multiclass(probs_test_post, y_test, n_classes)

            results.append({
                "dataset": dataset_name,
                "width": width,
                "width_idx": width_idx,
                "seed_idx": seed_idx,
                "seed": seed,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "acc": acc,
                "ece_pre": ece_pre,
                "nll_pre": nll_pre,
                "brier_pre": brier_pre,
                "T_star": T_star,
                "ece_post": ece_post,
                "nll_post": nll_post,
                "brier_post": brier_post,
                "converged": not conv_warn,
                "n_iter": int(clf.n_iter_),
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

    with open("round1_review_2_raw_results_rerun.json", "w") as f:
        json.dump({"results": results, "warn_log": warn_log,
                   "config": {"widths": WIDTHS, "n_seeds": N_SEEDS, "max_iter": MAX_ITER,
                              "n_bins": N_BINS, "base_seeds": BASE_SEEDS}}, f, indent=2)

    print(f"TOTAL runs: {len(results)}, non-converged: {len(warn_log)}", file=sys.stderr)
    print(f"Elapsed: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
