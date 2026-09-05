"""
Follow-up check (round-2 review): two independent reviewers asked whether T*
in the synthetic set at widths 16-256 is pinned at the coarse grid's upper
bound (20.0) rather than reflecting a true interior optimum. Re-run only the
synthetic dataset at widths >=16 with an extended grid (up to 500) and
compare T* and ece_post to the original [0.1, 20] grid.
"""
import json
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

WIDTHS = [16, 32, 64, 128, 256]
N_SEEDS = 20
MAX_ITER = 1500
BASE_SEED = 20_000


def get_seed(width_idx_full, seed_idx):
    return BASE_SEED + width_idx_full * 1000 + seed_idx


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


def ece(probs, y, n_bins=15):
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


def fit_temperature(logits_val, y_val, grid_max):
    grid = np.concatenate([np.linspace(0.1, 5.0, 50), np.linspace(5.0, grid_max, 60)])
    losses = [nll_from_logits(logits_val, y_val, T) for T in grid]
    best_T = grid[int(np.argmin(losses))]
    fine = np.linspace(max(0.05, best_T - 0.2), best_T + 0.2, 41)
    losses_fine = [nll_from_logits(logits_val, y_val, T) for T in fine]
    return float(fine[int(np.argmin(losses_fine))])


def main():
    t0 = time.time()
    Xs, ys = make_classification(
        n_samples=1800, n_features=20, n_informative=10, n_redundant=5,
        n_classes=4, n_clusters_per_class=2, class_sep=1.0, flip_y=0.03,
        random_state=42,
    )
    Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8)

    full_widths = [2, 4, 8, 16, 32, 64, 128, 256]
    out = []
    for width in WIDTHS:
        width_idx = full_widths.index(width)
        pinned_orig = 0
        t_orig, t_ext = [], []
        ece_orig, ece_ext = [], []
        for seed_idx in range(N_SEEDS):
            seed = get_seed(width_idx, seed_idx)
            X_train, X_rest, y_train, y_rest = train_test_split(
                Xs, ys, test_size=0.5, random_state=seed, stratify=ys)
            X_val, X_test, y_val, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
            clf = MLPClassifier(hidden_layer_sizes=(width,), activation="relu",
                                 solver="lbfgs", max_iter=MAX_ITER, random_state=seed,
                                 early_stopping=False)
            clf.fit(X_train, y_train)
            logits_val = forward_logits(clf, X_val)
            logits_test = forward_logits(clf, X_test)

            T_o = fit_temperature(logits_val, y_val, grid_max=20.0)
            T_e = fit_temperature(logits_val, y_val, grid_max=500.0)
            t_orig.append(T_o)
            t_ext.append(T_e)
            if T_o >= 19.9:
                pinned_orig += 1
            probs_test_o = softmax(logits_test / T_o)
            probs_test_e = softmax(logits_test / T_e)
            ece_orig.append(ece(probs_test_o, y_test))
            ece_ext.append(ece(probs_test_e, y_test))

        row = {
            "width": width,
            "pinned_at_20_of_20": pinned_orig,
            "mean_T_orig_grid20": float(np.mean(t_orig)),
            "mean_T_ext_grid500": float(np.mean(t_ext)),
            "max_T_ext": float(np.max(t_ext)),
            "mean_ece_post_orig": float(np.mean(ece_orig)),
            "mean_ece_post_ext": float(np.mean(ece_ext)),
        }
        out.append(row)
        print(row, flush=True)

    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
