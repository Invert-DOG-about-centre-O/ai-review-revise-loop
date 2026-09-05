"""
Round-2-review-requested follow-ups:
(1) Extend the K=2/K=8 granularity sweep + isolated-width-8 control to `blobs`
    (all four round-2 reviewers asked whether the digits granularity pattern
    replicates on the second dataset; previously skipped for compute budget).
(2) A *direct* paired test of isolated-width-8 ECE vs. K=4-ensemble ECE on
    digits (reviewer 3's concrete ask: the ~43% "residual" gap was previously
    only a difference-of-differences against the width-32 baseline, never
    tested directly against each other).
Reuses experiment.py's pipeline code; same seeds/splits as the main run.
"""
import json
import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split

from experiment import (
    train_mlp, get_logits, softmax, ece, nll,
    fit_temperature, ensemble_probs, count_params, N_SEEDS, BASE_WIDTH, T_GRID,
)


def run_k(X, y, n_classes, seed_offset, K):
    in_dim = X.shape[1]
    width = BASE_WIDTH // K
    diffs = []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
        X_train_n, X_val_n, X_test_n = (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd

        single_seed = seed * 1000 + 1
        single = train_mlp(X_train_n, y_train, in_dim, BASE_WIDTH, n_classes, single_seed)
        T_single, _ = fit_temperature(get_logits(single, X_val_n), y_val)
        p_single_cal = softmax(get_logits(single, X_test_n), T_single)

        ens_models = [train_mlp(X_train_n, y_train, in_dim, width, n_classes, seed * 1000 + 100 + k)
                      for k in range(K)]

        best_T_e, best_nll_e = 1.0, np.inf
        for T in T_GRID:
            n = nll(ensemble_probs(ens_models, X_val_n, T), y_val)
            if n < best_nll_e:
                best_nll_e, best_T_e = n, T
        p_ens_cal = ensemble_probs(ens_models, X_test_n, best_T_e)
        diffs.append(ece(p_single_cal, y_test) - ece(p_ens_cal, y_test))
    diffs = np.array(diffs)
    wstat, wp = wilcoxon(diffs)
    return dict(K=K, width=width, n=N_SEEDS, mean_diff=float(diffs.mean()), wilcoxon_p=float(wp),
                single_params=count_params(single), ens_params=K * count_params(ens_models[0]))


def run_single_width8(X, y, n_classes, seed_offset):
    in_dim = X.shape[1]
    diffs = []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
        X_train_n, X_val_n, X_test_n = (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd

        single32 = train_mlp(X_train_n, y_train, in_dim, BASE_WIDTH, n_classes, seed * 1000 + 1)
        T32, _ = fit_temperature(get_logits(single32, X_val_n), y_val)
        p32 = softmax(get_logits(single32, X_test_n), T32)

        single8 = train_mlp(X_train_n, y_train, in_dim, 8, n_classes, seed * 1000 + 100)
        T8, _ = fit_temperature(get_logits(single8, X_val_n), y_val)
        p8 = softmax(get_logits(single8, X_test_n), T8)

        diffs.append(ece(p32, y_test) - ece(p8, y_test))
    diffs = np.array(diffs)
    wstat, wp = wilcoxon(diffs)
    return dict(n=N_SEEDS, mean_diff=float(diffs.mean()), wilcoxon_p=float(wp))


def run_isolated_vs_ensemble_direct(X, y, n_classes, seed_offset, K=4):
    """Direct paired test: isolated (non-ensembled) width-8 ECE vs. the K=4
    ensemble's ECE, seed-for-seed -- not each vs. the width-32 baseline."""
    in_dim = X.shape[1]
    width = BASE_WIDTH // K
    diffs = []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
        X_train_n, X_val_n, X_test_n = (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd

        single8 = train_mlp(X_train_n, y_train, in_dim, 8, n_classes, seed * 1000 + 100)
        T8, _ = fit_temperature(get_logits(single8, X_val_n), y_val)
        p8 = softmax(get_logits(single8, X_test_n), T8)

        ens_models = [train_mlp(X_train_n, y_train, in_dim, width, n_classes, seed * 1000 + 100 + k)
                      for k in range(K)]
        best_T_e, best_nll_e = 1.0, np.inf
        for T in T_GRID:
            n = nll(ensemble_probs(ens_models, X_val_n, T), y_val)
            if n < best_nll_e:
                best_nll_e, best_T_e = n, T
        p_ens_cal = ensemble_probs(ens_models, X_test_n, best_T_e)

        diffs.append(ece(p8, y_test) - ece(p_ens_cal, y_test))
    diffs = np.array(diffs)
    wstat, wp = wilcoxon(diffs)
    return dict(n=N_SEEDS, mean_diff=float(diffs.mean()), wilcoxon_p=float(wp))


def main():
    t0 = time.time()
    out = {}

    digits = load_digits()
    Xd, yd = digits.data.astype(np.float64), digits.target.astype(np.int64)
    r = run_isolated_vs_ensemble_direct(Xd, yd, 10, seed_offset=1000, K=4)
    print(f"[digits] isolated-width8 vs K=4 ensemble (direct paired): mean_diff={r['mean_diff']:.4f} p={r['wilcoxon_p']:.4g}")
    out["digits_isolated_vs_ensemble_direct"] = r

    Xb, yb = make_blobs(n_samples=1800, centers=4, n_features=20, cluster_std=6.0, random_state=42)
    yb = yb.astype(np.int64)
    for K in [2, 8]:
        r = run_k(Xb, yb, 4, seed_offset=2000, K=K)
        print(f"[blobs] K={K}: width={r['width']} mean_diff={r['mean_diff']:.4f} p={r['wilcoxon_p']:.4g}")
        out[f"blobs_K{K}"] = r
    r8b = run_single_width8(Xb, yb, 4, seed_offset=2000)
    print(f"[blobs] single-width32 vs single-width8 (isolated): mean_diff={r8b['mean_diff']:.4f} p={r8b['wilcoxon_p']:.4g}")
    out["blobs_single_width8_isolated"] = r8b

    out["wall_time_seconds"] = time.time() - t0
    with open("follow_up_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
