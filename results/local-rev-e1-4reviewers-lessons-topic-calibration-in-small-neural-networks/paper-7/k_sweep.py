"""
Follow-up (reviewer-requested) robustness check: does the single-beats-ensemble
result on digits depend on the specific K=4 ensemble granularity, or hold across
K=2 (mildly narrower members) and K=8 (more severely narrower members) at the
same total parameter budget? Also isolates underfitting: does a single
UN-ensembled width-8 network (no averaging) alone explain the gap?

Reuses experiment.py's data pipeline/model/calibration code (digits only, same
20 seeds, same splits) to keep this a controlled, apples-to-apples add-on.
"""
import json
import time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from experiment import (
    MLP, train_mlp, get_logits, softmax, ece, nll, brier,
    fit_temperature, ensemble_probs, count_params, N_SEEDS, BASE_WIDTH, N_BINS,
)

def run_k(X, y, n_classes, seed_offset, K):
    in_dim = X.shape[1]
    width = BASE_WIDTH // K
    diffs = []
    single_widths_ece = []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
        X_train_n = (X_train - mu) / sd
        X_val_n = (X_val - mu) / sd
        X_test_n = (X_test - mu) / sd

        single_seed = seed * 1000 + 1
        single = train_mlp(X_train_n, y_train, in_dim, BASE_WIDTH, n_classes, single_seed)
        single_test_logits = get_logits(single, X_test_n)
        single_val_logits = get_logits(single, X_val_n)
        T_single, _ = fit_temperature(single_val_logits, y_val)
        p_single_cal = softmax(single_test_logits, T_single)

        ens_models = []
        for k in range(K):
            sub_seed = seed * 1000 + 100 + k
            m = train_mlp(X_train_n, y_train, in_dim, width, n_classes, sub_seed)
            ens_models.append(m)

        def ens_val_nll(T):
            p = ensemble_probs(ens_models, X_val_n, T)
            return nll(p, y_val)

        from experiment import T_GRID
        best_T_e, best_nll_e = 1.0, np.inf
        for T in T_GRID:
            n = ens_val_nll(T)
            if n < best_nll_e:
                best_nll_e, best_T_e = n, T

        p_ens_cal = ensemble_probs(ens_models, X_test_n, best_T_e)
        diffs.append(ece(p_single_cal, y_test) - ece(p_ens_cal, y_test))
    diffs = np.array(diffs)
    wstat, wp = wilcoxon(diffs)
    ens_params = K * count_params(ens_models[0])
    single_params = count_params(single)
    return dict(K=K, width=width, n=N_SEEDS, mean_diff=float(diffs.mean()),
                wilcoxon_p=float(wp), single_params=single_params, ens_params=ens_params)


def run_single_width8(X, y, n_classes, seed_offset):
    """Isolate underfitting: a single (non-ensembled) width-8 network alone,
    compared to the width-32 single network, at calibrated ECE."""
    in_dim = X.shape[1]
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
        X_train_n = (X_train - mu) / sd
        X_val_n = (X_val - mu) / sd
        X_test_n = (X_test - mu) / sd

        single_seed = seed * 1000 + 1
        single32 = train_mlp(X_train_n, y_train, in_dim, BASE_WIDTH, n_classes, single_seed)
        T32, _ = fit_temperature(get_logits(single32, X_val_n), y_val)
        p32 = softmax(get_logits(single32, X_test_n), T32)

        w8_seed = seed * 1000 + 100  # same seed as ensemble member 0, for comparability
        single8 = train_mlp(X_train_n, y_train, in_dim, 8, n_classes, w8_seed)
        T8, _ = fit_temperature(get_logits(single8, X_val_n), y_val)
        p8 = softmax(get_logits(single8, X_test_n), T8)

        diffs.append(ece(p32, y_test) - ece(p8, y_test))
    diffs = np.array(diffs)
    wstat, wp = wilcoxon(diffs)
    return dict(n=N_SEEDS, mean_diff=float(diffs.mean()), wilcoxon_p=float(wp))


def main():
    t0 = time.time()
    digits = load_digits()
    Xd, yd = digits.data.astype(np.float64), digits.target.astype(np.int64)

    out = {}
    for K in [2, 8]:
        r = run_k(Xd, yd, 10, seed_offset=1000, K=K)
        print(f"K={K}: width={r['width']} mean_diff(single-ens)={r['mean_diff']:.4f} "
              f"p={r['wilcoxon_p']:.5g} params single={r['single_params']} ens={r['ens_params']}")
        out[f"K{K}"] = r

    r8 = run_single_width8(Xd, yd, 10, seed_offset=1000)
    print(f"single-width32 vs single-width8 (no ensembling): mean_diff={r8['mean_diff']:.4f} "
          f"p={r8['wilcoxon_p']:.5g}")
    out["single_width8_isolated"] = r8

    out["wall_time_seconds"] = time.time() - t0
    with open("k_sweep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
