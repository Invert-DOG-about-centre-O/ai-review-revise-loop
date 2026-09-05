"""
Round-3-review-requested follow-ups (all four reviewers independently asked
for progress on the residual-gap mechanism; two asked for the blobs direct
test; two asked for a multi-draw check on the isolated-width-8 control):

(1) Direct isolated-width8-vs-K=4-ensemble paired test on `blobs` (mirrors
    the digits test already in follow_up.py; reviewer 2 asked whether the
    digits residual-cost finding replicates on the second dataset).
(2) A candidate mechanism probe: does the per-seed residual gap (isolated
    width-8 ECE minus K=4-ensemble ECE, both calibrated) correlate with
    member disagreement/diversity, as a Jensen's-inequality-style averaging
    effect would predict (reviewers 1 and 4's suggested hypothesis)?
(3) Multi-draw robustness check for the isolated-width-8 control: average
    over 3 independent width-8 initializations per seed instead of reusing
    ensemble member 0, and recompute the underfitting-attribution fraction
    on digits (reviewers 1 and 2's concrete ask).

Reuses experiment.py's pipeline code; same seeds/splits as the main run.
"""
import json
import time
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split

from experiment import (
    train_mlp, get_logits, softmax, ece, nll,
    fit_temperature, ensemble_probs, count_params, N_SEEDS, BASE_WIDTH, T_GRID,
)


def prep(X, y, seed):
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
    return (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd, y_train, y_val, y_test


def fit_ensemble_T(ens_models, X_val_n, y_val):
    best_T, best_nll = 1.0, np.inf
    for T in T_GRID:
        n = nll(ensemble_probs(ens_models, X_val_n, T), y_val)
        if n < best_nll:
            best_nll, best_T = n, T
    return best_T


def isolated_vs_ensemble_direct(X, y, n_classes, seed_offset, K=4):
    in_dim = X.shape[1]
    width = BASE_WIDTH // K
    diffs = []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train_n, X_val_n, X_test_n, y_train, y_val, y_test = prep(X, y, seed)

        single8 = train_mlp(X_train_n, y_train, in_dim, 8, n_classes, seed * 1000 + 100)
        T8, _ = fit_temperature(get_logits(single8, X_val_n), y_val)
        p8 = softmax(get_logits(single8, X_test_n), T8)

        ens_models = [train_mlp(X_train_n, y_train, in_dim, width, n_classes, seed * 1000 + 100 + k)
                      for k in range(K)]
        best_T_e = fit_ensemble_T(ens_models, X_val_n, y_val)
        p_ens_cal = ensemble_probs(ens_models, X_test_n, best_T_e)

        diffs.append(ece(p8, y_test) - ece(p_ens_cal, y_test))
    diffs = np.array(diffs)
    wstat, wp = wilcoxon(diffs)
    return dict(n=N_SEEDS, mean_diff=float(diffs.mean()), wilcoxon_p=float(wp))


def diversity_mechanism_probe(X, y, n_classes, seed_offset, K=4):
    """Per-seed: residual gap (isolated8 ece_cal - ensemble ece_cal) vs. two
    diversity measures on the test set -- mean pairwise L1 distance between
    member softmax vectors, and mean per-example variance of the top-class
    probability across members -- both computed post-calibration (shared T)."""
    in_dim = X.shape[1]
    width = BASE_WIDTH // K
    gaps, pairwise_l1, prob_variance = [], [], []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train_n, X_val_n, X_test_n, y_train, y_val, y_test = prep(X, y, seed)

        single8 = train_mlp(X_train_n, y_train, in_dim, 8, n_classes, seed * 1000 + 100)
        T8, _ = fit_temperature(get_logits(single8, X_val_n), y_val)
        p8 = softmax(get_logits(single8, X_test_n), T8)

        ens_models = [train_mlp(X_train_n, y_train, in_dim, width, n_classes, seed * 1000 + 100 + k)
                      for k in range(K)]
        best_T_e = fit_ensemble_T(ens_models, X_val_n, y_val)
        member_probs = [softmax(get_logits(m, X_test_n), best_T_e) for m in ens_models]
        p_ens_cal = np.mean(member_probs, axis=0)

        gaps.append(ece(p8, y_test) - ece(p_ens_cal, y_test))

        # mean pairwise L1 distance between member prob vectors, averaged over test points
        pw = []
        for a in range(K):
            for b in range(a + 1, K):
                pw.append(np.mean(np.sum(np.abs(member_probs[a] - member_probs[b]), axis=1)))
        pairwise_l1.append(float(np.mean(pw)))

        stacked = np.stack(member_probs, axis=0)  # (K, n, classes)
        prob_variance.append(float(np.mean(np.var(stacked, axis=0))))

    gaps = np.array(gaps)
    pairwise_l1 = np.array(pairwise_l1)
    prob_variance = np.array(prob_variance)
    r_l1, p_l1 = spearmanr(gaps, pairwise_l1)
    r_var, p_var = spearmanr(gaps, prob_variance)
    return dict(
        n=N_SEEDS,
        mean_gap=float(gaps.mean()),
        spearman_r_pairwise_l1=float(r_l1), spearman_p_pairwise_l1=float(p_l1),
        spearman_r_prob_variance=float(r_var), spearman_p_prob_variance=float(p_var),
        gaps=gaps.tolist(), pairwise_l1=pairwise_l1.tolist(), prob_variance=prob_variance.tolist(),
    )


def multidraw_isolated_width8(X, y, n_classes, seed_offset, n_draws=3):
    """Average width-8 ECE over n_draws independent initializations per seed
    (instead of reusing ensemble member 0's seed), to check how noisy the
    single-draw underfitting-attribution estimate is."""
    in_dim = X.shape[1]
    diffs_single_draw = []
    diffs_multi_draw = []
    for i in range(N_SEEDS):
        seed = seed_offset + i
        X_train_n, X_val_n, X_test_n, y_train, y_val, y_test = prep(X, y, seed)

        single32 = train_mlp(X_train_n, y_train, in_dim, BASE_WIDTH, n_classes, seed * 1000 + 1)
        T32, _ = fit_temperature(get_logits(single32, X_val_n), y_val)
        p32 = softmax(get_logits(single32, X_test_n), T32)
        ece32 = ece(p32, y_test)

        draw_eces = []
        for d in range(n_draws):
            draw_seed = seed * 1000 + 100 + d * 37  # independent from ensemble member seeds
            single8 = train_mlp(X_train_n, y_train, in_dim, 8, n_classes, draw_seed)
            T8, _ = fit_temperature(get_logits(single8, X_val_n), y_val)
            p8 = softmax(get_logits(single8, X_test_n), T8)
            draw_eces.append(ece(p8, y_test))

        diffs_single_draw.append(ece32 - draw_eces[0])
        diffs_multi_draw.append(ece32 - float(np.mean(draw_eces)))

    diffs_single_draw = np.array(diffs_single_draw)
    diffs_multi_draw = np.array(diffs_multi_draw)
    w1, p1 = wilcoxon(diffs_single_draw)
    w2, p2 = wilcoxon(diffs_multi_draw)
    return dict(
        n=N_SEEDS, n_draws=n_draws,
        mean_diff_single_draw=float(diffs_single_draw.mean()), wilcoxon_p_single_draw=float(p1),
        mean_diff_multi_draw=float(diffs_multi_draw.mean()), wilcoxon_p_multi_draw=float(p2),
    )


def main():
    t0 = time.time()
    out = {}

    digits = load_digits()
    Xd, yd = digits.data.astype(np.float64), digits.target.astype(np.int64)
    blobs_X, blobs_y = make_blobs(n_samples=1800, centers=4, n_features=20, cluster_std=6.0, random_state=42)
    blobs_y = blobs_y.astype(np.int64)

    r = isolated_vs_ensemble_direct(blobs_X, blobs_y, 4, seed_offset=2000, K=4)
    print(f"[blobs] isolated-width8 vs K=4 ensemble (direct paired): mean_diff={r['mean_diff']:.4f} p={r['wilcoxon_p']:.4g}")
    out["blobs_isolated_vs_ensemble_direct"] = r

    r = diversity_mechanism_probe(Xd, yd, 10, seed_offset=1000, K=4)
    print(f"[digits] residual-gap vs member-diversity: r(L1)={r['spearman_r_pairwise_l1']:.3f} "
          f"p={r['spearman_p_pairwise_l1']:.4g}; r(var)={r['spearman_r_prob_variance']:.3f} "
          f"p={r['spearman_p_prob_variance']:.4g}")
    out["digits_diversity_mechanism_probe"] = r

    r = multidraw_isolated_width8(Xd, yd, 10, seed_offset=1000, n_draws=3)
    print(f"[digits] multi-draw isolated-width8: single-draw mean_diff={r['mean_diff_single_draw']:.4f} "
          f"p={r['wilcoxon_p_single_draw']:.4g}; 3-draw-avg mean_diff={r['mean_diff_multi_draw']:.4f} "
          f"p={r['wilcoxon_p_multi_draw']:.4g}")
    out["digits_multidraw_isolated_width8"] = r

    out["wall_time_seconds"] = time.time() - t0
    with open("mechanism_probe_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
