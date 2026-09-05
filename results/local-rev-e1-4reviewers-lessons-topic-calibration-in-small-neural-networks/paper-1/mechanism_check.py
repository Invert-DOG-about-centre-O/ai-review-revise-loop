"""
Follow-up: WHY is ensemble4 worse-calibrated than single_large on digits?
Two candidate mechanisms, tested directly (not asserted):
  (a) bootstrap resampling shrinks each member's effective training data
      -> weaker individual members -> averaging doesn't fully compensate
  (b) averaging itself (even with full, non-bootstrapped data per member,
      differing only in init) biases confidence

We isolate (a) vs (b) by adding a THIRD ensemble variant, ensemble4_nb,
where all 4 members see the FULL (non-bootstrapped) training set and
differ only in random init.  If ensemble4_nb closes most of the gap to
single_large, bootstrap resampling (a) is the driver; if it does not,
averaging itself (b) is implicated.

We also report SIGNED calibration error (mean confidence - mean accuracy)
to determine the direction of miscalibration (over- vs under-confident),
computed directly from predictions rather than assumed.
"""
import json
import numpy as np
from experiment import (make_digits, train_mlp, proba_full, single_run,
                         ensemble_run, evaluate, fit_temperature,
                         apply_temperature, N_SEEDS, HIDDEN_SMALL,
                         HIDDEN_LARGE, N_ENSEMBLE)
from scipy.stats import wilcoxon


def ensemble_nobootstrap(X_tr, y_tr, hidden, seed, n_members, n_classes):
    members = []
    for m in range(n_members):
        member_seed = seed * 1009 + m * 131 + 3
        clf = train_mlp(X_tr, y_tr, hidden, member_seed, n_classes)
        members.append(clf)

    def predict(X):
        probs = [proba_full(c, X, n_classes) for c in members]
        return np.mean(probs, axis=0)
    return predict


def signed_ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        e += (mask.sum() / total) * (conf[mask].mean() - correct[mask].mean())
    return e  # positive => overconfident, negative => underconfident


def main():
    n_classes = 10
    results = {'single_large': [], 'ensemble4_boot': [], 'ensemble4_noboot': [],
               'single_small': []}
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_va, y_va, X_te, y_te, _ = make_digits(seed=seed)

        preds = {}
        preds['single_large'] = single_run(X_tr, y_tr, HIDDEN_LARGE, seed * 31337, n_classes)
        preds['single_small'] = single_run(X_tr, y_tr, HIDDEN_SMALL, seed * 31337 + 999, n_classes)
        preds['ensemble4_boot'], _ = ensemble_run(X_tr, y_tr, HIDDEN_SMALL, seed * 31337 + 1,
                                                    N_ENSEMBLE, n_classes)
        preds['ensemble4_noboot'] = ensemble_nobootstrap(X_tr, y_tr, HIDDEN_SMALL, seed * 31337 + 2,
                                                           N_ENSEMBLE, n_classes)

        for cname, predict in preds.items():
            p_val = predict(X_va)
            p_test = predict(X_te)
            pre = evaluate(p_test, y_te, n_classes)
            pre['signed_ece'] = float(signed_ece(p_test, y_te))
            T = fit_temperature(p_val, y_va)
            p_test_ts = apply_temperature(p_test, T)
            post = evaluate(p_test_ts, y_te, n_classes)
            post['signed_ece'] = float(signed_ece(p_test_ts, y_te))
            results[cname].append(dict(seed=seed, T=float(T), pre=pre, post=post))
        print(f"seed {seed} done")

    def col(cname, phase, key):
        return np.array([r[phase][key] for r in results[cname]])

    print("\n--- Summary (mean over 15 seeds) ---")
    for cname in results:
        print(f"{cname:18s} "
              f"pre_ece={col(cname,'pre','ece').mean():.4f} "
              f"pre_signed={col(cname,'pre','signed_ece').mean():+.4f} "
              f"post_ece={col(cname,'post','ece').mean():.4f} "
              f"post_signed={col(cname,'post','signed_ece').mean():+.4f} "
              f"acc={col(cname,'post','acc').mean():.4f}")

    tests = {}
    tests['noboot_vs_boot_post_ece'] = dict(zip(
        ['stat', 'p'], [float(x) for x in wilcoxon(
            col('ensemble4_boot', 'post', 'ece'),
            col('ensemble4_noboot', 'post', 'ece'))]))
    tests['noboot_vs_large_post_ece'] = dict(zip(
        ['stat', 'p'], [float(x) for x in wilcoxon(
            col('ensemble4_noboot', 'post', 'ece'),
            col('single_large', 'post', 'ece'))]))
    print("\ntests:", json.dumps(tests, indent=2))

    with open('mechanism_results.json', 'w') as f:
        json.dump(dict(results=results, tests=tests), f, indent=2)


if __name__ == '__main__':
    main()
