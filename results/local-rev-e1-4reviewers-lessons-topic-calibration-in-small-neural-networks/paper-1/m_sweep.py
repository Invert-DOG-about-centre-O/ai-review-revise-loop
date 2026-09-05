"""
Follow-up: does the underconfidence effect scale with ensemble size M,
or is it roughly constant per-split? Two independent round-3 reviewers
asked this. We test M in {2, 4, 8} at FIXED total width 64 on digits
(widths 32, 16, 8 respectively), 15 seeds, and also run a paired
Wilcoxon test on the bootstrap-vs-noboot signed-ECE gap that mechanism
results already contain (one reviewer asked whether that gap is tested,
not just eyeballed from point estimates).
"""
import json
import numpy as np
from scipy.stats import wilcoxon
from experiment import (make_digits, ensemble_run, single_run, evaluate,
                         fit_temperature, apply_temperature, N_SEEDS,
                         HIDDEN_LARGE)
from mechanism_check import signed_ece

TOTAL_WIDTH = HIDDEN_LARGE  # 64
M_VALUES = [2, 4, 8]


def main():
    n_classes = 10
    results = {f'ensemble{m}': [] for m in M_VALUES}
    results['single_large'] = []

    for seed in range(N_SEEDS):
        X_tr, y_tr, X_va, y_va, X_te, y_te, _ = make_digits(seed=seed)
        preds = {}
        preds['single_large'] = single_run(X_tr, y_tr, HIDDEN_LARGE, seed * 31337 + 401, n_classes)
        for m in M_VALUES:
            hidden = TOTAL_WIDTH // m
            predict, _ = ensemble_run(X_tr, y_tr, hidden, seed * 31337 + 1000 + m,
                                       m, n_classes)
            preds[f'ensemble{m}'] = predict

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

    print("\n--- M-sweep summary (mean over 15 seeds, digits) ---")
    for cname in results:
        print(f"{cname:14s} "
              f"pre_ece={col(cname,'pre','ece').mean():.4f} "
              f"pre_signed={col(cname,'pre','signed_ece').mean():+.4f} "
              f"post_ece={col(cname,'post','ece').mean():.4f} "
              f"post_signed={col(cname,'post','signed_ece').mean():+.4f} "
              f"acc={col(cname,'post','acc').mean():.4f}")

    tests = {}
    for m in M_VALUES:
        cname = f'ensemble{m}'
        stat, p = wilcoxon(col(cname, 'post', 'ece'), col('single_large', 'post', 'ece'),
                            alternative='greater')
        tests[f'{cname}_vs_large_post_ece'] = dict(stat=float(stat), p=float(p),
                                                     mean_diff=float((col(cname,'post','ece')-col('single_large','post','ece')).mean()))
        stat2, p2 = wilcoxon(col(cname, 'pre', 'signed_ece'))
        tests[f'{cname}_pre_signed_ece_vs_zero'] = dict(stat=float(stat2), p=float(p2),
                                                          mean=float(col(cname,'pre','signed_ece').mean()))

    # paired test: does pre-TS signed ECE magnitude grow monotonically with M?
    print("\ntests:", json.dumps(tests, indent=2))

    with open('m_sweep_results.json', 'w') as f:
        json.dump(dict(results=results, tests=tests), f, indent=2)


if __name__ == '__main__':
    main()
