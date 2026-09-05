"""
Effect-size and power analysis for the post-TS ECE null result.

All four round-2 reviewers independently asked the same question: with only
15 seeds, is "not significant" evidence of "no gap", or just an underpowered
test? This computes Cohen's d (paired), a bootstrap 95% CI on the mean
paired post-TS ECE difference (ensemble4 - single_large), and the minimum
detectable effect (via a paired-t power calc) at n=15 and at the n=50
follow-up sample size the paper's Future Work proposes.
"""
import json
import numpy as np
from scipy import stats

rng = np.random.RandomState(0)


def paired_diff(results, key_a, key_b, metric='ece', phase='post'):
    a = np.array([r[phase][metric] for r in results[key_a]])
    b = np.array([r[phase][metric] for r in results[key_b]])
    return a - b


def cohens_d_paired(diff):
    return diff.mean() / diff.std(ddof=1)


def bootstrap_ci(diff, n_boot=100000, alpha=0.05):
    n = len(diff)
    idx = rng.randint(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def min_detectable_d(n, alpha=0.05, power=0.8):
    """Minimum detectable Cohen's d for a one-sample/paired t-test, one-sided."""
    from scipy.stats import nct
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df)
    # search d such that power = P(T > t_crit | ncp = d*sqrt(n))
    lo, hi = 0.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ncp = mid * np.sqrt(n)
        p = 1 - nct.cdf(t_crit, df, ncp)
        if p < power:
            lo = mid
        else:
            hi = mid
    return hi


def report(name, results):
    diff = paired_diff(results, 'ensemble4', 'single_large')
    d = cohens_d_paired(diff)
    lo, hi = bootstrap_ci(diff)
    mde15 = min_detectable_d(15)
    mde50 = min_detectable_d(50)
    print(f"--- {name} (post-TS ECE, ensemble4 - single_large, n={len(diff)}) ---")
    print(f"  mean diff = {diff.mean():.5f}, sd = {diff.std(ddof=1):.5f}")
    print(f"  Cohen's d (paired) = {d:.3f}")
    print(f"  95% bootstrap CI on mean diff = [{lo:.5f}, {hi:.5f}]")
    print(f"  minimum detectable d at n=15, power=0.8, alpha=0.05 (one-sided) = {mde15:.3f}")
    print(f"  minimum detectable d at n=50, power=0.8, alpha=0.05 (one-sided) = {mde50:.3f}")
    return dict(mean_diff=float(diff.mean()), sd=float(diff.std(ddof=1)), cohens_d=float(d),
                ci95=[lo, hi], mde_n15=float(mde15), mde_n50=float(mde50))


def main():
    data = json.load(open('results.json'))
    out = {}
    for name in ['digits', 'synthetic']:
        out[name] = report(name, data['results'][name])
    with open('power_analysis.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
