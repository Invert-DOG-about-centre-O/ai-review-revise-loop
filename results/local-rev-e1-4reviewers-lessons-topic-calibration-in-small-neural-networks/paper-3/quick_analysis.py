import json, numpy as np
from scipy.stats import spearmanr, wilcoxon

with open('results_raw.json') as f:
    results = json.load(f)

for dataset in ['digits', 'synthetic']:
    rows = [r for r in results if r['dataset'] == dataset]
    widths = np.array([r['width'] for r in rows], dtype=float)
    T = np.array([r['T_star'] for r in rows])
    pre = np.array([r['pre_ece'] for r in rows])
    post = np.array([r['post_ece'] for r in rows])
    delta = pre - post

    mask = widths > 4
    rho, p = spearmanr(np.log2(widths[mask]), T[mask], alternative='greater')
    print(dataset, 'excl width4: rho=%.4f p=%.4g n=%d' % (rho, p, mask.sum()))

    terc = np.quantile(pre, [1/3, 2/3])
    low = pre <= terc[0]
    mid = (pre > terc[0]) & (pre <= terc[1])
    high = pre > terc[1]
    for name, m in [('low', low), ('mid', mid), ('high', high)]:
        if m.sum() > 0:
            try:
                w, pw = wilcoxon(pre[m], post[m], alternative='greater')
            except ValueError:
                w, pw = float('nan'), float('nan')
            print('  %s pre-ECE %s tercile: n=%d mean_pre=%.4f mean_post=%.4f mean_delta=%.4f wilcoxon_p=%.4g' % (
                dataset, name, m.sum(), pre[m].mean(), post[m].mean(), delta[m].mean(), pw))

    rho2, p2 = spearmanr(pre, delta)
    print('  %s spearman(pre_ece, delta) = %.4f, p=%.4g' % (dataset, rho2, p2))
    rho3, p3 = spearmanr(widths, delta)
    print('  %s spearman(width, delta) = %.4f, p=%.4g' % (dataset, rho3, p3))
