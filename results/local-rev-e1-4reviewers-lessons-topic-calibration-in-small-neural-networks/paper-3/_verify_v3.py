import json, numpy as np
from scipy.stats import mannwhitneyu, spearmanr

with open('results_mechanism_v3_decouple.json') as f:
    rows = json.load(f)
pre = np.array([r['pre_ece'] for r in rows])
fw = np.array([r['frac_worse'] for r in rows])
median = np.median(pre)
low = fw[pre <= median]
high = fw[pre > median]
print('n low/high', len(low), len(high), 'median', median)
u, p = mannwhitneyu(low, high, alternative='greater')
print('MannWhitney low>high p=', p)
rho, p2 = spearmanr(pre, fw)
print('spearman(pre,frac_worse) rho=%.4f p=%.4g' % (rho, p2))
