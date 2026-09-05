import json, numpy as np
from scipy import stats
d = json.load(open("results_raw.json"))
diffs = np.array([r['test_ece'] - r['test_ece_ts'] for r in d])
print('n=',len(diffs),'mean diff', diffs.mean(), 'sd', diffs.std(ddof=1))
t,p = stats.ttest_1samp(diffs, 0)
print('pooled paired t-test (all 90, pseudoreplicated):', t, p)

widths = sorted(set(r['width'] for r in d))
width_diffs = []
for w in widths:
    vals = [r['test_ece']-r['test_ece_ts'] for r in d if r['width']==w]
    width_diffs.append(np.mean(vals))
width_diffs = np.array(width_diffs)
t2,p2 = stats.ttest_1samp(width_diffs, 0)
print('width-level paired t-test (n=9 width means):', t2, p2)
print(list(zip(widths, width_diffs)))
