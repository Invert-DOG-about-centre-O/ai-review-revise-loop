import json
import numpy as np
from scipy import stats

data = json.load(open('followup_results.json'))

def partial_spearman(x, y, z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    def resid(a, b):
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope*b+intercept)
    rxr = resid(rx, rz)
    ryr = resid(ry, rz)
    r, p = stats.pearsonr(rxr, ryr)
    return r, p

for ds in ['moons', 'circles', 'breast_cancer']:
    rows = [r for r in data if r['dataset'] == ds]
    width = np.array([np.log2(r['width']) for r in rows])
    ece_pre = np.array([r['ece_test'] for r in rows])
    acc_train = np.array([r['acc_train'] for r in rows])
    r, p = partial_spearman(width, ece_pre, acc_train)
    print(ds, r, p, len(rows))
