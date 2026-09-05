import json, numpy as np
from scipy.stats import ttest_ind, spearmanr

r = json.load(open('results_raw.json'))
widths = [2,4,8,16,32,64,128,256,512]
logw = np.array([np.log2(x['width']) for x in r])
ece_all = np.array([x['test_ece'] for x in r])
nll_all = np.array([x['test_nll'] for x in r])
rho1, p1 = spearmanr(logw, ece_all)
rho2, p2 = spearmanr(logw, nll_all)
print('spearman ece', rho1, p1)
print('spearman nll', rho2, p2)

bc = json.load(open('results_bc.json'))
def ece_bc(w):
    return np.array([x['test_ece'] for x in bc if x['width']==w])
pooled48 = np.concatenate([ece_bc(4), ece_bc(8)])
plateau = np.concatenate([ece_bc(w) for w in [16,32,64,128]])
t,p = ttest_ind(pooled48, plateau, equal_var=False)
print('bc 4+8 vs plateau', t, p, pooled48.mean(), plateau.mean())

bc_widths = sorted(set(x['width'] for x in bc))
for w in bc_widths:
    xs = [x for x in bc if x['width']==w]
    acc = np.mean([x['test_acc'] for x in xs])
    ece = np.mean([x['test_ece'] for x in xs])
    nll = np.mean([x['test_nll'] for x in xs])
    print(w, len(xs), round(acc,3), round(ece,3), round(nll,3))

w2bc = [x for x in bc if x['width']==2]
print('bc w2 accs', sorted(x['test_acc'] for x in w2bc))
