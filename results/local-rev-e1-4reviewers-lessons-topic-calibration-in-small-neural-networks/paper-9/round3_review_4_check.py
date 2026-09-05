import json, numpy as np
from scipy import stats
d = json.load(open('results_raw.json'))
def ece_arr(ws):
    return np.array([r['test_ece'] for r in d if r['width'] in ws])
peak = ece_arr([4,8])
plateau = ece_arr([16,32,64,128,256,512])
t,p = stats.ttest_ind(peak, plateau, equal_var=False)
print('4+8 vs plateau ECE t,p:', t, p, 'means', peak.mean(), plateau.mean())

def acc_arr(w):
    return np.array([r['test_acc'] for r in d if r['width']==w])
t2,p2 = stats.ttest_ind(acc_arr(128), acc_arr(512), equal_var=False)
print('128 vs 512 acc t,p:', t2, p2)

ws = np.array([r['width'] for r in d])
eces = np.array([r['test_ece'] for r in d])
nlls = np.array([r['test_nll'] for r in d])
rho1,pv1 = stats.spearmanr(np.log2(ws), eces)
rho2,pv2 = stats.spearmanr(np.log2(ws), nlls)
print('spearman ece', rho1, pv1)
print('spearman nll', rho2, pv2)
