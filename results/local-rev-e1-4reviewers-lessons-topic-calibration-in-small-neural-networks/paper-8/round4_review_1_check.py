import json, numpy as np
from scipy import stats
v4 = json.load(open('results_v4.json'))
res_extra = v4['extra_seeds']['results']
T8e = [r['T_star'] for r in res_extra if r['width']==8]
T16e = [r['T_star'] for r in res_extra if r['width']==16]
print('extra T8', np.mean(T8e), np.std(T8e,ddof=1))
print('extra T16', np.mean(T16e), np.std(T16e,ddof=1))

v3 = json.load(open('results_v3.json'))
v2 = json.load(open('results_v2.json'))
T8_orig = [r['T_star'] for r in v3['dense']['results'] if r['width']==8]
T16_orig = [r['T_star'] for r in v2['main']['results'] if r['width']==16 and r['label_smoothing']==0.0]
T8_all = T8_orig+T8e
T16_all = T16_orig+T16e
t,p = stats.ttest_ind(T8_all, T16_all, equal_var=False)
print('pooled n=20 t=',t,'p=',p)

res_ls = v4['ls_dense']['results']
for w in [8,32]:
    T_LS = np.mean([r['T_star'] for r in res_ls if r['width']==w])
    ece_LS = np.mean([r['ece'] for r in res_ls if r['width']==w])
    print('LS width',w,'T*',T_LS,'ece',ece_LS)
