import json, numpy as np
from scipy import stats
v3 = json.load(open('results_v3.json'))
dense = v3['dense']['results']
for w in [8,32]:
    vals = [r['T_star'] for r in dense if r['width']==w]
    print(w, 'T* mean', np.mean(vals), 'sd', np.std(vals, ddof=1))
w2l = [r['T_star'] for r in v3['w2_long']['results']]
print('w2 400ep', np.mean(w2l), np.std(w2l, ddof=1))

v2 = json.load(open('results_v2.json'))
main = v2['main']['results']
widths_main = [2,4,16,64,256]
T = {w: [r['T_star'] for r in main if r['width']==w and r['label_smoothing']==0.0] for w in widths_main}
T[8] = [r['T_star'] for r in dense if r['width']==8]
T[32] = [r['T_star'] for r in dense if r['width']==32]
for w in [2,4,8,16,32,64,256]:
    vals=T[w]
    print(w, 'mean',np.mean(vals),'sd',np.std(vals,ddof=1))

ordered=[2,4,8,16,32,64,256]
for a,b in zip(ordered[:-1], ordered[1:]):
    t,p = stats.ttest_ind(T[a], T[b], equal_var=False)
    print(a,b,t,p)

w2_80 = T[2]
t,p = stats.ttest_ind(w2_80, w2l, equal_var=False)
print('w2 80 vs 400', np.mean(w2_80), np.mean(w2l), t, p)

# label smoothing fold change
ls_res = [r for r in main if r['label_smoothing']==0.1]
noLS_res = [r for r in main if r['label_smoothing']==0.0]
for w in widths_main:
    ece_noLS = np.mean([r['ece'] for r in noLS_res if r['width']==w])
    ece_LS = np.mean([r['ece'] for r in ls_res if r['width']==w])
    Tls = np.mean([r['T_star'] for r in ls_res if r['width']==w])
    print('width', w, 'ece_noLS', ece_noLS, 'ece_LS', ece_LS, 'fold', ece_LS/ece_noLS, 'T_LS', Tls)

# second instance
if 'instance2' in v2:
    print(v2['instance2'].keys())
