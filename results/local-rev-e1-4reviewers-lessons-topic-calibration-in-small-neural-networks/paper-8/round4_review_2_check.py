import json, numpy as np
from scipy import stats

v4 = json.load(open('results_v4.json'))
v3 = json.load(open('results_v3.json'))
v2 = json.load(open('results_v2.json'))

for w in [8,32]:
    res = [r for r in v4['ls_dense']['results'] if r['width']==w]
    T_LS = np.mean([r['T_star'] for r in res])
    ece_LS = np.mean([r['ece'] for r in res])
    noLS = [r for r in v3['dense']['results'] if r['width']==w]
    T_noLS = np.mean([r['T_star'] for r in noLS])
    ece_noLS = np.mean([r['ece'] for r in noLS])
    print(f'width {w}: T*(noLS)={T_noLS:.3f} T*(LS)={T_LS:.3f} ECE {ece_noLS:.4f}->{ece_LS:.4f} fold={ece_LS/ece_noLS:.1f}x')

w2 = v4['w2_train_check']
acc80 = np.array(w2['epochs80']['train_acc'])
acc400 = np.array(w2['epochs400']['train_acc'])
loss80 = np.array(w2['epochs80']['train_loss'])
loss400 = np.array(w2['epochs400']['train_loss'])
print('train acc80', acc80.mean(), acc80.std(), 'loss80', loss80.mean(), loss80.std())
print('train acc400', acc400.mean(), acc400.std(), 'loss400', loss400.mean(), loss400.std())

extra = v4['extra_seeds']['results']
T8_extra = [r['T_star'] for r in extra if r['width']==8]
T16_extra = [r['T_star'] for r in extra if r['width']==16]
print('T8 extra', np.mean(T8_extra), np.std(T8_extra))
print('T16 extra', np.mean(T16_extra), np.std(T16_extra))

T8_orig = [r['T_star'] for r in v3['dense']['results'] if r['width']==8]
T16_orig = [r['T_star'] for r in v2['main']['results'] if r['width']==16 and r['label_smoothing']==0.0]
T8_all = T8_orig+T8_extra
T16_all = T16_orig+T16_extra
t,p = stats.ttest_ind(T8_all, T16_all, equal_var=False)
print('n=20 test', np.mean(T8_all), np.mean(T16_all), t, p)

# width2 vs width4 etc adjacent tests from Table1 (main results.json?)
