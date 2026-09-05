import json, numpy as np
from scipy.stats import ttest_ind, ttest_rel

r = json.load(open('results_raw.json'))

def ece(w):
    return np.array([x['test_ece'] for x in r if x['width'] == w])

pooled48 = np.concatenate([ece(4), ece(8)])
plateau = np.concatenate([ece(w) for w in [16, 32, 64, 128, 256, 512]])
t, p = ttest_ind(pooled48, plateau, equal_var=False)
print('4+8 vs plateau', t, p, pooled48.mean(), plateau.mean())

t2, p2 = ttest_ind(ece(4), ece(16), equal_var=False)
print('4 vs 16', t2, p2)

acc128 = np.array([x['test_acc'] for x in r if x['width'] == 128])
acc512 = np.array([x['test_acc'] for x in r if x['width'] == 512])
t3, p3 = ttest_ind(acc128, acc512, equal_var=False)
print('acc 128 vs 512', t3, p3)

w2 = [x for x in r if x['width'] == 2]
accs = sorted(x['test_acc'] for x in w2)
print('w2 accs sorted', accs)

collapsed = [x for x in w2 if x['test_acc'] < 0.15]
noncollapsed = [x for x in w2 if x['test_acc'] >= 0.15]
print('collapsed n', len(collapsed), 'mean acc', np.mean([x['test_acc'] for x in collapsed]), 'mean ece', np.mean([x['test_ece'] for x in collapsed]))
print('noncollapsed n', len(noncollapsed), 'mean acc', np.mean([x['test_acc'] for x in noncollapsed]), 'mean ece', np.mean([x['test_ece'] for x in noncollapsed]))

nc_ece = np.array([x['test_ece'] for x in noncollapsed])
t4, p4 = ttest_ind(nc_ece, ece(16), equal_var=False)
print('noncollapsed w2 vs w16', t4, p4)

# temperature scaling paired t-tests
widths = [2,4,8,16,32,64,128,256,512]
for w in widths:
    xs = [x for x in r if x['width']==w]
    pre = np.array([x['test_ece'] for x in xs])
    post = np.array([x['test_ece_ts'] for x in xs])
    t5,p5 = ttest_rel(pre, post)
    print('temp scaling width', w, 'pre', pre.mean(), 'post', post.mean(), 't', t5, 'p', p5)
