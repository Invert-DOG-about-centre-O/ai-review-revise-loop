import json, numpy as np
d = json.load(open('multiseed15_results.json'))
acc = np.array([r['accuracy'] for r in d['results']])
ece_gap = np.array([r['ece_logp']-r['ece_self_cons'] for r in d['results']])
print('corr acc vs ece_gap', np.corrcoef(acc, ece_gap)[0,1])
logp = np.array([r['auroc_logp'] for r in d['results']])
sem = np.array([r['auroc_sem_ent'] for r in d['results']])
sc = np.array([r['auroc_self_cons'] for r in d['results']])
from scipy import stats
print('wins sem', (logp>sem).sum(), 'wins sc', (logp>sc).sum())
print('sign test p (14/15, one-sided)', stats.binomtest(14,15,0.5,alternative='greater').pvalue)

# check the 3 highest accuracy seeds
idx = np.argsort(acc)[::-1][:3]
for i in idx:
    r = d['results'][i]
    print(r['seed'], r['accuracy'], r['ece_logp'], r['ece_self_cons'])
