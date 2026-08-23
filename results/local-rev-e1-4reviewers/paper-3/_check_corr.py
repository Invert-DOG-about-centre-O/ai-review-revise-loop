import json, numpy as np
d = json.load(open('multiseed15_results.json'))
r = d['results']
acc = np.array([x['accuracy'] for x in r])
ece_gap = np.array([x['ece_logp']-x['ece_self_cons'] for x in r])
print('corr acc vs ece_gap:', np.corrcoef(acc, ece_gap)[0,1])
idx = np.argsort(-acc)
for i in idx[:5]:
    print(r[i]['seed'], acc[i], r[i]['ece_logp'], r[i]['ece_self_cons'])
print('logp wins sem:', sum(1 for x in r if x['auroc_logp']>x['auroc_sem_ent']))
print('logp wins sc:', sum(1 for x in r if x['auroc_logp']>x['auroc_self_cons']))
sc_better = sum(1 for x in r if x['ece_self_cons']<x['ece_logp'])
print('sc better calibrated count:', sc_better, 'of', len(r))
