import json, numpy as np
r = json.load(open('results.json'))
etas = [row['eta'] for row in r['mitigation_eta']]
Hs = [row['H_mean'] for row in r['mitigation_eta']]
Accs = [row['Acc_mean'] for row in r['mitigation_eta']]
target = 0.29552331349206395
eta_at_target = np.interp(target, Hs[::-1], etas[::-1])
acc_at_target = np.interp(eta_at_target, etas, Accs)
print('eta_at_target', eta_at_target)
print('acc_at_target', acc_at_target)
print('relative accuracy drop from eta=0 acc', Accs[0], ':', 1-acc_at_target/Accs[0])
