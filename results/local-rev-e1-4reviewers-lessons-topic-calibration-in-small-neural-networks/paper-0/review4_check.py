import json
d = json.load(open("multiseed_results.json"))
res = d['results']
from collections import defaultdict
by_seed = defaultdict(dict)
for r in res:
    if not r['smoothing']:
        by_seed[r['seed']][r['hidden']] = r['direction']

widths = sorted(set(r['hidden'] for r in res))
print('widths', widths)
first_over = {}
for s in sorted(by_seed):
    row = by_seed[s]
    seq = [row.get(w) for w in widths]
    print(s, seq)
    for w in widths:
        if row.get(w) == 'overconfident':
            first_over[s] = w
            break
print('first overconfident per seed:', first_over)

h2 = [by_seed[s][2] for s in by_seed]
print('hidden=2 all underconfident?', all(x=='underconfident' for x in h2), h2)
for w in [32,64,128,256,512]:
    vals = [by_seed[s][w] for s in by_seed]
    print(w, 'all overconfident?', all(x=='overconfident' for x in vals))

# smoothing table 3 check
sm = defaultdict(list)
for r in res:
    if r['smoothing']:
        sm[r['hidden']].append(r)
for w in [2,4,8,16]:
    rows = sm[w]
    fracs_under = sum(1 for r in rows if r['direction']=='underconfident')
    mean_ece_pre = sum(r['test_ece_pre'] for r in rows)/len(rows)
    print(w, 'n=',len(rows), 'frac_under', fracs_under, 'mean_ece_pre', round(mean_ece_pre,4))

# unsmoothed for ratio
un = defaultdict(list)
for r in res:
    if not r['smoothing']:
        un[r['hidden']].append(r)
for w in [2,4,8,16]:
    rows = un[w]
    mean_ece_pre = sum(r['test_ece_pre'] for r in rows)/len(rows)
    print('unsmoothed', w, 'mean_ece_pre', round(mean_ece_pre,4))
