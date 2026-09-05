import json
import collections

d = json.load(open('multiseed_results.json'))['results']

by_seed = collections.defaultdict(dict)
for r in d:
    if r['smoothing'] == 0.0:
        by_seed[r['seed']][r['hidden']] = r['direction']

widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
firsts = []
for seed in range(10):
    dirs = by_seed[seed]
    first_over = None
    for w in widths:
        if dirs[w] == 'overconfident':
            first_over = w
            break
    firsts.append(first_over)
    print(seed, first_over, [dirs[w][:4] for w in widths])

print("firsts:", firsts)

print("\nTable 3 check (smoothing=0.1):")
for w in widths:
    pres = [r['test_ece_pre'] for r in d if r['smoothing']==0.1 and r['hidden']==w]
    posts = [r['test_ece_post'] for r in d if r['smoothing']==0.1 and r['hidden']==w]
    pres0 = [r['test_ece_pre'] for r in d if r['smoothing']==0.0 and r['hidden']==w]
    dirs = [r['direction'] for r in d if r['smoothing']==0.1 and r['hidden']==w]
    n_under = sum(1 for x in dirs if x=='underconfident')
    mean_pre = sum(pres)/len(pres)
    mean_post = sum(posts)/len(posts)
    mean_pre0 = sum(pres0)/len(pres0)
    ratio = mean_pre/mean_pre0
    print(w, n_under, round(mean_pre,4), round(ratio,2), round(mean_post,4))

print("\nhidden=2 smoothing=0.1 fitted T per seed:")
for r in d:
    if r['smoothing']==0.1 and r['hidden']==2:
        print(r['seed'], r['fitted_T'], r['test_ece_pre'], r['test_ece_post'])
