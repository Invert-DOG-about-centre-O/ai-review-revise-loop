import json
d = json.load(open('multiseed_results.json'))['results']
widths=[2,4,8,16,32,64,128,256,512]

crossover={}
for seed in range(10):
    rows = [r for r in d if r['seed']==seed and r['smoothing']==0.0]
    rows.sort(key=lambda r: widths.index(r['hidden']))
    first=None
    for r in rows:
        if r['direction']=='overconfident':
            first=r['hidden']; break
    crossover[seed]=first
print('crossover', crossover)

for w in widths[:4]:
    rows=[r for r in d if r['smoothing']==0.1 and r['hidden']==w]
    frac_under = sum(1 for r in rows if r['direction']=='underconfident')
    mean_ece_pre = sum(r['test_ece_pre'] for r in rows)/len(rows)
    mean_ece_post = sum(r['test_ece_post'] for r in rows)/len(rows)
    rows0=[r for r in d if r['smoothing']==0.0 and r['hidden']==w]
    mean_ece_pre0 = sum(r['test_ece_pre'] for r in rows0)/len(rows0)
    print(w, frac_under, len(rows), round(mean_ece_pre,4), round(mean_ece_pre/mean_ece_pre0,2), round(mean_ece_post,4))

rows=[r for r in d if r['smoothing']==0.1 and r['hidden']==2]
for r in rows:
    print(r['seed'], r['fitted_T'], r['test_ece_pre'], r['test_ece_post'])
