import json
d = json.load(open('multiseed_results.json'))
r = d['results']

cross = {}
for seed in range(10):
    rows = [x for x in r if x['seed']==seed and x['smoothing']==0.0]
    rows.sort(key=lambda x: x['hidden'])
    first_over = None
    for x in rows:
        if x['direction']=='overconfident':
            first_over = x['hidden']
            break
    cross[seed] = first_over
print(cross)

for h in [2,4,8,16]:
    rows = [x for x in r if x['hidden']==h and x['smoothing']==0.1]
    n_under = sum(1 for x in rows if x['direction']=='underconfident')
    mean_pre = sum(x['test_ece_pre'] for x in rows)/len(rows)
    mean_post = sum(x['test_ece_post'] for x in rows)/len(rows)
    print(h, len(rows), n_under, mean_pre, mean_post)
