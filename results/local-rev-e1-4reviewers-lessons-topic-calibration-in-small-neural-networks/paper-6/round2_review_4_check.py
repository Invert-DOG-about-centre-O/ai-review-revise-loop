import json
from collections import Counter

d = json.load(open('raw_results.json'))
print('n results', len(d['results']))
print('warn_log len', len(d['warn_log']))
c = Counter((w['dataset'], w['width']) for w in d['warn_log'])
print(c)

hurts = {'digits': 0, 'synthetic': 0}
tot = {'digits': 0, 'synthetic': 0}
for r in d['results']:
    ds = r['dataset']
    tot[ds] += 1
    if r['ece_post'] > r['ece_pre']:
        hurts[ds] += 1
print('hurts', hurts, 'tot', tot)

w2 = [r for r in d['results'] if r['dataset'] == 'digits' and r['width'] == 2]
w2hurt = sum(1 for r in w2 if r['ece_post'] > r['ece_pre'])
print('digits w2 hurt', w2hurt, '/', len(w2))
