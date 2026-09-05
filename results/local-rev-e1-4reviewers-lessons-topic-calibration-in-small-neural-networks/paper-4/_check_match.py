import json
raw = json.load(open('raw_results.json'))
fu = json.load(open('followup_results.json'))
print(raw[0])
print(fu[0])
key = lambda r: (r['dataset'], r['width'], r['seed'])
raw_map = {key(r): r for r in raw}
fu_map = {key(r): r for r in fu}
maxdiff = 0.0
for k in raw_map:
    a = raw_map[k].get('ece_pre', raw_map[k].get('ece_test'))
    b = fu_map[k]['ece_test']
    maxdiff = max(maxdiff, abs(a-b))
print('max abs diff', maxdiff)
