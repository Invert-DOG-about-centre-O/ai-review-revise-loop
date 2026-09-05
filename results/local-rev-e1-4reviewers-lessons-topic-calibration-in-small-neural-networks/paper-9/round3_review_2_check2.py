import json
d = json.load(open('results_raw.json'))
rows = [r for r in d if r['width']==2]
accs = sorted(r['test_acc'] for r in rows)
print(accs)
