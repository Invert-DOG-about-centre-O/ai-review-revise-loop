import json, numpy as np
v2 = json.load(open('results_v2.json'))
rob = v2['robustness']
print(rob.keys() if isinstance(rob, dict) else type(rob))
res = rob['results'] if isinstance(rob, dict) and 'results' in rob else rob
widths2 = [2,4,16,64,256]
for w in widths2:
    vals = [r['T_star'] for r in res if r['width']==w]
    accs = [r['acc'] for r in res if r['width']==w]
    print(w, 'T*mean', np.mean(vals), 'acc range', min(accs), max(accs))
