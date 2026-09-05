import json, numpy as np

d = json.load(open("results_synth2class.json"))
widths = sorted(set(r['width'] for r in d))
print("=== synth2class table ===")
for w in widths:
    accs = [r['test_acc'] for r in d if r['width']==w]
    eces = [r['test_ece'] for r in d if r['width']==w]
    print(w, round(np.mean(accs),3), round(np.std(accs),3), round(np.mean(eces),3), round(np.std(eces),3))

d2 = json.load(open("results_raw.json"))
w2 = sorted([r['test_acc'] for r in d2 if r['width']==2])
print("=== width2 digits sorted accs ===")
print(w2)
