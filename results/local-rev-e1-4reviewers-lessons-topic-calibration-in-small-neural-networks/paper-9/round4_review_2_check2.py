import json, numpy as np
from scipy.stats import ttest_ind

d = json.load(open("results_raw.json"))
widths = sorted(set(r['width'] for r in d))
print("=== digits main table ===")
for w in widths:
    accs = [r['test_acc'] for r in d if r['width']==w]
    eces = [r['test_ece'] for r in d if r['width']==w]
    print(w, round(np.mean(accs),3), round(np.std(accs),3), round(np.mean(eces),3), round(np.std(eces),3))

ece48 = [r['test_ece'] for r in d if r['width'] in (4,8)]
ece_plat = [r['test_ece'] for r in d if r['width'] in (16,32,64,128,256,512)]
t,p = ttest_ind(ece48, ece_plat, equal_var=False)
print("4+8 vs 16-512 plateau:", t, p, "means", np.mean(ece48), np.mean(ece_plat))

acc128 = [r['test_acc'] for r in d if r['width']==128]
acc512 = [r['test_acc'] for r in d if r['width']==512]
t2,p2 = ttest_ind(acc128, acc512, equal_var=False)
print("128 vs 512 acc:", t2, p2)

fb = json.load(open("followup_budget.json"))
print("=== followup_budget keys ===", type(fb))
if isinstance(fb, list):
    print(fb[:2])
else:
    print(list(fb.keys())[:10])
