import json, statistics as st
d = json.load(open('multiseed_results.json'))['results']
rows = [r for r in d if r['smoothing']==0.1 and r['hidden']==2]
floor_count = sum(1 for r in rows if r['fitted_T'] <= 1e-3+1e-9)
print('floor count 6/10 claim:', floor_count, [round(r['fitted_T'],5) for r in rows])
widths=[2,4,8,16,32,64,128,256,512]
for w in widths:
    rr=[r for r in d if r['smoothing']==0.1 and r['hidden']==w]
    frac=sum(1 for r in rr if r['direction']=='underconfident')
    print(w, frac,'/10')
seed0=[r for r in d if r['seed']==0 and r['smoothing']==0.0]
for r in sorted(seed0,key=lambda r:r['hidden']):
    print(r['hidden'], round(r['test_acc'],3), round(r['test_conf_pre'],3), round(r['test_ece_pre'],4), round(r['fitted_T'],3), round(r['test_ece_post'],4), r['direction'])
