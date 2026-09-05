import json, numpy as np
from scipy.stats import ttest_ind

d = json.load(open('followup_brier.json'))
for w in [2,4,8,16,512]:
    rows=[r for r in d if r['width']==w]
    b=np.array([r['brier'] for r in rows])
    e10=np.array([r['ece10'] for r in rows]); e15=np.array([r['ece15'] for r in rows]); e20=np.array([r['ece20'] for r in rows])
    acc=np.array([r['test_acc'] for r in rows])
    print(w, 'brier',b.mean(),'acc',acc.mean(),'ece10',e10.mean(),'ece15',e15.mean(),'ece20',e20.mean())
b48=np.array([r['brier'] for r in d if r['width'] in (4,8)])
bpl=np.array([r['brier'] for r in d if r['width'] in (16,512)])
t,p=ttest_ind(b48,bpl,equal_var=False)
print('brier t,p', t,p)

d2=json.load(open('followup_budget.json'))
for w in [4,8,16]:
    rows=[r for r in d2 if r['width']==w]
    acc=np.array([r['test_acc'] for r in rows]); ece=np.array([r['ece'] for r in rows])
    print('600ep',w,'acc',acc.mean(),acc.std(),'ece',ece.mean(),ece.std())
e48=np.array([r['ece'] for r in d2 if r['width'] in (4,8)])
e16=np.array([r['ece'] for r in d2 if r['width']==16])
t2,p2=ttest_ind(e48,e16,equal_var=False)
print('600ep t,p',t2,p2)
