import json
import numpy as np

d = json.load(open('raw_results.json'))['results']
print(len(d))
print(d[0].keys())

worse=0
tot=0
lowacc_worse=0
lowacc_tot=0
for r in d:
    tot+=1
    if r['ece_ts']>r['ece_raw']:
        worse+=1
    if r['acc_test']<0.4:
        lowacc_tot+=1
        if r['ece_ts']>r['ece_raw']:
            lowacc_worse+=1
print('worse',worse,'tot',tot, worse/tot)
print('lowacc worse',lowacc_worse,'tot',lowacc_tot, lowacc_worse/lowacc_tot if lowacc_tot else None)

sl=[r for r in d if r['dataset']=='digits' and r['width']==2 and r['condition']=='baseline']
print('n',len(sl))
accs=[r['acc_test'] for r in sl]
print('mean acc', np.mean(accs))
temps=[r['temperature'] for r in sl]
print('temp mean/sd', np.mean(temps), np.std(temps,ddof=1))
ece_ts=[r['ece_ts'] for r in sl]
print('ece_ts mean/sd', np.mean(ece_ts), np.std(ece_ts,ddof=1))
ece_raw=[r['ece_raw'] for r in sl]
print('ece_raw mean/sd', np.mean(ece_raw), np.std(ece_raw,ddof=1))

b = [r for r in d if r['dataset']=='blobs' and r['width']==2 and r['condition']=='baseline']
dg = [r for r in d if r['dataset']=='digits' and r['width']==4 and r['condition']=='baseline']
print('blobs w2 acc', np.mean([r['acc_test'] for r in b]))
print('digits w4 acc', np.mean([r['acc_test'] for r in dg]))
print('blobs w2 improve count', sum(1 for r in b if r['ece_ts']<r['ece_raw']), len(b))
print('digits w4 worsen count', sum(1 for r in dg if r['ece_ts']>r['ece_raw']), len(dg))
