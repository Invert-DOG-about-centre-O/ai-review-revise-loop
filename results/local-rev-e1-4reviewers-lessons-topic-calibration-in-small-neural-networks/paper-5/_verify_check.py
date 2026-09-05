import json, statistics as st
d = json.load(open('raw_results.json'))['results']
print(len(d))

def filt(dataset, cond, width):
    return [r for r in d if r['dataset']==dataset and r['condition']==cond and r['width']==width]

for w in [2,4,16,32,256]:
    rows = filt('digits','baseline',w)
    accs = [r['acc_test'] for r in rows]
    eces = [r['ece_raw'] for r in rows]
    print('digits baseline width', w, 'n=',len(rows),'mean acc', round(st.mean(accs),3), 'mean ece_raw', round(st.mean(eces),3))

print()
for (ds,w) in [('digits',4),('blobs',2)]:
    rows = filt(ds,'baseline',w)
    accs=[r['acc_test'] for r in rows]
    ece_raw=[r['ece_raw'] for r in rows]
    ece_ts=[r['ece_ts'] for r in rows]
    worse = sum(1 for r in rows if r['ece_ts']>r['ece_raw'])
    print(ds,w,'n',len(rows),'acc',round(st.mean(accs),3),'raw',round(st.mean(ece_raw),3),'ts',round(st.mean(ece_ts),3),'worse',worse)

all_rows = [r for r in d]
inc = sum(1 for r in all_rows if r['ece_ts']>r['ece_raw'])
print('total rows', len(all_rows), 'ts worse count', inc, inc/len(all_rows))
lowacc = [r for r in all_rows if r['acc_test']<0.4]
incl = sum(1 for r in lowacc if r['ece_ts']>r['ece_raw'])
print('lowacc n', len(lowacc), 'worse', incl, incl/len(lowacc) if lowacc else None)
