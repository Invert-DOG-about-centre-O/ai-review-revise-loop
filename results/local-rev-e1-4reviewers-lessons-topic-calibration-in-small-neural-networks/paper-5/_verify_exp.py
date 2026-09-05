import json, statistics
d=json.load(open('raw_results.json'))
rows=[r for r in d['results'] if r['dataset']=='digits' and r['width']==2]
print(len(rows))
temps=[r['temperature'] for r in rows]
ece_ts=[r['ece_ts'] for r in rows]
ece_raw=[r['ece_raw'] for r in rows]
print('temp mean/sd', statistics.mean(temps), statistics.pstdev(temps))
print('ece_ts mean/sd/min/max', statistics.mean(ece_ts), statistics.pstdev(ece_ts), min(ece_ts), max(ece_ts))
print('ece_raw mean/sd', statistics.mean(ece_raw), statistics.pstdev(ece_raw))
print(rows[0].keys())

# overall counts
allr = d['results']
worse = sum(1 for r in allr if r['ece_ts'] > r['ece_raw'])
print('overall TS worse count', worse, len(allr))
low = [r for r in allr if r['acc_test'] < 0.4]
worse_low = sum(1 for r in low if r['ece_ts'] > r['ece_raw'])
print('low-acc count', len(low), 'worse', worse_low)
