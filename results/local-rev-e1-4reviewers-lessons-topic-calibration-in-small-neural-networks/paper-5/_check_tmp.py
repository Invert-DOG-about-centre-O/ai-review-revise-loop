import json, statistics as st
d = json.load(open('raw_results.json'))
print(type(d), len(d))
runs = d['results']

print(runs[0].keys())

worse = sum(1 for r in runs if r['ece_ts'] > r['ece_raw'])
print('TS worse count/total:', worse, len(runs), worse/len(runs))

low_acc = [r for r in runs if r['acc_test'] < 0.4]
worse_low = sum(1 for r in low_acc if r['ece_ts'] > r['ece_raw'])
print('low acc n:', len(low_acc), 'worse:', worse_low, worse_low/len(low_acc) if low_acc else None)

d16 = [r for r in runs if r['dataset']=='digits' and r['width']==2]
print('digits width2 n:', len(d16))
temps = [r['temperature'] for r in d16]
ece_ts = [r['ece_ts'] for r in d16]
ece_raw = [r['ece_raw'] for r in d16]
print('temp mean/std:', st.mean(temps), st.pstdev(temps))
print('ece_ts mean/std/min/max:', st.mean(ece_ts), st.pstdev(ece_ts), min(ece_ts), max(ece_ts))
print('ece_raw mean/std:', st.mean(ece_raw), st.pstdev(ece_raw))
accs = [r['acc_test'] for r in d16]
print('acc mean:', st.mean(accs))
