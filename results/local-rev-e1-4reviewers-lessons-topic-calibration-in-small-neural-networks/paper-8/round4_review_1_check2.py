import json, numpy as np
v2 = json.load(open('results_v2.json'))
rob = v2['robustness']
print('widths', rob['widths'], 'seeds', rob['seeds'], 'bayes_acc', rob['bayes_acc'])
res = rob['results']
widths = sorted(set(r['width'] for r in res))
for w in widths:
    Ts = [r['T_star'] for r in res if r['width']==w]
    accs = [r['acc'] for r in res if r['width']==w]
    print(w, 'T*', np.mean(Ts), 'acc', np.mean(accs), 'n=',len(Ts))

# label smoothing table 2 ECE fold changes
v2main = v2['main']['results']
for w in [2,4,16,64,256]:
    ece_nols = np.mean([r['ece'] for r in v2main if r['width']==w and r['label_smoothing']==0.0])
    ece_ls = np.mean([r['ece'] for r in v2main if r['width']==w and r['label_smoothing']==0.1])
    Tnols = np.mean([r['T_star'] for r in v2main if r['width']==w and r['label_smoothing']==0.0])
    Tls = np.mean([r['T_star'] for r in v2main if r['width']==w and r['label_smoothing']==0.1])
    print('width', w, 'T*noLS', round(Tnols,3), 'T*LS', round(Tls,3), 'ECE noLS', round(ece_nols,4), 'ECE LS', round(ece_ls,4), 'fold', round(ece_ls/ece_nols,1))
