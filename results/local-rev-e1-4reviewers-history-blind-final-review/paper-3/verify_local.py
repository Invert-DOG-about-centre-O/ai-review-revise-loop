import json, statistics
data = json.load(open('multiseed_results.json'))
excl = {1,4,5,9}
sub = [d for d in data if d['seed'] not in excl]
for key in ['auc_mlp','auc_se','auc_sc']:
    vals=[d[key] for d in sub]
    print(key, statistics.mean(vals), statistics.pstdev(vals), statistics.stdev(vals))
for key in ['auc_ent1','auc_mlp','auc_se','auc_sc']:
    vals=[d[key] for d in data]
    print('full', key, statistics.mean(vals), statistics.stdev(vals))
accs=[d['acc'] for d in data]
print('acc range', min(accs), max(accs), statistics.mean(accs), statistics.stdev(accs))

wins_se = sum(1 for d in data if d['auc_mlp']>d['auc_se'])
wins_sc = sum(1 for d in data if d['auc_mlp']>d['auc_sc'])
print('wins vs se, sc (10 seeds):', wins_se, wins_sc)
wins_se6 = sum(1 for d in sub if d['auc_mlp']>d['auc_se'])
wins_sc6 = sum(1 for d in sub if d['auc_mlp']>d['auc_sc'])
print('wins vs se, sc (6 seeds):', wins_se6, wins_sc6)

sig_se = sum(1 for d in data if d['diff_mlp_se']['p_le0']<0.05)
sig_sc = sum(1 for d in data if d['diff_mlp_sc']['p_le0']<0.05)
worse_se = sum(1 for d in data if d['diff_mlp_se']['p_le0']>0.95)
worse_sc = sum(1 for d in data if d['diff_mlp_sc']['p_le0']>0.95)
print('sig se, sc:', sig_se, sig_sc, 'worse se, sc:', worse_se, worse_sc)
