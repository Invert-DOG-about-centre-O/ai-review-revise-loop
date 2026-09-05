import json
from scipy.stats import wilcoxon
import numpy as np

with open("results.json") as f:
    d = json.load(f)

res = d['results']

def arr(dataset, cond, phase, metric):
    return np.array([r[phase][metric] for r in res[dataset][cond]])

for dataset in ['digits', 'synthetic']:
    sl_pre = arr(dataset, 'single_large', 'pre', 'ece')
    en_pre = arr(dataset, 'ensemble4', 'pre', 'ece')
    sl_post = arr(dataset, 'single_large', 'post', 'ece')
    en_post = arr(dataset, 'ensemble4', 'post', 'ece')
    w_pre = wilcoxon(en_pre, sl_pre, alternative='greater')
    w_post = wilcoxon(en_post, sl_post, alternative='greater')
    print(dataset, 'pre-TS ECE means', sl_pre.mean(), en_pre.mean(), 'p=', w_pre.pvalue)
    print(dataset, 'post-TS ECE means', sl_post.mean(), en_post.mean(), 'p=', w_post.pvalue)
    ss_post = arr(dataset, 'single_small', 'post', 'ece')
    print(dataset, 'single_small post ECE mean', ss_post.mean())
    acc_sl = arr(dataset,'single_large','pre','acc').mean()
    acc_en = arr(dataset,'ensemble4','pre','acc').mean()
    acc_ss = arr(dataset,'single_small','pre','acc').mean()
    conf_sl = arr(dataset,'single_large','pre','mean_conf').mean()
    conf_en = arr(dataset,'ensemble4','pre','mean_conf').mean()
    print(dataset, 'acc SL/EN/SS', acc_sl, acc_en, acc_ss, 'conf SL/EN', conf_sl, conf_en)
    print()

with open("mechanism_results.json") as f:
    m = json.load(f)
print(list(m.keys()))
