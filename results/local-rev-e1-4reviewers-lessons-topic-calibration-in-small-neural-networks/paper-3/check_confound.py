import json, numpy as np
from scipy.stats import spearmanr

np.random.seed(0)

def load(dataset, path='results_raw.json'):
    with open(path) as f:
        results = json.load(f)
    rows = [r for r in results if r['dataset'] == dataset]
    pre = np.array([r['pre_ece'] for r in rows])
    post = np.array([r['post_ece'] for r in rows])
    return pre, post

datasets = [('digits', 'results_raw.json'), ('synthetic', 'results_raw.json')]
with open('results_dataset3_raw.json') as f:
    d3 = json.load(f)
pre3 = np.array([r['pre_ece'] for r in d3])
post3 = np.array([r['post_ece'] for r in d3])

all_data = {}
for name, path in datasets:
    pre, post = load(name, path)
    all_data[name] = (pre, post)
all_data['breast_cancer'] = (pre3, post3)

N_PERM = 20000
print(f"{'dataset':12s} {'corr(pre,post)':>15s} {'obs rho(pre,delta)':>20s} {'obs p':>10s} {'perm-null mean':>15s} {'perm-null 95%CI':>20s} {'perm p (obs exceeds null)':>26s}")
for name, (pre, post) in all_data.items():
    delta = pre - post
    obs_rho, obs_p = spearmanr(pre, delta)
    corr_pre_post, p_pre_post = spearmanr(pre, post)

    null_rhos = np.empty(N_PERM)
    rng = np.random.default_rng(42)
    for i in range(N_PERM):
        post_perm = rng.permutation(post)
        delta_perm = pre - post_perm
        null_rhos[i] = spearmanr(pre, delta_perm)[0]
    perm_p = (np.sum(null_rhos >= obs_rho) + 1) / (N_PERM + 1)
    ci = np.percentile(null_rhos, [2.5, 97.5])
    print(f"{name:12s} {corr_pre_post:15.4f} {obs_rho:20.4f} {obs_p:10.2g} {null_rhos.mean():15.4f} [{ci[0]:.3f},{ci[1]:.3f}]        {perm_p:10.4g}")

    # partial correlation of delta vs pre controlling nothing extra needed; also regression slope test
    # regression: post = a + b*pre ; test b vs 1 (if b<1, low pre-ece models see little change -> mechanical)
    b, a = np.polyfit(pre, post, 1)
    print(f"    regression post = {a:.4f} + {b:.4f}*pre  (slope<1 => reduction correlates with pre mechanically)")
