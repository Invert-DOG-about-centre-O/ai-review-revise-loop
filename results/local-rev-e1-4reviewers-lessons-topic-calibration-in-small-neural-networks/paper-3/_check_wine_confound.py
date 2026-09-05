import json, numpy as np
from scipy.stats import spearmanr

with open('results_dataset4_raw.json') as f:
    d4 = json.load(f)
pre = np.array([r['pre_ece'] for r in d4])
post = np.array([r['post_ece'] for r in d4])
delta = pre - post
obs_rho, obs_p = spearmanr(pre, delta)
corr_pre_post, _ = spearmanr(pre, post)
N_PERM = 20000
rng = np.random.default_rng(42)
null_rhos = np.empty(N_PERM)
for i in range(N_PERM):
    post_perm = rng.permutation(post)
    delta_perm = pre - post_perm
    null_rhos[i] = spearmanr(pre, delta_perm)[0]
perm_p = (np.sum(null_rhos >= obs_rho) + 1) / (N_PERM + 1)
ci = np.percentile(null_rhos, [2.5, 97.5])
print('corr(pre,post)=%.4f obs_rho=%.4f null_mean=%.4f CI=[%.3f,%.3f] perm_p=%.4g' % (corr_pre_post, obs_rho, null_rhos.mean(), ci[0], ci[1], perm_p))
