import json
import collections
import numpy as np
from scipy import stats

with open("raw_results_extgrid.json") as f:
    data = json.load(f)

results = data["results"]
by_ds = collections.defaultdict(list)
for r in results:
    by_ds[r["dataset"]].append(r)


def permutation_corr_pre_reduction(ece_pre, ece_post, n_perm=10000, seed=0):
    ece_pre = np.array(ece_pre)
    ece_post = np.array(ece_post)
    reduction = ece_pre - ece_post
    obs_r, _ = stats.pearsonr(ece_pre, reduction)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        post_shuf = rng.permutation(ece_post)
        red_shuf = ece_pre - post_shuf
        r, _ = stats.pearsonr(ece_pre, red_shuf)
        if abs(r) >= abs(obs_r):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs_r, p


for ds, rows in by_ds.items():
    ece_pre = [r["ece_pre"] for r in rows]
    ece_post = [r["ece_post"] for r in rows]
    naive_r, naive_p = stats.pearsonr(ece_pre, [a-b for a,b in zip(ece_pre,ece_post)])
    r3, p3 = permutation_corr_pre_reduction(ece_pre, ece_post, seed=3)
    print(ds, "naive_r=", naive_r, "naive_p=", naive_p, "perm_p=", p3)
