"""
Sanity check for the H3 permutation-null test used in analysis.py.

The H3 test (does TS's benefit relate to ece_pre, beyond the arithmetic
coupling reduction = ece_pre - ece_post) uses a permutation null that
shuffles ece_post independently of ece_pre. Because "post" here is not
provably a free variable (a skeptical reviewer could ask whether it is
secretly a deterministic function of "pre"), we verify the permutation
test's own false-positive rate under a KNOWN true null: pre and post
independently sampled, no relationship of any kind.
"""
import numpy as np
from scipy import stats
import json


def perm_test(pre, post, n_perm=300, seed=0):
    reduction = pre - post
    obs_r, _ = stats.pearsonr(pre, reduction)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        post_shuf = rng.permutation(post)
        red_shuf = pre - post_shuf
        r, _ = stats.pearsonr(pre, red_shuf)
        if abs(r) >= abs(obs_r):
            count += 1
    return (count + 1) / (n_perm + 1)


rng = np.random.RandomState(999)
n = 160  # matches per-dataset n in the real experiment
n_sims = 600
n_perm = 300
false_pos = 0
for s in range(n_sims):
    pre = rng.gamma(2, 1, n)
    post = rng.gamma(2, 1, n)  # independent of pre by construction: true null
    p = perm_test(pre, post, n_perm=n_perm, seed=5000 + s)
    if p < 0.05:
        false_pos += 1

fpr = false_pos / n_sims
se = (0.05 * 0.95 / n_sims) ** 0.5
result = {
    "n_sims": n_sims, "n_perm": n_perm, "n_per_sample": n,
    "false_positive_rate": fpr, "nominal_alpha": 0.05,
    "binomial_se_under_nominal": se,
    "within_2se_of_nominal": abs(fpr - 0.05) < 2 * se,
}
print(result)
with open("round1_review_3_sanity_results.json", "w") as f:
    json.dump(result, f, indent=2)
