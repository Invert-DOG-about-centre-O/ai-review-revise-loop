"""Validate the permutation-null diagnostic itself (reviewer question, round3):
is its false-positive rate controlled when pre-ECE and post-ECE are truly
independent by construction, given post-ECE is not an arbitrary shuffle but a
deterministic function of the same underlying model as pre-ECE?

Procedure: for each dataset, take the empirical marginal distribution of
pre-ECE and of post-ECE (each n=140). Build N_SIM synthetic "null" datasets by
independently resampling (with replacement) from each marginal separately, so
pre and post are independent by construction (true corr = 0). For each
synthetic dataset, run the same permutation-null procedure as
check_confound.py and record whether it falsely reports perm_p < 0.05. The
false-positive rate across the synthetic datasets should be close to 0.05 if
the diagnostic is well-calibrated. Vectorized with numpy for speed.
"""
import json, numpy as np

def rank_matrix(x):
    # average ranks along last axis, vectorized (ties handled approximately via argsort;
    # ECE values are continuous floats so exact ties are rare)
    order = np.argsort(x, axis=-1)
    ranks = np.empty_like(order, dtype=float)
    idx = np.arange(x.shape[-1])
    np.put_along_axis(ranks, order, idx, axis=-1)
    return ranks

def spearman_vec(a_ranks, b_matrix_ranks):
    # a_ranks: (n,) fixed; b_matrix_ranks: (m, n) varying -> returns (m,) pearson corr of ranks
    a_c = a_ranks - a_ranks.mean()
    b_c = b_matrix_ranks - b_matrix_ranks.mean(axis=1, keepdims=True)
    num = (b_c * a_c).sum(axis=1)
    den = np.sqrt((a_c**2).sum()) * np.sqrt((b_c**2).sum(axis=1))
    return num / den

def load(dataset, path='results_raw.json'):
    with open(path) as f:
        results = json.load(f)
    rows = [r for r in results if r['dataset'] == dataset]
    pre = np.array([r['pre_ece'] for r in rows])
    post = np.array([r['post_ece'] for r in rows])
    return pre, post

datasets = {}
datasets['digits'] = load('digits')
datasets['synthetic'] = load('synthetic')
with open('results_dataset3_raw.json') as f:
    d3 = json.load(f)
datasets['breast_cancer'] = (np.array([r['pre_ece'] for r in d3]),
                              np.array([r['post_ece'] for r in d3]))

N_SIM = 500
N_PERM = 2000
ALPHA = 0.05
rng = np.random.default_rng(123)

print(f"{'dataset':12s} {'n':>5s} {'false-positive rate':>20s} {'expected':>10s} {'95% MC-CI':>18s}")
for name, (pre, post) in datasets.items():
    n = len(pre)
    false_pos = 0
    for s in range(N_SIM):
        pre_s = rng.choice(pre, size=n, replace=True)
        post_s = rng.choice(post, size=n, replace=True)  # independent resample -> true corr 0
        delta_s = pre_s - post_s
        rpre = rank_matrix(pre_s)
        rdelta_obs = rank_matrix(delta_s)
        obs_rho = float(spearman_vec(rpre, rdelta_obs[None, :])[0])

        # generate N_PERM permutations of post_s at once
        perm_idx = np.argsort(rng.random((N_PERM, n)), axis=1)
        post_perm_matrix = post_s[perm_idx]
        delta_perm_matrix = pre_s[None, :] - post_perm_matrix
        rdelta_perm = rank_matrix(delta_perm_matrix)
        null_rhos = spearman_vec(rpre, rdelta_perm)

        perm_p = (np.sum(null_rhos >= obs_rho) + 1) / (N_PERM + 1)
        if perm_p < ALPHA:
            false_pos += 1
    rate = false_pos / N_SIM
    se = np.sqrt(ALPHA * (1 - ALPHA) / N_SIM)
    print(f"{name:12s} {n:5d} {rate:20.4f} {ALPHA:10.2f} [{max(0,rate-1.96*se):.3f},{rate+1.96*se:.3f}]")
