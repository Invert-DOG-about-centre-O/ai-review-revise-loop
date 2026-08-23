import json, math
d = json.load(open('robustness_results.json'))
l2 = d['sycophancy_penalty']
pc = d['precommit_answer']
print('L2 mean/std', l2['sycophancy_ambiguous_mean'], l2['sycophancy_ambiguous_std'])
print('PC mean/std', pc['sycophancy_ambiguous_mean'], pc['sycophancy_ambiguous_std'])
n = 10
gap = pc['sycophancy_ambiguous_mean'] - l2['sycophancy_ambiguous_mean']
se_of_mean = math.sqrt((l2['sycophancy_ambiguous_std']**2 + pc['sycophancy_ambiguous_std']**2) / n)
print('gap', gap, 'pooled SE of the mean (std/sqrt(n))', se_of_mean, 'z', gap/se_of_mean)

out = d['naive_rlhf_with_outliers']
print('g mean/std outlier', out['g_mean'], out['g_std'])
print('sycophancy_ambiguous std outlier', out['sycophancy_ambiguous_std'])
print('accuracy std outlier', out['accuracy_std'])

# cost-normalized comparison using crossed no_outlier data
naive = d['crossed']['naive__no_outlier']
for name in ['l2_penalty', 'precommit', 'l2_plus_precommit', 'rubric_reweight', 'median_agg']:
    row = d['crossed'][f'{name}__no_outlier']
    benefit = naive['sycophancy_ambiguous_mean'] - row['sycophancy_ambiguous_mean']
    cost = naive['approval_proxy_mean'] - row['approval_proxy_mean']
    ratio = benefit / cost if cost != 0 else float('inf')
    print(f'{name}: benefit={benefit:.4f} cost={cost:.4f} ratio={ratio:.3f}')
