"""
Significance testing for the 5-seed maxprob-minus-self-consistency AUROC gap
reported in multiseed_results.json: a paired bootstrap CI on the mean gap
(resampling seeds with replacement) and an exact sign test on the direction.
"""
import json
import numpy as np

with open("multiseed_results.json") as f:
    data = json.load(f)

gaps = np.array([row["gap_maxprob_minus_agree"] for row in data["per_seed"]])
print("per-seed gaps:", gaps.tolist())
print("mean", gaps.mean(), "std", gaps.std(ddof=1))

rng = np.random.RandomState(0)
boot_means = np.array([gaps[rng.randint(0, len(gaps), size=len(gaps))].mean()
                        for _ in range(100_000)])
lo, hi = np.percentile(boot_means, [2.5, 97.5])
print(f"bootstrap 95% CI on mean gap (10^5 resamples of the 5 seeds): [{lo:.4f}, {hi:.4f}]")
print("fraction of bootstrap resamples with mean gap <= 0:", (boot_means <= 0).mean())

p_sign = 0.5 ** len(gaps)
print(f"exact one-sided sign test p-value (all {len(gaps)}/{len(gaps)} seeds same direction under null p=.5): {p_sign}")

accs = [row["acc"] for row in data["per_seed"]]
n = 400
for i, a in enumerate(accs):
    c = round(a * n)
    print(f"seed {i}: acc={a}, correct={c}/{n}, incorrect={n-c}/{n}")
