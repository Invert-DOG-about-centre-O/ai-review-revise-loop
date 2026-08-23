"""
Round-3 review (weakness 3 / question 1): the bridging cluster-geometry
diagnostic (Sec 3.5) reports means +/- SD over the same 10 seeds as the main
experiment but was never permutation-tested. Test whether bridging's
extreme-fraction and cluster-spread differences from similarity/engagement
are statistically distinguishable at n=10, and (via a normal-approximation
power calculation) how many seeds would be needed for any claim that isn't
yet significant.
"""
import json
import numpy as np

with open("results.json") as f:
    main_results = json.load(f)


def cluster_centers(opinions, gap=0.08):
    s = np.sort(np.asarray(opinions))
    if len(s) == 0:
        return []
    splits = np.where(np.diff(s) > gap)[0] + 1
    return [float(np.mean(g)) for g in np.split(s, splits)]


def metrics_for(policy):
    rows = [r for r in main_results if r["policy"] == policy]
    spreads, extreme_frac = [], []
    for r in rows:
        centers = cluster_centers(r["final_opinions"])
        spreads.append(max(centers) - min(centers) if len(centers) >= 2 else 0.0)
        ops = np.array(r["final_opinions"])
        extreme_frac.append(float(np.mean(np.abs(ops) > 0.5)))
    return np.array(spreads), np.array(extreme_frac)


def permutation_test(a, b, n_perm=100000, seed=0):
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:na].mean() - perm[na:].mean()
    p = np.mean(np.abs(diffs) >= np.abs(obs))
    return obs, p


def n_needed_for_sig(a, b, target_p=0.05, alpha_z=1.96):
    # normal-approx two-sample power calc: n per group needed for the
    # observed effect size (pooled SD) to reach ~alpha_z z-score
    diff = abs(a.mean() - b.mean())
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if diff == 0:
        return float("inf")
    n = 2 * (alpha_z * pooled_sd / diff) ** 2
    return n


sim_spread, sim_extreme = metrics_for("similarity")
eng_spread, eng_extreme = metrics_for("engagement")
brg_spread, brg_extreme = metrics_for("bridging")

for name, a, b, alab, blab in [
    ("extreme_frac", brg_extreme, eng_extreme, "bridging", "engagement"),
    ("extreme_frac", brg_extreme, sim_extreme, "bridging", "similarity"),
    ("cluster_spread", brg_spread, eng_spread, "bridging", "engagement"),
    ("cluster_spread", brg_spread, sim_spread, "bridging", "similarity"),
]:
    obs, p = permutation_test(a, b, seed=hash(name + alab) % 1000)
    n_req = n_needed_for_sig(a, b)
    print(f"{name} {alab}-{blab}: diff={obs:+.4f}, p={p:.4f}, "
          f"approx n/group for p<0.05 = {n_req:.1f}")
