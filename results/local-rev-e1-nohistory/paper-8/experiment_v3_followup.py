"""
Follow-up for round-2 review: (1) re-run none/broadcast/shared_a0.9 at 40 seeds
(seeds 0..39, a superset of the original 20) to check whether the
broadcast-vs-none p=0.053 result is seed-count-sensitive, and (2) apply a
Holm-Bonferroni correction to the 8 significance tests already computed in
results_v2.json.
"""
import json
import time
import numpy as np
from scipy import stats
from experiment_v2 import run_condition, RNG_SEED

def paired_t(a, b, name_a, name_b):
    t, p = stats.ttest_rel(a, b)
    print(f"paired t-test {name_a} vs {name_b}: t={t:.3f}, p={p:.4g}")
    return {"t": float(t), "p": float(p)}

def holm_bonferroni(pvals_dict):
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, adj)
        out[name] = {"p_raw": p, "p_holm": running_max, "reject_at_0.05": running_max < 0.05}
    return out

def main():
    t0 = time.time()
    N_SEEDS = 40
    gaps = {"none": [], "broadcast": [], "shared_a0.9": []}
    for s in range(N_SEEDS):
        rng = np.random.default_rng(RNG_SEED * 1000 + s)
        gaps["none"].append(run_condition("none", 0.0, rng)["between_gap"])
        rng = np.random.default_rng(RNG_SEED * 1000 + s)
        gaps["broadcast"].append(run_condition("broadcast", 0.0, rng)["between_gap"])
        rng = np.random.default_rng(RNG_SEED * 1000 + s)
        gaps["shared_a0.9"].append(run_condition("shared", 0.9, rng)["between_gap"])

    print(f"none mean={np.mean(gaps['none']):.4f} broadcast mean={np.mean(gaps['broadcast']):.4f} "
          f"shared_a0.9 mean={np.mean(gaps['shared_a0.9']):.4f}  (n=40)")
    sig40 = {}
    sig40["broadcast_vs_none_n40"] = paired_t(gaps["broadcast"], gaps["none"], "broadcast", "none")
    sig40["shared_a0.9_vs_broadcast_n40"] = paired_t(gaps["shared_a0.9"], gaps["broadcast"], "shared_a0.9", "broadcast")
    sig40["shared_a0.9_vs_none_n40"] = paired_t(gaps["shared_a0.9"], gaps["none"], "shared_a0.9", "none")

    # Also compute at n=20 subset (first 20 of the 40) for direct before/after comparison
    sig20 = {}
    sig20["broadcast_vs_none_n20subset"] = paired_t(gaps["broadcast"][:20], gaps["none"][:20], "broadcast", "none")

    with open("results_v2.json") as f:
        prior = json.load(f)
    holm = holm_bonferroni({k: v["p"] for k, v in prior["significance"].items()})
    print("\nHolm-Bonferroni correction over the 8 round-2 tests:")
    for k, v in holm.items():
        print(f"  {k:32s} p_raw={v['p_raw']:.4g} p_holm={v['p_holm']:.4g} reject@0.05={v['reject_at_0.05']}")

    out = {"seed_sensitivity_n40": sig40, "seed_sensitivity_n20_subset": sig20, "holm_bonferroni": holm}
    with open("results_v3_followup.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWall time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
