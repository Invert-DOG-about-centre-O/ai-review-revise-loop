"""
Round-3 review follow-up: (1) generalize the topology from 2 to 3 equal,
structurally disconnected communities (N=120, 40/community) to test whether
the sycophancy-vs-broadcast distinction (Finding 1) survives beyond the
simplest two-block case, and (2) apply a single Holm-Bonferroni correction
across all 11 tests (the 8 main-sweep tests + the 3 n=40 seed-sensitivity
tests) to answer whether pooling changes any conclusion.
"""
import json
import time
import numpy as np
from scipy import stats

RNG_SEED = 0
N_COMMUNITIES = 3
N_PER_COMMUNITY = 40
N = N_PER_COMMUNITY * N_COMMUNITIES
EPS = 0.15
MU = 0.3
MU_AI = 0.3
AI_LR = 0.05
P_AI = 0.3
STEPS = 40000
INTRA_P = 0.15
INTER_P = 0.0


def build_block_graph(rng):
    adj = [[] for _ in range(N)]
    community = np.repeat(np.arange(N_COMMUNITIES), N_PER_COMMUNITY)
    for i in range(N):
        for j in range(i + 1, N):
            p = INTRA_P if community[i] == community[j] else INTER_P
            if rng.random() < p:
                adj[i].append(j)
                adj[j].append(i)
    for i in range(N):
        if not adj[i]:
            same = [j for j in range(N) if j != i and community[j] == community[i]]
            j = rng.choice(same)
            adj[i].append(j)
            adj[j].append(i)
    return adj, community


def run_condition(mode, alpha, rng):
    opinions = rng.random(N)
    adj, community = build_block_graph(rng)

    if mode == "shared":
        advisor_state = np.array([0.5])
    else:
        advisor_state = None

    for step in range(STEPS):
        i = rng.integers(N)
        use_ai = mode != "none" and rng.random() < P_AI
        if use_ai:
            if mode == "broadcast":
                response = float(np.mean(opinions))
            else:
                a = advisor_state[0]
                response = (1 - alpha) * a + alpha * opinions[i]
            if abs(response - opinions[i]) < EPS:
                opinions[i] += MU_AI * (response - opinions[i])
            if mode == "shared":
                advisor_state[0] += AI_LR * (opinions[i] - advisor_state[0])
        else:
            neighbors = adj[i]
            if not neighbors:
                continue
            j = neighbors[rng.integers(len(neighbors))]
            if abs(opinions[i] - opinions[j]) < EPS:
                delta = opinions[j] - opinions[i]
                opinions[i] += MU * delta
                opinions[j] -= MU * delta

    means = np.array([np.mean(opinions[community == c]) for c in range(N_COMMUNITIES)])
    # generalize "between-community gap" to max pairwise |mean_a - mean_b|
    max_gap = float(np.max(np.abs(means[:, None] - means[None, :])))
    within_var = float(np.mean([np.var(opinions[community == c]) for c in range(N_COMMUNITIES)]))
    return {"max_pairwise_gap": max_gap, "within_var": within_var}


def main():
    t0 = time.time()
    N_SEEDS = 15
    modes = [("none", 0.0), ("broadcast", 0.0), ("shared", 0.9)]
    gaps = {m if a == 0.0 and m != "shared" else f"{m}_a{a}": [] for m, a in modes}
    keys = list(gaps.keys())
    for m, a in modes:
        key = m if m != "shared" else f"shared_a{a}"
        for s in range(N_SEEDS):
            rng = np.random.default_rng(RNG_SEED * 1000 + s)
            r = run_condition(m, a, rng)
            gaps[key].append(r["max_pairwise_gap"])

    summary = {}
    for key, vals in gaps.items():
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": N_SEEDS}
    baseline = summary["none"]["mean"]
    for key in summary:
        summary[key]["leakage_index"] = float(baseline - summary[key]["mean"])

    print(f"=== 3-community topology (N={N}, {N_PER_COMMUNITY}/community, n_seeds={N_SEEDS}) ===")
    for key, s in summary.items():
        print(f"{key:16s} gap={s['mean']:.4f} (+-{s['std']:.4f}) leak={s['leakage_index']:.4f}")

    sig = {}
    t, p = stats.ttest_rel(gaps["shared_a0.9"], gaps["none"])
    sig["shared_a0.9_vs_none"] = {"t": float(t), "p": float(p)}
    print(f"paired t-test shared_a0.9 vs none: t={t:.3f}, p={p:.4g}")
    t, p = stats.ttest_rel(gaps["shared_a0.9"], gaps["broadcast"])
    sig["shared_a0.9_vs_broadcast"] = {"t": float(t), "p": float(p)}
    print(f"paired t-test shared_a0.9 vs broadcast: t={t:.3f}, p={p:.4g}")
    t, p = stats.ttest_rel(gaps["broadcast"], gaps["none"])
    sig["broadcast_vs_none"] = {"t": float(t), "p": float(p)}
    print(f"paired t-test broadcast vs none: t={t:.3f}, p={p:.4g}")

    # Pooled Holm-Bonferroni across all 11 two-community-round tests (8 main + 3 n=40 follow-up)
    with open("results_v2.json") as f:
        v2 = json.load(f)
    with open("results_v3_followup.json") as f:
        v3 = json.load(f)
    pooled_p = {f"main_{k}": v["p"] for k, v in v2["significance"].items()}
    pooled_p.update({f"n40_{k}": v["p"] for k, v in v3["seed_sensitivity_n40"].items()})
    items = sorted(pooled_p.items(), key=lambda kv: kv[1])
    m = len(items)
    holm_pooled = {}
    running_max = 0.0
    for rank, (name, p_) in enumerate(items, start=1):
        adj_p = min(1.0, (m - rank + 1) * p_)
        running_max = max(running_max, adj_p)
        holm_pooled[name] = {"p_raw": p_, "p_holm": running_max, "reject_at_0.05": running_max < 0.05}
    print(f"\nPooled Holm-Bonferroni across all {m} tests (main 8 + n40 follow-up 3):")
    for k, v in holm_pooled.items():
        print(f"  {k:36s} p_raw={v['p_raw']:.4g} p_holm={v['p_holm']:.4g} reject@0.05={v['reject_at_0.05']}")

    out = {"three_community": summary, "three_community_sig": sig, "pooled_holm_11tests": holm_pooled}
    with open("results_v4_3community.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
