"""
Round-3 review follow-up (reviewer Q3): P_AI was fixed at 0.3. Sweep
P_AI in {0.1, 0.3, 0.5} for none / broadcast / shared_a0.9 on the original
2-community topology to check whether the leakage effect and the
sycophancy-vs-broadcast gap scale with how often the AI pathway fires.
"""
import json
import time
import numpy as np
from scipy import stats
from experiment_v2 import build_block_graph, N, RNG_SEED, EPS, MU, MU_AI, AI_LR, STEPS


def run_condition(mode, alpha, p_ai, rng):
    opinions = rng.random(N)
    adj, community = build_block_graph(rng)
    advisor_state = np.array([0.5]) if mode == "shared" else None
    for step in range(STEPS):
        i = rng.integers(N)
        use_ai = mode != "none" and rng.random() < p_ai
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
    gap = abs(np.mean(opinions[community == 0]) - np.mean(opinions[community == 1]))
    return float(gap)


def main():
    t0 = time.time()
    N_SEEDS = 10
    p_ai_values = [0.1, 0.3, 0.5]
    modes = [("none", 0.0), ("broadcast", 0.0), ("shared", 0.9)]
    results = {}
    for p_ai in p_ai_values:
        gaps = {}
        for m, a in modes:
            key = m if m != "shared" else f"shared_a{a}"
            vals = []
            for s in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED * 1000 + s)
                vals.append(run_condition(m, a, p_ai, rng))
            gaps[key] = vals
        t_sb, p_sb = stats.ttest_rel(gaps["shared_a0.9"], gaps["broadcast"])
        t_bn, p_bn = stats.ttest_rel(gaps["broadcast"], gaps["none"])
        results[f"p_ai_{p_ai}"] = {
            "none_mean": float(np.mean(gaps["none"])),
            "broadcast_mean": float(np.mean(gaps["broadcast"])),
            "shared_a0.9_mean": float(np.mean(gaps["shared_a0.9"])),
            "shared_vs_broadcast": {"t": float(t_sb), "p": float(p_sb)},
            "broadcast_vs_none": {"t": float(t_bn), "p": float(p_bn)},
        }
        print(f"P_AI={p_ai}: none={np.mean(gaps['none']):.4f} "
              f"broadcast={np.mean(gaps['broadcast']):.4f} "
              f"shared_a0.9={np.mean(gaps['shared_a0.9']):.4f} "
              f"(shared vs broadcast p={p_sb:.3g})")

    with open("results_v5_pai_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
