"""
Revision-round experiment: extends experiment.py to (1) add a non-sycophantic
"broadcast" advisor condition that isolates the marginal contribution of mere
shared/all-to-all state from the marginal contribution of sycophancy, (2)
increase seeds from 5 to 20 and run paired significance tests (paired t-test
across matched seeds, since each seed s uses the same RNG seed -> same graph
and initial opinions -> across all modes/alphas), and (3) a robustness check
of the personalized-condition U-shaped alpha curve under a different AI
learning rate.
"""
import json
import time
import numpy as np
from scipy import stats

RNG_SEED = 0
N_PER_COMMUNITY = 60
N_COMMUNITIES = 2
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
    community = np.array([0] * N_PER_COMMUNITY + [1] * N_PER_COMMUNITY)
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


def run_condition(mode, alpha, rng, ai_lr=AI_LR):
    """mode: 'none' | 'shared' | 'personalized' | 'broadcast'."""
    opinions = rng.random(N)
    adj, community = build_block_graph(rng)

    if mode == "shared":
        advisor_state = np.array([0.5])
    elif mode == "personalized":
        advisor_state = rng.random(N) * 0 + 0.5
    else:
        advisor_state = None

    for step in range(STEPS):
        i = rng.integers(N)
        use_ai = mode != "none" and rng.random() < P_AI
        if use_ai:
            if mode == "broadcast":
                # Non-sycophantic: advisor always relays the TRUE current
                # global population mean (all-to-all coupling, no pull
                # toward the individual user's stated opinion).
                response = float(np.mean(opinions))
            else:
                a_idx = 0 if mode == "shared" else i
                a = advisor_state[a_idx]
                response = (1 - alpha) * a + alpha * opinions[i]
            if abs(response - opinions[i]) < EPS:
                opinions[i] += MU_AI * (response - opinions[i])
            if mode in ("shared", "personalized"):
                a_idx = 0 if mode == "shared" else i
                advisor_state[a_idx] += ai_lr * (opinions[i] - advisor_state[a_idx])
        else:
            neighbors = adj[i]
            if not neighbors:
                continue
            j = neighbors[rng.integers(len(neighbors))]
            if abs(opinions[i] - opinions[j]) < EPS:
                delta = opinions[j] - opinions[i]
                opinions[i] += MU * delta
                opinions[j] -= MU * delta

    within_var = np.mean([np.var(opinions[community == c]) for c in range(N_COMMUNITIES)])
    between_gap = abs(np.mean(opinions[community == 0]) - np.mean(opinions[community == 1]))
    global_var = np.var(opinions)
    return {"within_var": float(within_var), "between_gap": float(between_gap),
            "global_var": float(global_var)}


def run_sweep(n_seeds, alphas, modes, ai_lr=AI_LR, label=""):
    results = {}
    per_seed_gaps = {}
    for mode in modes:
        alpha_list = [None] if mode in ("none", "broadcast") else alphas
        for alpha in alpha_list:
            key = f"{mode}" if alpha is None else f"{mode}_a{alpha}"
            gaps, within_vars, global_vars = [], [], []
            for s in range(n_seeds):
                rng = np.random.default_rng(RNG_SEED * 1000 + s)
                r = run_condition(mode, alpha if alpha is not None else 0.0, rng, ai_lr=ai_lr)
                gaps.append(r["between_gap"])
                within_vars.append(r["within_var"])
                global_vars.append(r["global_var"])
            per_seed_gaps[key] = gaps
            results[key] = {
                "between_gap_mean": float(np.mean(gaps)),
                "between_gap_std": float(np.std(gaps)),
                "within_var_mean": float(np.mean(within_vars)),
                "global_var_mean": float(np.mean(global_vars)),
                "n_seeds": n_seeds,
            }
    if "none" in results:
        baseline_gap = results["none"]["between_gap_mean"]
        for key, agg in results.items():
            agg["leakage_index"] = float(baseline_gap - agg["between_gap_mean"])
    print(f"\n=== {label} (n_seeds={n_seeds}) ===")
    for key, agg in results.items():
        leak = agg.get("leakage_index", float("nan"))
        print(f"{key:24s} gap={agg['between_gap_mean']:.4f} (+-{agg['between_gap_std']:.4f})"
              f" leak={leak:.4f}")
    return results, per_seed_gaps


def paired_t(gaps_a, gaps_b, name_a, name_b):
    t, p = stats.ttest_rel(gaps_a, gaps_b)
    print(f"paired t-test {name_a} vs {name_b}: t={t:.3f}, p={p:.4g}")
    return {"t": float(t), "p": float(p)}


def main():
    t0 = time.time()
    alphas = [0.0, 0.3, 0.6, 0.9, 1.0]
    N_SEEDS = 20

    # Main sweep: none / personalized / shared / broadcast, 20 seeds
    modes_main = ["none", "personalized", "shared", "broadcast"]
    results, gaps = run_sweep(N_SEEDS, alphas, modes_main, label="main sweep")

    sig = {}
    # Q1: does the shared advisor's leakage effect at alpha=0.9 significantly
    # beat the non-sycophantic broadcast condition (isolating sycophancy's
    # marginal contribution from mere shared-state hub-bridging)?
    sig["shared_a0.9_vs_broadcast"] = paired_t(gaps["shared_a0.9"], gaps["broadcast"],
                                                "shared_a0.9", "broadcast")
    sig["shared_a0.9_vs_none"] = paired_t(gaps["shared_a0.9"], gaps["none"],
                                           "shared_a0.9", "none")
    sig["broadcast_vs_none"] = paired_t(gaps["broadcast"], gaps["none"],
                                         "broadcast", "none")
    # Q3: is shared vs personalized ranking significant at matched alpha?
    for a in alphas:
        key_s, key_p = f"shared_a{a}", f"personalized_a{a}"
        sig[f"shared_vs_personalized_a{a}"] = paired_t(gaps[key_s], gaps[key_p], key_s, key_p)

    # Q2: robustness of the personalized U-shape under a different AI_LR
    results_lr, gaps_lr = run_sweep(N_SEEDS, alphas, ["personalized"], ai_lr=0.15,
                                     label="personalized robustness (AI_LR=0.15)")

    out = {"main": results, "significance": sig,
           "personalized_alt_lr015": results_lr}
    with open("results_v2.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
