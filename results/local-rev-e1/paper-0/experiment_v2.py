"""
Follow-up experiments for round-1 review of the AI-leakage paper.

Adds:
  (a) a non-sycophantic "broadcast" advisor condition: the shared state is
      always exactly the live population mean opinion (no sycophancy term,
      no per-user learning) -- isolates how much of the shared-advisor
      leakage effect is generic hub-bridging vs. specific to sycophancy.
  (b) a higher-seed-count (n=20) re-run of personalized vs shared at
      alpha=0.9, with a paired t-test (paired on seed) on between_gap.
  (c) a robustness sweep of the personalized alpha=0.6 U-shape point under
      alternate AI_LR and MU_AI values, to check it isn't a parameterization
      artifact.
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


def run_condition(mode, alpha, rng, mu_ai=0.3, ai_lr=0.05):
    """mode: 'none' | 'shared' | 'personalized' | 'broadcast'"""
    opinions = rng.random(N)
    adj, community = build_block_graph(rng)

    if mode == "shared":
        advisor_state = np.array([0.5])
    elif mode == "personalized":
        advisor_state = rng.random(N) * 0 + 0.5
    else:
        advisor_state = None

    for _ in range(STEPS):
        i = rng.integers(N)
        use_ai = mode != "none" and rng.random() < P_AI
        if use_ai:
            if mode == "broadcast":
                response = np.mean(opinions)  # non-sycophantic: live population mean
            else:
                a_idx = 0 if mode == "shared" else i
                a = advisor_state[a_idx]
                response = (1 - alpha) * a + alpha * opinions[i]
            if abs(response - opinions[i]) < EPS:
                opinions[i] += mu_ai * (response - opinions[i])
            if mode in ("shared", "personalized"):
                advisor_state[a_idx] += ai_lr * (opinions[i] - a)
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
    return {"within_var": float(within_var), "between_gap": float(between_gap)}


def run_many(mode, alpha, n_seeds, mu_ai=0.3, ai_lr=0.05, seed_offset=0):
    gaps, wvars = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(RNG_SEED * 1000 + seed_offset + s)
        r = run_condition(mode, alpha, rng, mu_ai=mu_ai, ai_lr=ai_lr)
        gaps.append(r["between_gap"])
        wvars.append(r["within_var"])
    return np.array(gaps), np.array(wvars)


def main():
    t0 = time.time()
    out = {}

    # (a) broadcast (non-sycophantic hub) baseline, 5 seeds, matched to original
    baseline_gaps, _ = run_many("none", 0.0, 5)
    broadcast_gaps, broadcast_wvars = run_many("broadcast", 0.0, 5)
    out["baseline"] = {"gap_mean": float(baseline_gaps.mean()), "gap_std": float(baseline_gaps.std())}
    out["broadcast"] = {
        "gap_mean": float(broadcast_gaps.mean()), "gap_std": float(broadcast_gaps.std()),
        "within_var_mean": float(broadcast_wvars.mean()),
        "leakage_index": float(baseline_gaps.mean() - broadcast_gaps.mean()),
    }
    print("broadcast:", out["broadcast"])

    # (b) higher-seed re-run + paired t-test, personalized vs shared @ alpha=0.9
    n_seeds_b = 20
    pers_gaps, _ = run_many("personalized", 0.9, n_seeds_b)
    shared_gaps, _ = run_many("shared", 0.9, n_seeds_b)
    t_stat, p_val = stats.ttest_rel(pers_gaps, shared_gaps)
    out["alpha0.9_20seed"] = {
        "personalized_gap_mean": float(pers_gaps.mean()), "personalized_gap_std": float(pers_gaps.std()),
        "shared_gap_mean": float(shared_gaps.mean()), "shared_gap_std": float(shared_gaps.std()),
        "paired_t_stat": float(t_stat), "paired_p_value": float(p_val), "n_seeds": n_seeds_b,
    }
    print("alpha0.9 20-seed paired test:", out["alpha0.9_20seed"])

    # also re-check shared alpha=0.3 (flagged as noisy) at 20 seeds vs baseline
    baseline_gaps_20, _ = run_many("none", 0.0, n_seeds_b)
    shared_a03_gaps, _ = run_many("shared", 0.3, n_seeds_b)
    t2, p2 = stats.ttest_rel(baseline_gaps_20, shared_a03_gaps)
    out["shared_a0.3_20seed"] = {
        "gap_mean": float(shared_a03_gaps.mean()), "gap_std": float(shared_a03_gaps.std()),
        "leakage_index": float(baseline_gaps_20.mean() - shared_a03_gaps.mean()),
        "paired_t_vs_baseline": float(t2), "paired_p_vs_baseline": float(p2),
    }
    print("shared a=0.3 20-seed:", out["shared_a0.3_20seed"])

    # (c) U-shape robustness: personalized alpha in {0,0.3,0.6,0.9,1.0} under
    # alternate AI_LR / MU_AI combos (5 seeds each, matching original protocol)
    combos = [("orig", 0.05, 0.3), ("low_lr", 0.02, 0.3), ("high_lr", 0.15, 0.3),
              ("low_mu", 0.05, 0.15), ("high_mu", 0.05, 0.5)]
    alphas = [0.0, 0.3, 0.6, 0.9, 1.0]
    ushape = {}
    for name, lr, mu in combos:
        row = {}
        for al in alphas:
            g, _ = run_many("personalized", al, 5, mu_ai=mu, ai_lr=lr)
            row[str(al)] = float(g.mean())
        ushape[name] = row
        print(name, row)
    out["ushape_robustness"] = ushape

    out["wall_time_s"] = time.time() - t0
    with open("results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
