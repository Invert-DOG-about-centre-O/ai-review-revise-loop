"""
Agent-based simulation: sycophantic AI advisors as hidden coupling nodes
in opinion dynamics across socially disconnected communities.

Model
-----
- N human agents live in one of two communities (a stochastic block model:
  dense intra-community edges, sparse/zero inter-community edges).
- Peer-to-peer influence follows the Deffuant-Weisbuch bounded-confidence
  rule: two connected agents whose opinions differ by less than `eps` pull
  toward each other by step size `mu`.
- Some fraction of interaction steps are instead "AI consultations": a human
  states its opinion x_i to an AI advisor. The advisor holds an internal
  opinion estimate. Its *response* is r = (1 - alpha) * a + alpha * x_i,
  where alpha in [0, 1] is a sycophancy coefficient (alpha=0: the advisor
  always gives its own fixed view; alpha=1: the advisor simply mirrors the
  user). The human then updates toward r using the same bounded-confidence
  rule. The advisor also updates its OWN internal opinion a small step
  toward x_i (it "learns" from the interaction, like a personalization
  loop / recommender retrained on feedback).
- Two advisor conditions:
    * "shared": a single global advisor state a serves every human in both
      communities -> can transmit information between communities that
      have no direct social edge (a hidden coupling channel).
    * "personalized": each human has a private advisor state a_i that only
      that human ever talks to -> no cross-community channel, isolates the
      pure sycophancy/anchoring effect from the cross-community leakage
      effect.
- Baseline: no AI at all (pure peer-to-peer bounded confidence on the same
  block-structured graph).

We sweep the sycophancy coefficient alpha and the advisor condition, and
measure, after convergence:
  - within-community opinion variance (local polarization)
  - between-community mean opinion gap (community polarization)
  - a "leakage index" = reduction in the between-community gap relative to
    the no-AI baseline, attributable to the shared advisor.

Everything is seeded and deterministic given the seed; runtime target is a
few seconds per condition, a few minutes total for the full sweep.
"""
import json
import time
import numpy as np

RNG_SEED = 0
N_PER_COMMUNITY = 60
N_COMMUNITIES = 2
N = N_PER_COMMUNITY * N_COMMUNITIES
EPS = 0.15          # bounded-confidence threshold
MU = 0.3            # peer step size
MU_AI = 0.3         # human step size toward AI response
AI_LR = 0.05        # advisor internal-opinion learning rate
P_AI = 0.3          # probability a given interaction step is an AI consult
STEPS = 40000
INTRA_P = 0.15      # intra-community edge probability
INTER_P = 0.0       # inter-community edge probability (0 = fully disconnected socially)


def build_block_graph(rng):
    """Adjacency list for a two-block stochastic block model."""
    adj = [[] for _ in range(N)]
    community = np.array([0] * N_PER_COMMUNITY + [1] * N_PER_COMMUNITY)
    for i in range(N):
        for j in range(i + 1, N):
            p = INTRA_P if community[i] == community[j] else INTER_P
            if rng.random() < p:
                adj[i].append(j)
                adj[j].append(i)
    # guarantee no isolated nodes: connect any isolated node to one random
    # same-community neighbor
    for i in range(N):
        if not adj[i]:
            same = [j for j in range(N) if j != i and community[j] == community[i]]
            j = rng.choice(same)
            adj[i].append(j)
            adj[j].append(i)
    return adj, community


def run_condition(mode, alpha, rng):
    """
    mode: 'none' | 'shared' | 'personalized'
    alpha: sycophancy coefficient (ignored if mode == 'none')
    """
    opinions = rng.random(N)
    adj, community = build_block_graph(rng)

    if mode == "shared":
        advisor_state = np.array([0.5])  # single global scalar
    elif mode == "personalized":
        advisor_state = rng.random(N) * 0 + 0.5  # per-human, start neutral
    else:
        advisor_state = None

    for _ in range(STEPS):
        i = rng.integers(N)
        use_ai = mode != "none" and rng.random() < P_AI
        if use_ai:
            a_idx = 0 if mode == "shared" else i
            a = advisor_state[a_idx]
            response = (1 - alpha) * a + alpha * opinions[i]
            if abs(response - opinions[i]) < EPS:
                opinions[i] += MU_AI * (response - opinions[i])
            advisor_state[a_idx] += AI_LR * (opinions[i] - a)
        else:
            neighbors = adj[i]
            if not neighbors:
                continue
            j = neighbors[rng.integers(len(neighbors))]
            if abs(opinions[i] - opinions[j]) < EPS:
                delta = opinions[j] - opinions[i]
                opinions[i] += MU * delta
                opinions[j] -= MU * delta

    within_var = np.mean([
        np.var(opinions[community == c]) for c in range(N_COMMUNITIES)
    ])
    between_gap = abs(
        np.mean(opinions[community == 0]) - np.mean(opinions[community == 1])
    )
    global_var = np.var(opinions)
    return {
        "within_var": float(within_var),
        "between_gap": float(between_gap),
        "global_var": float(global_var),
        "final_opinions": opinions.tolist(),
        "community": community.tolist(),
    }


def main():
    t0 = time.time()
    rng_master = np.random.default_rng(RNG_SEED)
    alphas = [0.0, 0.3, 0.6, 0.9, 1.0]
    modes = ["none", "personalized", "shared"]
    n_seeds = 5

    results = {}
    for mode in modes:
        alpha_list = [None] if mode == "none" else alphas
        for alpha in alpha_list:
            key = f"{mode}" if alpha is None else f"{mode}_a{alpha}"
            runs = []
            for s in range(n_seeds):
                rng = np.random.default_rng(RNG_SEED * 1000 + s)
                r = run_condition(mode, alpha if alpha is not None else 0.0, rng)
                runs.append(r)
            agg = {
                "within_var_mean": float(np.mean([r["within_var"] for r in runs])),
                "within_var_std": float(np.std([r["within_var"] for r in runs])),
                "between_gap_mean": float(np.mean([r["between_gap"] for r in runs])),
                "between_gap_std": float(np.std([r["between_gap"] for r in runs])),
                "global_var_mean": float(np.mean([r["global_var"] for r in runs])),
                "n_seeds": n_seeds,
                "example_final_opinions": runs[0]["final_opinions"],
                "example_community": runs[0]["community"],
            }
            results[key] = agg
            print(f"{key:24s} between_gap={agg['between_gap_mean']:.4f}"
                  f" (+-{agg['between_gap_std']:.4f})"
                  f"  within_var={agg['within_var_mean']:.4f}"
                  f"  global_var={agg['global_var_mean']:.4f}")

    baseline_gap = results["none"]["between_gap_mean"]
    for key, agg in results.items():
        agg["leakage_index"] = float(baseline_gap - agg["between_gap_mean"])

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline (no AI) between-community gap: {baseline_gap:.4f}")
    print("Leakage index (reduction in gap vs baseline; positive = AI bridges communities):")
    for key, agg in results.items():
        print(f"  {key:24s} leakage_index={agg['leakage_index']:.4f}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
