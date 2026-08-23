"""
Simulation of algorithmic exposure recommenders in a bounded-confidence
opinion dynamics model (Hegselmann-Krause style), to study how the
recommender's exposure policy shapes polarization and echo-chamber
formation.

Agents each hold a scalar opinion in [-1, 1]. Each round, a recommender
selects a small subset of "candidate posts" (other agents' current
opinions) to show each agent, out of a larger random candidate pool
(mimicking a feed that samples from a broader universe of content but
ranks/filters it). The agent then updates its opinion via a bounded-
confidence rule using only the opinions it was actually shown, weighted
by inverse opinion distance (agents move more toward opinions that are
close but distinct, and ignore opinions that are farther than a
confidence threshold epsilon).

Four recommenders are compared:
  - random:      shows a uniformly random sample from the candidate pool.
  - similarity:  shows the K closest opinions to the agent's own (filter
                 bubble / homophily-maximizing).
  - engagement:  shows opinions that are close-ish AND extreme, modeling
                 engagement-optimized ranking that favors sensational
                 content among agreeable content.
  - bridging:    shows a mix: half closest opinions, half opinions from
                 the opposite side of the spectrum (diversity injection).

Metrics tracked over time: opinion variance (polarization), mean
absolute opinion (extremity), and final number of opinion clusters
(via simple 1D clustering with a distance threshold).
"""
import json
import time
import numpy as np

RNG_SEED_BASE = 12345


def cluster_count(opinions, gap=0.08):
    """Count clusters in sorted 1D opinions by gap threshold."""
    s = np.sort(opinions)
    if len(s) == 0:
        return 0
    gaps = np.diff(s)
    return int(1 + np.sum(gaps > gap))


def select_candidates(rng, opinions, agent_idx, pool_size):
    """Draw a random candidate pool of other agents' opinions for exposure."""
    n = len(opinions)
    others = np.delete(np.arange(n), agent_idx)
    pool_idx = rng.choice(others, size=min(pool_size, len(others)), replace=False)
    return pool_idx


def recommend(policy, rng, opinions, agent_idx, pool_idx, k):
    """Return indices (subset of pool_idx) of length k to show agent_idx."""
    own = opinions[agent_idx]
    pool_ops = opinions[pool_idx]

    if policy == "random":
        chosen = rng.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)
        return chosen

    if policy == "similarity":
        dist = np.abs(pool_ops - own)
        order = np.argsort(dist)
        return pool_idx[order[:k]]

    if policy == "engagement":
        # Engagement score rewards closeness (agreement) but also extremity
        # of the candidate's opinion (sensational/attention-grabbing content).
        dist = np.abs(pool_ops - own)
        agreement_score = 1.0 - dist  # in roughly [0,2] -> higher = closer
        extremity_score = np.abs(pool_ops)  # in [0,1]
        score = 0.7 * agreement_score + 0.3 * extremity_score
        order = np.argsort(-score)
        return pool_idx[order[:k]]

    if policy == "bridging":
        dist = np.abs(pool_ops - own)
        order_close = np.argsort(dist)
        k_close = k // 2
        k_far = k - k_close
        close_idx = pool_idx[order_close[:k_close]]
        # opposite side: opinions with sign different from own (or farthest if own~0)
        order_far = np.argsort(-dist)
        far_idx = pool_idx[order_far[:k_far]]
        return np.concatenate([close_idx, far_idx])

    raise ValueError(policy)


def run_simulation(policy, n_agents=200, n_rounds=150, pool_size=30, k_shown=8,
                    epsilon=0.35, mu=0.35, seed=0):
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(-1, 1, size=n_agents)

    variance_traj = np.zeros(n_rounds + 1)
    extremity_traj = np.zeros(n_rounds + 1)
    echo_traj = np.zeros(n_rounds + 1)
    variance_traj[0] = np.var(opinions)
    extremity_traj[0] = np.mean(np.abs(opinions))
    echo_traj[0] = np.nan

    for t in range(1, n_rounds + 1):
        new_opinions = opinions.copy()
        order = rng.permutation(n_agents)
        exposure_dists = []
        for i in order:
            pool_idx = select_candidates(rng, opinions, i, pool_size)
            shown_idx = recommend(policy, rng, opinions, i, pool_idx, k_shown)
            shown_ops = opinions[shown_idx]
            dist = np.abs(shown_ops - opinions[i])
            exposure_dists.append(dist.mean())
            within = dist <= epsilon
            if not np.any(within):
                continue
            influencers = shown_ops[within]
            # bounded-confidence update: move toward mean of accepted influencers
            target = np.mean(influencers)
            new_opinions[i] = opinions[i] + mu * (target - opinions[i])
        new_opinions = np.clip(new_opinions, -1, 1)
        opinions = new_opinions
        variance_traj[t] = np.var(opinions)
        extremity_traj[t] = np.mean(np.abs(opinions))
        # echo-chamber index: mean |shown - own| averaged over agents this round
        # (lower = agents are shown content closer to their own opinion = more insular)
        echo_traj[t] = float(np.mean(exposure_dists))

    n_clusters = cluster_count(opinions)
    return {
        "policy": policy,
        "seed": seed,
        "epsilon": epsilon,
        "variance_traj": variance_traj.tolist(),
        "extremity_traj": extremity_traj.tolist(),
        "echo_traj": echo_traj.tolist(),
        "final_variance": float(variance_traj[-1]),
        "final_extremity": float(extremity_traj[-1]),
        "mean_echo_index": float(np.nanmean(echo_traj)),
        "n_clusters": n_clusters,
        "final_opinions": opinions.tolist(),
    }


def summarize(results, policies, group_key="policy"):
    lines = []
    header = f"{'policy':<12}{'final_var':>16}{'final_extremity':>20}{'echo_idx':>14}{'n_clusters':>14}"
    lines.append(header)
    print(header)
    for policy in policies:
        rows = [r for r in results if r[group_key] == policy]
        fv = np.array([r["final_variance"] for r in rows])
        fe = np.array([r["final_extremity"] for r in rows])
        ei = np.array([r["mean_echo_index"] for r in rows])
        nc = np.array([r["n_clusters"] for r in rows])
        line = (f"{policy:<12}{fv.mean():>8.4f}+/-{fv.std():<7.4f}"
                f"{fe.mean():>10.4f}+/-{fe.std():<7.4f}"
                f"{ei.mean():>6.3f}+/-{ei.std():<5.3f}"
                f"{nc.mean():>7.2f}+/-{nc.std():<6.2f}")
        print(line)
        lines.append(line)
    return "\n".join(lines)


def main():
    t0 = time.time()
    policies = ["random", "similarity", "engagement", "bridging"]
    n_seeds = 10
    results = []
    for policy in policies:
        for s in range(n_seeds):
            r = run_simulation(policy, seed=RNG_SEED_BASE + s)
            results.append(r)
        print(f"finished policy={policy} ({time.time()-t0:.1f}s elapsed)")

    with open("results.json", "w") as f:
        json.dump(results, f)

    print("\n=== Main experiment: recommender policy comparison (epsilon=0.35, 10 seeds) ===")
    summary_main = summarize(results, policies)

    # --- Ablation: sensitivity to bounded-confidence threshold epsilon ---
    print("\n=== Ablation: varying confidence threshold epsilon (5 seeds each) ===")
    ablation_results = []
    n_seeds_abl = 5
    epsilons = [0.2, 0.35, 0.5]
    for eps in epsilons:
        for policy in policies:
            for s in range(n_seeds_abl):
                r = run_simulation(policy, epsilon=eps, seed=RNG_SEED_BASE + 100 + s)
                r["group"] = f"eps={eps}/{policy}"
                ablation_results.append(r)
        print(f"finished epsilon={eps} ({time.time()-t0:.1f}s elapsed)")

    with open("ablation_results.json", "w") as f:
        json.dump(ablation_results, f)

    ablation_summary_lines = []
    for eps in epsilons:
        print(f"\n-- epsilon={eps} --")
        groups = [f"eps={eps}/{p}" for p in policies]
        s = summarize(ablation_results, groups, group_key="group")
        ablation_summary_lines.append(f"epsilon={eps}\n{s}")

    # --- Plot: mean variance trajectory per policy (main experiment) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"random": "#888888", "similarity": "#1f77b4",
              "engagement": "#d62728", "bridging": "#2ca02c"}
    for policy in policies:
        rows = [r for r in results if r["policy"] == policy]
        traj = np.array([r["variance_traj"] for r in rows])
        mean_traj = traj.mean(axis=0)
        axes[0].plot(mean_traj, label=policy, color=colors[policy])

        ext_traj = np.array([r["extremity_traj"] for r in rows])
        mean_ext = ext_traj.mean(axis=0)
        axes[1].plot(mean_ext, label=policy, color=colors[policy])

    axes[0].set_xlabel("round")
    axes[0].set_ylabel("opinion variance (polarization)")
    axes[0].set_title("Polarization over time")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("round")
    axes[1].set_ylabel("mean |opinion| (extremity)")
    axes[1].set_title("Extremity over time")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("trajectories.png", dpi=150)
    print("\nsaved plot to trajectories.png")

    with open("summary.txt", "w") as f:
        f.write("=== Main experiment (epsilon=0.35, 10 seeds) ===\n")
        f.write(summary_main + "\n\n")
        f.write("=== Ablation over epsilon (5 seeds each) ===\n")
        f.write("\n\n".join(ablation_summary_lines) + "\n")

    print(f"\ntotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
