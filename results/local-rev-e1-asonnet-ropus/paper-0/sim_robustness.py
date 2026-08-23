"""
Robustness experiments addressing round-1 review:
  (A) Is confidence-bound inertness an artifact of pool density collapsing to
      near-zero distance? Sweep content-generation rate (post_frac) down to 0
      (no replenishment at all) and re-run the confidence-bound ablation to see
      whether the bound ever binds, and at what post_frac the transition happens.
  (B) Is the policy ranking (engagement-max > diversity > random, in variance)
      sensitive to the assumed engagement-probability function? Re-run the
      policy comparison with a "novelty-seeking" engagement function that peaks
      at a nonzero distance instead of decaying monotonically from distance 0.
"""
import numpy as np
import json
import time

RNG_SEED_BASE = 12345


def run_simulation(
    policy="engagement",
    N=300,
    M_cap=3000,
    T=150,
    k=5,
    confidence_bound=0.5,
    mu=0.15,
    backfire_prob=0.15,
    backfire_rate=0.10,
    engagement_sigma=0.4,
    base_engage_rate=0.9,
    post_frac=0.05,
    epsilon=0.3,
    seed=0,
    engagement_mode="proximity",  # "proximity" or "novelty" (peaks at novelty_d0)
    novelty_d0=0.5,
    pool_init_size=None,
):
    rng = np.random.default_rng(RNG_SEED_BASE + seed)

    opinions = rng.uniform(-1, 1, size=N)
    init_size = min(M_cap, N * 3) if pool_init_size is None else pool_init_size
    pool = rng.uniform(-1, 1, size=init_size)

    history = {
        "variance": [],
        "extremism": [],
        "engagement_rate": [],
        "diversity_consumed": [],
        "nearest_rec_dist": [],
    }

    for t in range(T):
        dist_matrix = np.abs(opinions[:, None] - pool[None, :])  # N x M

        if policy == "random":
            idx = rng.integers(0, len(pool), size=(N, k))
        elif policy == "engagement":
            idx = np.argsort(dist_matrix, axis=1)[:, :k]
        elif policy == "diversity":
            close_k = max(1, int(round(k * (1 - epsilon))))
            div_k = k - close_k
            close_idx = np.argsort(dist_matrix, axis=1)[:, :close_k]
            if div_k > 0:
                div_idx = rng.integers(0, len(pool), size=(N, div_k))
                idx = np.concatenate([close_idx, div_idx], axis=1)
            else:
                idx = close_idx
        else:
            raise ValueError(policy)

        rec_items = pool[idx]  # N x k
        rec_dist = np.abs(opinions[:, None] - rec_items)

        if engagement_mode == "proximity":
            engage_prob = base_engage_rate * np.exp(-(rec_dist ** 2) / (2 * engagement_sigma ** 2))
        elif engagement_mode == "novelty":
            engage_prob = base_engage_rate * np.exp(-((rec_dist - novelty_d0) ** 2) / (2 * engagement_sigma ** 2))
        else:
            raise ValueError(engagement_mode)
        engaged = rng.random(size=rec_dist.shape) < engage_prob  # N x k boolean

        engagement_rate = engaged.sum() / (N * k)
        consumed_dists = rec_dist[engaged]
        diversity_consumed = consumed_dists.mean() if consumed_dists.size > 0 else 0.0

        delta = np.zeros(N)
        for j in range(k):
            eng_j = engaged[:, j]
            item_op = rec_items[:, j]
            d = item_op - opinions
            within = np.abs(d) <= confidence_bound
            assimilate_mask = eng_j & within
            delta[assimilate_mask] += mu * d[assimilate_mask]

            outside = eng_j & (~within)
            backfire_roll = rng.random(N) < backfire_prob
            backfire_mask = outside & backfire_roll
            delta[backfire_mask] -= backfire_rate * d[backfire_mask]

        opinions = np.clip(opinions + delta, -1, 1)

        posters = rng.random(N) < post_frac
        new_items = opinions[posters]
        if new_items.size > 0:
            pool = np.concatenate([pool, new_items])
            if len(pool) > M_cap:
                pool = pool[-M_cap:]

        history["variance"].append(float(np.var(opinions)))
        history["extremism"].append(float(np.mean(np.abs(opinions) > 0.7)))
        history["engagement_rate"].append(float(engagement_rate))
        history["diversity_consumed"].append(float(diversity_consumed))
        history["nearest_rec_dist"].append(float(np.min(dist_matrix, axis=1).mean()))

    return history, opinions


def summarize(history, last_frac=0.2):
    n = len(history["variance"])
    tail = max(1, int(n * last_frac))
    out = {}
    for key in history:
        arr = np.array(history[key][-tail:])
        out[key + "_final_mean"] = float(arr.mean())
    return out


def main():
    t0 = time.time()
    results = {}

    # --- Experiment A: does the confidence-bound bind at lower content-generation rates? ---
    # post_frac controls how fast the pool re-homogenizes around current opinions.
    # Sweep it down to 0 (no replenishment -> pool stays at its original diverse
    # random draw forever) and re-run the cb ablation at each level.
    post_fracs = [0.05, 0.01, 0.002, 0.0]
    cb_values = [0.2, 0.5, 0.8]
    n_seeds_a = 4
    density_results = {}
    for pf in post_fracs:
        cb_block = {}
        for cb in cb_values:
            seed_summaries = []
            nearest = []
            for s in range(n_seeds_a):
                hist, _ = run_simulation(policy="engagement", confidence_bound=cb,
                                          post_frac=pf, seed=s)
                seed_summaries.append(summarize(hist))
                nearest.append(summarize(hist)["nearest_rec_dist_final_mean"])
            vals_var = [x["variance_final_mean"] for x in seed_summaries]
            vals_ext = [x["extremism_final_mean"] for x in seed_summaries]
            cb_block[str(cb)] = {
                "variance_avg": float(np.mean(vals_var)),
                "variance_std": float(np.std(vals_var)),
                "extremism_avg": float(np.mean(vals_ext)),
                "nearest_dist_avg": float(np.mean(nearest)),
            }
        density_results[str(pf)] = cb_block
        print(f"[density sweep] post_frac={pf}: done ({time.time()-t0:.1f}s elapsed)")
    results["confidence_bound_density_sweep"] = density_results

    # --- Experiment B: policy ranking under a novelty-seeking engagement function ---
    # (engagement peaks at distance=novelty_d0 rather than at distance=0), while the
    # recommender's policies (random/engagement/diversity) are UNCHANGED -- this tests
    # whether the recommender's naive proximity-based "engagement-maximizing" policy
    # still produces the most polarization when the true engagement function it is
    # (mis)optimizing for is not actually monotone-decreasing in distance.
    policies = ["random", "engagement", "diversity"]
    n_seeds_b = 5
    novelty_results = {}
    for pol in policies:
        seed_summaries = []
        for s in range(n_seeds_b):
            hist, _ = run_simulation(policy=pol, seed=s, engagement_mode="novelty", novelty_d0=0.5)
            seed_summaries.append(summarize(hist))
        agg = {}
        for key in ["variance_final_mean", "extremism_final_mean", "engagement_rate_final_mean"]:
            vals = [x[key] for x in seed_summaries]
            agg[key + "_avg"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        novelty_results[pol] = agg
        print(f"[novelty engagement] {pol}: done ({time.time()-t0:.1f}s elapsed)")
    results["novelty_engagement_policy_comparison"] = novelty_results

    results["total_runtime_sec"] = time.time() - t0
    with open("results_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Total runtime: {time.time()-t0:.1f}s")
    print("Saved results_robustness.json")


if __name__ == "__main__":
    main()
