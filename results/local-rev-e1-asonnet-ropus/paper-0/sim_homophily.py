"""Follow-up to reviewer Q1: does the policy ranking return under mild homophily in the
following graph, instead of the fully random following graph tested in sim_network.py?
Each agent follows the `degree` agents whose INITIAL opinion is closest to its own,
rather than a uniformly random set."""
import numpy as np
import json
import time

RNG_SEED_BASE = 12345


def run_homophily_sim(policy="engagement", N=300, T=150, k=5, degree=20,
                       confidence_bound=0.5, mu=0.15, backfire_prob=0.15,
                       backfire_rate=0.10, engagement_sigma=0.4, base_engage_rate=0.9,
                       post_frac=0.05, epsilon=0.3, seed=0, history_cap=30):
    rng = np.random.default_rng(RNG_SEED_BASE + seed)
    opinions = rng.uniform(-1, 1, size=N)
    init_opinions = opinions.copy()

    # homophilous following: each agent follows the `degree` closest-opinion others
    init_dist = np.abs(init_opinions[:, None] - init_opinions[None, :])
    np.fill_diagonal(init_dist, np.inf)
    following = np.argsort(init_dist, axis=1)[:, :degree]

    local_pool = [list(init_opinions[following[i]]) for i in range(N)]

    var_hist, ext_hist = [], []
    for t in range(T):
        recs = np.zeros((N, k))
        for i in range(N):
            pool_i = np.array(local_pool[i][-history_cap:])
            if len(pool_i) < k:
                pool_i = np.concatenate([pool_i, rng.uniform(-1, 1, size=k - len(pool_i))])
            d = np.abs(opinions[i] - pool_i)
            if policy == "random":
                idx = rng.integers(0, len(pool_i), size=k)
            elif policy == "engagement":
                idx = np.argsort(d)[:k]
            elif policy == "diversity":
                close_k = max(1, int(round(k * (1 - epsilon))))
                div_k = k - close_k
                close_idx = np.argsort(d)[:close_k]
                idx = np.concatenate([close_idx, rng.integers(0, len(pool_i), size=div_k)]) if div_k > 0 else close_idx
            recs[i] = pool_i[idx.astype(int)]

        rec_dist = np.abs(opinions[:, None] - recs)
        engage_prob = base_engage_rate * np.exp(-(rec_dist ** 2) / (2 * engagement_sigma ** 2))
        engaged = rng.random(size=rec_dist.shape) < engage_prob

        delta = np.zeros(N)
        for j in range(k):
            eng_j = engaged[:, j]
            d = recs[:, j] - opinions
            within = np.abs(d) <= confidence_bound
            assim = eng_j & within
            delta[assim] += mu * d[assim]
            outside = eng_j & (~within)
            backfire_mask = outside & (rng.random(N) < backfire_prob)
            delta[backfire_mask] -= backfire_rate * d[backfire_mask]
        opinions = np.clip(opinions + delta, -1, 1)

        posters = np.where(rng.random(N) < post_frac)[0]
        for p in posters:
            followers = np.where((following == p).any(axis=1))[0]
            for f in followers:
                local_pool[f].append(opinions[p])

        var_hist.append(float(np.var(opinions)))
        ext_hist.append(float(np.mean(np.abs(opinions) > 0.7)))

    tail = int(T * 0.2)
    return float(np.mean(var_hist[-tail:])), float(np.mean(ext_hist[-tail:]))


def main():
    t0 = time.time()
    results = {}
    for pol in ["random", "engagement", "diversity"]:
        vs, es = [], []
        for s in range(4):
            v, e = run_homophily_sim(policy=pol, seed=s)
            vs.append(v); es.append(e)
        results[pol] = {"variance_avg": float(np.mean(vs)), "variance_std": float(np.std(vs)),
                         "extremism_avg": float(np.mean(es)), "extremism_std": float(np.std(es))}
        print(f"{pol}: done ({time.time()-t0:.1f}s)")
    results["total_runtime_sec"] = time.time() - t0
    with open("results_homophily.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
