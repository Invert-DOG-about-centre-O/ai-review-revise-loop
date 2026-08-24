import numpy as np
import json
import itertools

N = 200
T = 150
K = 8
M = 30
MU = 0.35
EPS_DEFAULT = 0.35

def run_sim(policy, eps, seed, ext_weight=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=N)
    var_traj = []
    ext_traj = []
    echo_sum = 0.0
    echo_n = 0
    for t in range(T):
        new_x = x.copy()
        for i in range(N):
            pool = rng.choice([j for j in range(N) if j != i], size=M, replace=False)
            pool_x = x[pool]
            if policy == "random":
                shown = rng.choice(pool, size=K, replace=False)
            elif policy == "similarity":
                dist = np.abs(pool_x - x[i])
                shown = pool[np.argsort(dist)[:K]]
            elif policy == "engagement":
                dist = np.abs(pool_x - x[i])
                agree = 1 - dist
                score = (1 - ext_weight) * agree + ext_weight * np.abs(pool_x)
                shown = pool[np.argsort(-score)[:K]]
            elif policy == "bridging":
                dist = np.abs(pool_x - x[i])
                close = pool[np.argsort(dist)[:K // 2]]
                far = pool[np.argsort(-dist)[:K // 2]]
                shown = np.concatenate([close, far])
            shown_x = x[shown]
            echo_sum += np.sum(np.abs(shown_x - x[i]))
            echo_n += len(shown_x)
            persuasive = shown_x[np.abs(shown_x - x[i]) <= eps]
            if len(persuasive) > 0:
                target = np.mean(persuasive)
                new_x[i] = x[i] + MU * (target - x[i])
        x = new_x
        var_traj.append(np.var(x))
        ext_traj.append(np.mean(np.abs(x)))
    clusters = count_clusters(x)
    return {
        "final_variance": var_traj[-1],
        "final_extremity": ext_traj[-1],
        "echo_chamber_index": echo_sum / echo_n,
        "clusters": clusters,
        "var_traj": var_traj,
        "ext_traj": ext_traj,
    }

def count_clusters(x, gap=0.08):
    xs = np.sort(x)
    if len(xs) == 0:
        return 0
    n_clusters = 1
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > gap:
            n_clusters += 1
    return n_clusters

def holm_bonferroni(named_pvals):
    """named_pvals: list of (name, p). Returns dict name -> (p, adj_alpha_rank, reject_at_0.05)."""
    order = sorted(named_pvals, key=lambda kv: kv[1])
    m = len(order)
    out = {}
    reject_all_below = True
    for rank, (name, p) in enumerate(order, start=1):
        thresh = 0.05 / (m - rank + 1)
        reject = reject_all_below and (p < thresh)
        if not reject:
            reject_all_below = False
        out[name] = {"p": p, "holm_threshold": thresh, "significant_holm": reject}
    return out

def permutation_test(a, b, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:na]) - np.mean(pooled[na:])
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, p

if __name__ == "__main__":
    policies = ["random", "similarity", "engagement", "bridging"]
    seeds = list(range(10))

    # main experiment
    main_results = {p: [] for p in policies}
    for p in policies:
        for s in seeds:
            main_results[p].append(run_sim(p, EPS_DEFAULT, s))

    print("=== Main experiment (eps=0.35, 10 seeds) ===")
    summary = {}
    for p in policies:
        var = np.array([r["final_variance"] for r in main_results[p]])
        ext = np.array([r["final_extremity"] for r in main_results[p]])
        echo = np.array([r["echo_chamber_index"] for r in main_results[p]])
        clus = np.array([r["clusters"] for r in main_results[p]])
        summary[p] = {
            "var_mean": var.mean(), "var_std": var.std(),
            "ext_mean": ext.mean(), "ext_std": ext.std(),
            "echo_mean": echo.mean(), "echo_std": echo.std(),
            "clus_mean": clus.mean(), "clus_std": clus.std(),
            "var_raw": var.tolist(),
            "ext_raw": ext.tolist(),
            "echo_raw": echo.tolist(),
        }
        print(p, "var", var.mean(), var.std(), "ext", ext.mean(), ext.std(),
              "echo", echo.mean(), echo.std(), "clus", clus.mean(), clus.std())

    # significance tests on key comparisons
    print("\n=== Permutation tests (final variance) ===")
    sig_results = {}
    pairs = [("engagement", "similarity"), ("similarity", "random"),
             ("bridging", "similarity"), ("engagement", "bridging")]
    for a, b in pairs:
        va = np.array(summary[a]["var_raw"])
        vb = np.array(summary[b]["var_raw"])
        obs, p = permutation_test(va, vb, n_perm=10000, seed=42)
        sig_results[f"{a}_vs_{b}"] = {"diff": obs, "p": p}
        print(f"{a} vs {b}: diff={obs:.4f}, p={p:.4f}")

    print("\n=== Permutation tests (extremity and echo-chamber index) ===")
    for a, b in pairs:
        for metric, key in [("extremity", "ext_raw"), ("echo_chamber_index", "echo_raw")]:
            va = np.array(summary[a][key])
            vb = np.array(summary[b][key])
            obs, p = permutation_test(va, vb, n_perm=10000, seed=42)
            sig_results[f"{a}_vs_{b}_{metric}"] = {"diff": obs, "p": p}
            print(f"{a} vs {b} ({metric}): diff={obs:.4f}, p={p:.4f}")

    print("\n=== Holm-Bonferroni correction (main-experiment family, m=12) ===")
    holm_main = holm_bonferroni([(k, v["p"]) for k, v in sig_results.items()])
    for k, v in holm_main.items():
        print(f"{k}: p={v['p']:.4f}, holm_threshold={v['holm_threshold']:.4f}, significant_holm={v['significant_holm']}")

    # extremity-weight sweep
    SWEEP_SEEDS = 30
    print(f"\n=== Extremity weight sweep (engagement policy, eps=0.35, {SWEEP_SEEDS} seeds) ===")
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    sweep_results = {}
    sweep_raw = {}
    for w in weights:
        vs = []
        for s in range(SWEEP_SEEDS):
            r = run_sim("engagement", EPS_DEFAULT, s, ext_weight=w)
            vs.append(r["final_variance"])
        vs = np.array(vs)
        sweep_results[w] = {"mean": vs.mean(), "std": vs.std()}
        sweep_raw[w] = vs.tolist()
        print(f"weight={w}: var={vs.mean():.4f} +/- {vs.std():.4f}")

    print("\n=== Permutation tests on weight sweep (adjacent steps + extremes) ===")
    for w1, w2 in zip(weights[:-1], weights[1:]):
        va = np.array(sweep_raw[w1])
        vb = np.array(sweep_raw[w2])
        obs, p = permutation_test(va, vb, n_perm=10000, seed=42)
        sweep_results[f"sig_{w1}_vs_{w2}"] = {"diff": obs, "p": p}
        print(f"w={w1} vs w={w2}: diff={obs:.4f}, p={p:.4f}")
    va = np.array(sweep_raw[0.0])
    vb = np.array(sweep_raw[0.5])
    obs, p = permutation_test(va, vb, n_perm=10000, seed=42)
    sweep_results["sig_0.0_vs_0.5"] = {"diff": obs, "p": p}
    print(f"w=0.0 vs w=0.5 (extremes): diff={obs:.4f}, p={p:.4f}")

    # internal-consistency check: w=0 engagement (30 seeds) vs main-experiment
    # similarity policy (10 seeds) should be statistically indistinguishable,
    # since w=0 collapses the engagement score to pure similarity ranking.
    va = np.array(sweep_raw[0.0])
    vb = np.array(summary["similarity"]["var_raw"])
    obs, p = permutation_test(va, vb, n_perm=10000, seed=42)
    sweep_results["sig_w0_vs_main_similarity"] = {"diff": obs, "p": p}
    print(f"w=0.0 (sweep) vs similarity (main, 10 seeds): diff={obs:.4f}, p={p:.4f}")

    print("\n=== Holm-Bonferroni correction (weight-sweep family, m=8) ===")
    sweep_pvals = [(k, v["p"]) for k, v in sweep_results.items() if isinstance(k, str) and k.startswith("sig_")]
    holm_sweep = holm_bonferroni(sweep_pvals)
    for k, v in holm_sweep.items():
        print(f"{k}: p={v['p']:.4f}, holm_threshold={v['holm_threshold']:.4f}, significant_holm={v['significant_holm']}")
    sweep_results["holm_bonferroni"] = holm_sweep

    # ablation over epsilon
    print("\n=== Ablation over epsilon (5 seeds) ===")
    eps_values = [0.2, 0.35, 0.5]
    ablation_results = {}
    for eps in eps_values:
        ablation_results[eps] = {}
        for p in policies:
            vs = []
            for s in range(5):
                r = run_sim(p, eps, s)
                vs.append(r["final_variance"])
            vs = np.array(vs)
            ablation_results[eps][p] = {"mean": vs.mean(), "std": vs.std()}
            print(f"eps={eps} {p}: var={vs.mean():.4f} +/- {vs.std():.4f}")

    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)
    sig_results["holm_bonferroni"] = holm_main
    with open("significance_results.json", "w") as f:
        json.dump(sig_results, f, indent=2)
    with open("extremity_weight_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    with open("ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
