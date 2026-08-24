"""
Revision add-on experiments, run after experiment.py, addressing round-1 review
questions: (1) sensitivity to non-Dirichlet (bimodal / power-law) cluster-mass
shapes, and (2) how entailment-clustering error (label noise) interacts with
the pure sampling bias quantified in experiment.py. Appends to results.json
under new top-level keys; does not modify the original experiment.
"""
import hashlib
import json
import time
import numpy as np
from scipy import stats

def stable_seed(*parts) -> int:
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def make_p(rng, C, shape):
    if shape == "bimodal":
        # one dominant cluster (0.55-0.85 mass), rest split unevenly via Dirichlet
        dom = rng.uniform(0.55, 0.85)
        rest = rng.dirichlet(0.5 * np.ones(C - 1)) * (1 - dom)
        p = np.concatenate([[dom], rest])
        rng.shuffle(p)
        return p
    elif shape == "powerlaw":
        ranks = np.arange(1, C + 1)
        w = 1.0 / ranks ** 1.3
        p = w / w.sum()
        rng.shuffle(p)
        return p
    else:
        raise ValueError(shape)

def simulate_cell_shape(n_queries, K, c_lo, c_hi, shape, seed):
    rng = np.random.default_rng(seed)
    H_true = np.empty(n_queries); H_plugin = np.empty(n_queries); H_mm = np.empty(n_queries)
    top1_conf = np.empty(n_queries)
    for i in range(n_queries):
        C = rng.integers(c_lo, c_hi + 1)
        p = make_p(rng, C, shape)
        H_true[i] = -np.sum(p * np.log(np.clip(p, 1e-300, 1)))
        samples = rng.choice(C, size=K, p=p)
        counts = np.bincount(samples, minlength=C)
        obs = counts[counts > 0]
        q = obs / K
        H_plugin[i] = -np.sum(q * np.log(q))
        H_mm[i] = H_plugin[i] + (len(obs) - 1) / (2.0 * K)
        top1_conf[i] = obs.max() / K
    return dict(H_true=H_true, H_plugin=H_plugin, H_mm=H_mm, top1_conf=top1_conf)

def auroc(scores_high_means_hard, labels_hard):
    order = np.argsort(scores_high_means_hard)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    n1 = labels_hard.sum(); n0 = len(labels_hard) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    sum_ranks_pos = ranks[labels_hard == 1].sum()
    return (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)

def simulate_cell_noisy(n_queries, K, alpha, c_lo, c_hi, epsilon, seed):
    """Like experiment.py's simulate_cell, but each of the K sampled *labels*
    is independently mislabeled to a uniformly random *other* cluster with
    prob epsilon (a crude model of entailment-clustering error). H_true is
    still computed on the clean underlying distribution p, so this measures
    total bias (sampling + clustering-noise) relative to the ground truth a
    real pipeline would want to recover."""
    rng = np.random.default_rng(seed)
    H_true = np.empty(n_queries); H_plugin = np.empty(n_queries); H_mm = np.empty(n_queries)
    for i in range(n_queries):
        C = rng.integers(c_lo, c_hi + 1)
        p = rng.dirichlet(alpha * np.ones(C))
        H_true[i] = -np.sum(p * np.log(np.clip(p, 1e-300, 1)))
        samples = rng.choice(C, size=K, p=p)
        if epsilon > 0:
            flip = rng.random(K) < epsilon
            n_flip = flip.sum()
            if n_flip > 0 and C > 1:
                # relabel flipped samples to a uniformly random different cluster
                new_labels = rng.integers(0, C - 1, size=n_flip)
                orig = samples[flip]
                new_labels = new_labels + (new_labels >= orig)  # skip own label
                samples = samples.copy()
                samples[flip] = new_labels
        counts = np.bincount(samples, minlength=C)
        obs = counts[counts > 0]
        q = obs / K
        H_plugin[i] = -np.sum(q * np.log(q))
        H_mm[i] = H_plugin[i] + (len(obs) - 1) / (2.0 * K)
    return dict(H_true=H_true, H_plugin=H_plugin, H_mm=H_mm)

def main():
    t0 = time.time()
    N_QUERIES = 2000
    N_REPLICATES = 30
    C_RANGE = (2, 10)

    # ---- (Q2) Non-Dirichlet cluster-mass shapes ----
    shape_results = {}
    for shape in ["bimodal", "powerlaw"]:
        for K in [3, 5, 10, 20]:
            bias_plugin, bias_mm = [], []
            auc_plugin, auc_mm, auc_top1 = [], [], []
            for r in range(N_REPLICATES):
                seed = stable_seed("shape", shape, K, r)
                d = simulate_cell_shape(N_QUERIES, K, C_RANGE[0], C_RANGE[1], shape, seed)
                err_p = d["H_plugin"] - d["H_true"]; err_m = d["H_mm"] - d["H_true"]
                bias_plugin.append(err_p.mean()); bias_mm.append(err_m.mean())
                hard = (d["H_true"] > np.median(d["H_true"])).astype(int)
                auc_plugin.append(auroc(d["H_plugin"], hard))
                auc_mm.append(auroc(d["H_mm"], hard))
                auc_top1.append(auroc(-d["top1_conf"], hard))
            tt_bias = stats.ttest_rel(np.abs(bias_mm), np.abs(bias_plugin))
            tt_auc = stats.ttest_rel(auc_mm, auc_plugin)
            key = f"{shape}_K{K}"
            shape_results[key] = dict(
                bias_plugin=float(np.mean(bias_plugin)), bias_mm=float(np.mean(bias_mm)),
                bias_reduction_pct=float(100 * (1 - abs(np.mean(bias_mm)) / abs(np.mean(bias_plugin)))),
                auc_plugin=float(np.mean(auc_plugin)), auc_mm=float(np.mean(auc_mm)),
                auc_top1=float(np.mean(auc_top1)),
                paired_bias_p=float(tt_bias.pvalue), paired_auc_p=float(tt_auc.pvalue),
            )
            print(key, shape_results[key])

    # ---- (Q3) Clustering-error (label-flip) interaction with sampling bias ----
    noise_results = {}
    for epsilon in [0.0, 0.05, 0.1, 0.2]:
        for K in [5, 10]:
            bias_plugin, bias_mm = [], []
            for r in range(N_REPLICATES):
                seed = stable_seed("noise", epsilon, K, r)
                d = simulate_cell_noisy(N_QUERIES, K, 1.0, C_RANGE[0], C_RANGE[1], epsilon, seed)
                bias_plugin.append(np.mean(d["H_plugin"] - d["H_true"]))
                bias_mm.append(np.mean(d["H_mm"] - d["H_true"]))
            key = f"eps{epsilon}_K{K}"
            noise_results[key] = dict(
                bias_plugin=float(np.mean(bias_plugin)), bias_mm=float(np.mean(bias_mm)),
            )
            print(key, noise_results[key])

    elapsed = time.time() - t0
    out = dict(shape_robustness=shape_results, clustering_noise=noise_results, elapsed_seconds=elapsed)
    with open("results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal elapsed (v2): {elapsed:.1f}s")

if __name__ == "__main__":
    main()
