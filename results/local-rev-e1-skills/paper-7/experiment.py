"""
Simulation study of finite-sample semantic-entropy estimation for LLM
uncertainty quantification (UQ).

Why simulation rather than a real LLM: this environment has no working TLS
path to model hubs (huggingface.co presents a malformed CA certificate that
Python's ssl module correctly rejects), so no pretrained weights can be
downloaded. Instead we simulate the *output* of the semantic-clustering step
that real semantic-entropy pipelines (Kuhn et al. 2023/2024; Farquhar et al.,
Nature 2024) already perform: each "query" has a true, unknown categorical
distribution over C semantic-equivalence clusters, and we observe K samples
from it (K = number of LLM generations, the expensive resource in practice).
This isolates the statistical estimation problem (entropy estimation from
K << population samples) from the upstream entailment-clustering problem,
which is itself the subject of a large, separate literature (see v1.md
Related Work).

Deterministic: every seed is derived via hashlib.sha256 (never Python's
built-in hash(), which is process-randomized).
"""
import hashlib
import json
import time
import numpy as np
from scipy import stats

def stable_seed(*parts) -> int:
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def simulate_cell(n_queries, K, alpha, c_lo, c_hi, seed):
    """Simulate n_queries independent (true_dist, K samples) pairs.
    Returns per-query: H_true, H_plugin, H_mm, top1_conf, C_true, C_obs (all natural log / nats).
    """
    rng = np.random.default_rng(seed)
    H_true = np.empty(n_queries)
    H_plugin = np.empty(n_queries)
    H_mm = np.empty(n_queries)
    top1_conf = np.empty(n_queries)
    C_true = np.empty(n_queries, dtype=int)
    C_obs = np.empty(n_queries, dtype=int)

    for i in range(n_queries):
        C = rng.integers(c_lo, c_hi + 1)
        p = rng.dirichlet(alpha * np.ones(C))
        H_true[i] = -np.sum(p * np.log(np.clip(p, 1e-300, 1)))
        C_true[i] = C

        samples = rng.choice(C, size=K, p=p)
        counts = np.bincount(samples, minlength=C)
        obs = counts[counts > 0]
        C_obs[i] = len(obs)
        q = obs / K
        H_plugin[i] = -np.sum(q * np.log(q))
        H_mm[i] = H_plugin[i] + (len(obs) - 1) / (2.0 * K)
        top1_conf[i] = obs.max() / K

    return dict(H_true=H_true, H_plugin=H_plugin, H_mm=H_mm,
                top1_conf=top1_conf, C_true=C_true, C_obs=C_obs)

def auroc(scores_high_means_hard, labels_hard):
    # Mann-Whitney U based AUROC, labels_hard in {0,1}
    order = np.argsort(scores_high_means_hard)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    n1 = labels_hard.sum()
    n0 = len(labels_hard) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    sum_ranks_pos = ranks[labels_hard == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc

def main():
    t0 = time.time()
    K_GRID = [3, 5, 10, 20, 50]
    ALPHA_GRID = {"skewed": 0.3, "moderate": 1.0, "flat": 3.0}
    C_RANGE = (2, 10)  # realistic semantic-alphabet-size range per literature anchor (see paper)
    N_QUERIES = 2000
    N_REPLICATES = 30  # independent replicate runs per cell, for paired stats

    results = {}  # cell key -> dict of replicate-level summaries
    for alpha_name, alpha in ALPHA_GRID.items():
        for K in K_GRID:
            key = f"{alpha_name}_K{K}"
            bias_plugin, bias_mm = [], []
            mse_plugin, mse_mm = [], []
            corr_plugin, corr_mm, corr_top1 = [], [], []
            auc_plugin, auc_mm, auc_top1 = [], [], []
            c_undercount = []
            for r in range(N_REPLICATES):
                seed = stable_seed("cell", alpha_name, K, r)
                d = simulate_cell(N_QUERIES, K, alpha, C_RANGE[0], C_RANGE[1], seed)
                err_p = d["H_plugin"] - d["H_true"]
                err_m = d["H_mm"] - d["H_true"]
                bias_plugin.append(err_p.mean())
                bias_mm.append(err_m.mean())
                mse_plugin.append(np.mean(err_p ** 2))
                mse_mm.append(np.mean(err_m ** 2))
                corr_plugin.append(stats.spearmanr(d["H_plugin"], d["H_true"]).correlation)
                corr_mm.append(stats.spearmanr(d["H_mm"], d["H_true"]).correlation)
                corr_top1.append(stats.spearmanr(-d["top1_conf"], d["H_true"]).correlation)
                hard = (d["H_true"] > np.median(d["H_true"])).astype(int)
                auc_plugin.append(auroc(d["H_plugin"], hard))
                auc_mm.append(auroc(d["H_mm"], hard))
                auc_top1.append(auroc(-d["top1_conf"], hard))
                c_undercount.append(np.mean(d["C_obs"] < d["C_true"]))

            def summ(x):
                x = np.array(x)
                return dict(mean=float(x.mean()), std=float(x.std(ddof=1)))

            paired_bias_t = stats.ttest_rel(np.abs(bias_mm), np.abs(bias_plugin))
            paired_mse_t = stats.ttest_rel(mse_mm, mse_plugin)
            paired_auc_t = stats.ttest_rel(auc_mm, auc_plugin)
            paired_auc_top1_t = stats.ttest_rel(auc_plugin, auc_top1)

            results[key] = dict(
                alpha=alpha, K=K,
                bias_plugin=summ(bias_plugin), bias_mm=summ(bias_mm),
                mse_plugin=summ(mse_plugin), mse_mm=summ(mse_mm),
                corr_plugin=summ(corr_plugin), corr_mm=summ(corr_mm), corr_top1=summ(corr_top1),
                auc_plugin=summ(auc_plugin), auc_mm=summ(auc_mm), auc_top1=summ(auc_top1),
                c_undercount=summ(c_undercount),
                paired_abs_bias_mm_vs_plugin=dict(t=float(paired_bias_t.statistic), p=float(paired_bias_t.pvalue)),
                paired_mse_mm_vs_plugin=dict(t=float(paired_mse_t.statistic), p=float(paired_mse_t.pvalue)),
                paired_auc_mm_vs_plugin=dict(t=float(paired_auc_t.statistic), p=float(paired_auc_t.pvalue)),
                paired_auc_plugin_vs_top1=dict(t=float(paired_auc_top1_t.statistic), p=float(paired_auc_top1_t.pvalue)),
            )
            print(f"{key}: bias_plugin={results[key]['bias_plugin']['mean']:.4f} "
                  f"bias_mm={results[key]['bias_mm']['mean']:.4f} "
                  f"AUC_plugin={results[key]['auc_plugin']['mean']:.4f} "
                  f"AUC_mm={results[key]['auc_mm']['mean']:.4f} "
                  f"AUC_top1={results[key]['auc_top1']['mean']:.4f}")

    # ---- Power analysis for the headline claim: MM correction reduces |bias| at low K ----
    # Focus on the "moderate" alpha regime at K=5 (a realistic, low sampling budget).
    power_results = {}
    for n in [3, 5, 10, 20, 30, 50, 100, 300]:
        sig_count = 0
        n_power_reps = 20
        for rep in range(n_power_reps):
            seed = stable_seed("power", n, rep)
            d = simulate_cell(n, 5, ALPHA_GRID["moderate"], C_RANGE[0], C_RANGE[1], seed)
            err_p = np.abs(d["H_plugin"] - d["H_true"])
            err_m = np.abs(d["H_mm"] - d["H_true"])
            # bootstrap-free paired t-test would need replicate means; instead do a
            # paired t-test across the n_queries themselves (valid since each query
            # independently contributes one paired (plugin,mm) absolute-error pair).
            tt = stats.ttest_rel(err_m, err_p)
            if tt.pvalue < 0.05 and tt.statistic < 0:
                sig_count += 1
        power_results[n] = sig_count / n_power_reps
    print("Power analysis (fraction of replicates with significant |bias| reduction, K=5, moderate alpha):")
    print(power_results)

    # ---- Robustness: does C-range (semantic alphabet size regime) change the qualitative story? ----
    c_range_results = {}
    for name, (lo, hi) in {"narrow_C2-4": (2, 4), "wide_C2-10": (2, 10), "large_C5-20": (5, 20)}.items():
        bias_plugin, bias_mm = [], []
        for r in range(N_REPLICATES):
            seed = stable_seed("crange", name, r)
            d = simulate_cell(N_QUERIES, 5, ALPHA_GRID["moderate"], lo, hi, seed)
            bias_plugin.append(np.mean(d["H_plugin"] - d["H_true"]))
            bias_mm.append(np.mean(d["H_mm"] - d["H_true"]))
        tt = stats.ttest_rel(np.abs(bias_mm), np.abs(bias_plugin))
        c_range_results[name] = dict(
            bias_plugin_mean=float(np.mean(bias_plugin)),
            bias_mm_mean=float(np.mean(bias_mm)),
            paired_t=float(tt.statistic), paired_p=float(tt.pvalue),
        )
    print("C-range robustness check:", c_range_results)

    elapsed = time.time() - t0
    out = dict(
        grid_results=results,
        power_analysis=power_results,
        c_range_robustness=c_range_results,
        config=dict(K_GRID=K_GRID, ALPHA_GRID=ALPHA_GRID, C_RANGE=C_RANGE,
                    N_QUERIES=N_QUERIES, N_REPLICATES=N_REPLICATES),
        elapsed_seconds=elapsed,
    )
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal elapsed: {elapsed:.1f}s. Results written to results.json")

if __name__ == "__main__":
    main()
