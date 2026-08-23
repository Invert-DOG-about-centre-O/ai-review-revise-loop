"""
Address reviewer Q1/Q2: paired significance test for mean_token_entropy vs
best expensive signal, and check whether self_consistency_disagreement and
semantic_entropy are near-degenerate (rank-correlated) on the compact answer
space, especially in seed 2 where their AUROCs coincided.
"""
import json
import numpy as np

SEEDS = [0, 1, 2]


def auroc_np(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUROC (handles ties like the paper's implementation)
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return count / (len(pos) * len(neg))


all_recs = {}
for seed in SEEDS:
    with open(f"records_seed{seed}.json") as f:
        all_recs[seed] = json.load(f)

# --- Q2: correlation between the two expensive "semantic" signals ---
print("=== Q2: self_consistency_disagreement vs semantic_entropy agreement ===")
for seed in SEEDS:
    recs = all_recs[seed]
    scd = np.array([r["self_consistency_disagreement"] for r in recs])
    se = np.array([r["semantic_entropy"] for r in recs])
    corr = np.corrcoef(scd, se)[0, 1]
    exact_equal_rank = np.mean(scd == scd[np.argsort(se)][np.argsort(np.argsort(se))]) if False else None
    # simpler: Spearman via rankdata
    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        return avg[inv]
    r_scd, r_se = rankdata(scd), rankdata(se)
    spearman = np.corrcoef(r_scd, r_se)[0, 1]
    print(f"seed {seed}: Pearson r={corr:.3f}  Spearman rho={spearman:.3f}  "
          f"(n={len(recs)}, both derived from the same K=8 sample set)")

# --- Q1: paired bootstrap significance test, pooling seeds, seed-level resampling ---
print("\n=== Q1: paired significance tests, mean_token_entropy vs each expensive signal ===")


def bootstrap_diff_ci(seed_scores_a, seed_scores_b, n_boot=10000, seed0=0):
    """Bootstrap over the 3 seeds (resample seeds with replacement) to get a
    CI on the mean AUROC gap, reflecting our actual unit of replication."""
    rng = np.random.default_rng(seed0)
    a = np.array(seed_scores_a)
    b = np.array(seed_scores_b)
    diffs = a - b
    n = len(a)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(diffs[idx].mean())
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_le_zero = np.mean(boot_means <= 0)
    return diffs.mean(), ci_lo, ci_hi, p_le_zero


def example_level_bootstrap_auroc_diff(recs_by_seed, sig_a, sig_b, n_boot=2000, seed0=1):
    """Within-seed example-level bootstrap of the AUROC gap, then averaged
    across seeds - captures sampling noise from the n=200 test set itself."""
    rng = np.random.default_rng(seed0)
    per_seed_p = []
    for seed, recs in recs_by_seed.items():
        y = np.array([0 if r["correct"] else 1 for r in recs])
        sa = np.array([r[sig_a] for r in recs])
        sb = np.array([r[sig_b] for r in recs])
        n = len(recs)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(auroc_np(sa[idx], y[idx]) - auroc_np(sb[idx], y[idx]))
        diffs = np.array(diffs)
        p_le_zero = np.mean(diffs <= 0)
        per_seed_p.append((seed, diffs.mean(), p_le_zero))
    return per_seed_p


expensive_signals = ["semantic_entropy", "self_consistency_disagreement", "lexical_diversity"]

with open("summary.json") as f:
    summary = json.load(f)

mte_by_seed = [summary["individual_signal_auroc"]["mean_token_entropy (cheap, 1 pass)"]]
# recompute per-seed AUROCs directly from records for exactness
per_seed_auroc = {sig: [] for sig in expensive_signals + ["mean_token_entropy"]}
for seed in SEEDS:
    recs = all_recs[seed]
    y = np.array([0 if r["correct"] else 1 for r in recs])
    for sig in expensive_signals + ["mean_token_entropy"]:
        s = np.array([r[sig] for r in recs])
        per_seed_auroc[sig].append(auroc_np(s, y))

print("Per-seed AUROC recomputed from records.json (sanity check against results_seed*.json):")
for sig in per_seed_auroc:
    print(f"  {sig}: {[round(v,3) for v in per_seed_auroc[sig]]}")

print("\nSeed-level bootstrap (resampling which of the 3 seeds we 'ran'), 10000 resamples:")
for sig in expensive_signals:
    mean_diff, lo, hi, p = bootstrap_diff_ci(per_seed_auroc["mean_token_entropy"], per_seed_auroc[sig])
    print(f"  mean_token_entropy - {sig}: mean_diff={mean_diff:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  "
          f"P(diff<=0 under seed-resampling)={p:.3f}")

print("\nExample-level bootstrap (within each seed's 200-example test set, 2000 resamples/seed),")
print("P(diff <= 0) per seed for mean_token_entropy vs semantic_entropy (best expensive signal):")
res = example_level_bootstrap_auroc_diff(all_recs, "mean_token_entropy", "semantic_entropy")
for seed, mean_diff, p in res:
    print(f"  seed {seed}: mean_diff={mean_diff:+.3f}  P(diff<=0)={p:.3f}")

out = {
    "per_seed_auroc_recheck": {k: v for k, v in per_seed_auroc.items()},
    "seed_level_bootstrap": {
        sig: dict(zip(
            ["mean_diff", "ci_lo", "ci_hi", "p_le_zero"],
            bootstrap_diff_ci(per_seed_auroc["mean_token_entropy"], per_seed_auroc[sig]),
        ))
        for sig in expensive_signals
    },
    "example_level_bootstrap_vs_semantic_entropy": {
        f"seed{seed}": {"mean_diff": mean_diff, "p_le_zero": p} for seed, mean_diff, p in res
    },
    "scd_vs_se_correlation": {
        f"seed{seed}": {
            "pearson": float(np.corrcoef(
                [r["self_consistency_disagreement"] for r in all_recs[seed]],
                [r["semantic_entropy"] for r in all_recs[seed]])[0, 1])
        } for seed in SEEDS
    },
}
with open("stats_check.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved stats_check.json")
