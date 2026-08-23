"""
Reviewer round-2 Q1: the example-level bootstrap in stats_check.py always
compares mean_token_entropy vs semantic_entropy. Redo it against whichever
expensive signal is BEST for that seed (paired, same resampled example set
for both signals each draw), to check if cherry-picking the weakest expensive
comparator was inflating the "matches" conclusion.
"""
import json
import numpy as np

SEEDS = [0, 1, 2]
expensive_signals = ["semantic_entropy", "self_consistency_disagreement", "lexical_diversity"]


def auroc_np(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return count / (len(pos) * len(neg))


all_recs = {}
for seed in SEEDS:
    with open(f"records_seed{seed}.json") as f:
        all_recs[seed] = json.load(f)

rng = np.random.default_rng(2)
results = {}
for seed in SEEDS:
    recs = all_recs[seed]
    y = np.array([0 if r["correct"] else 1 for r in recs])
    mte = np.array([r["mean_token_entropy"] for r in recs])
    per_sig_auroc = {sig: auroc_np(np.array([r[sig] for r in recs]), y) for sig in expensive_signals}
    best_sig = max(per_sig_auroc, key=per_sig_auroc.get)
    best_scores = np.array([r[best_sig] for r in recs])
    n = len(recs)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auroc_np(mte[idx], y[idx]) - auroc_np(best_scores[idx], y[idx]))
    diffs = np.array(diffs)
    p_le_zero = float(np.mean(diffs <= 0))
    results[f"seed{seed}"] = {
        "best_expensive_signal": best_sig,
        "best_expensive_auroc": per_sig_auroc[best_sig],
        "mean_token_entropy_auroc": float(auroc_np(mte, y)),
        "mean_diff": float(diffs.mean()),
        "p_le_zero": p_le_zero,
    }
    print(f"seed {seed}: best expensive signal = {best_sig} (AUROC={per_sig_auroc[best_sig]:.3f}), "
          f"mean_token_entropy={auroc_np(mte, y):.3f}, mean_diff={diffs.mean():+.3f}, P(diff<=0)={p_le_zero:.3f}")

with open("paired_best_bootstrap.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved paired_best_bootstrap.json")
