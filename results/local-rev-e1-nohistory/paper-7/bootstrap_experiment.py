"""
Round-2 revision: address reviewer's core complaint that the 14-epoch
in-distribution ablation headline numbers (semantic=0.931, lexical=0.930,
token=0.883) do not reproduce run-to-run and carry no uncertainty estimate.

This script (a) runs MANY more seeds of the in-distribution ablation than
the original 5, to shrink Monte-Carlo noise in the pooled AUROC estimate,
and (b) computes seed-level bootstrap 95% CIs and paired bootstrap
significance tests for the pairwise AUROC gaps (semantic-token,
semantic-lexical, lexical-token), so the paper can report an interval and
an explicit "is this gap distinguishable from zero" answer instead of a
single point estimate.
"""
import json
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Round-3 fix: the round-2 reviewer's re-run of this exact script (same
# seeds) produced different AUROC numbers because BLAS/threading
# non-determinism affects sampling here, exactly as diagnosed (but not
# acted on) in followup_experiments.py's determinism check. Pin to a
# single thread so the headline numbers are actually bit-reproducible.
torch.set_num_threads(1)

import experiment as exp

t0 = time.time()
N_SEEDS = 20
EPOCHS = 14
SEED_START = 500

runs = []
for i in range(N_SEEDS):
    seed = SEED_START + i
    r = exp.run_seed(seed, epochs=EPOCHS)
    runs.append(r)
    print(f"[{time.time()-t0:.1f}s] seed={seed} acc_seen={r['acc_seen']:.2f} acc_unseen={r['acc_unseen']:.2f}")

# Pool all "seen" records across seeds, tagging each with its seed so we can
# bootstrap-resample at the SEED level (not the record level), which is the
# correct unit of resampling here since records within a seed share a model.
pooled = []
for r in runs:
    for rec in r["records"]:
        if rec["split"] == "seen":
            pooled.append(dict(seed=r["seed"], correct=rec["correct"],
                                pred_entropy=rec["pred_entropy"],
                                lex_entropy=rec["lex_entropy"],
                                sem_entropy=rec["sem_entropy"]))

seeds_present = sorted(set(p["seed"] for p in pooled))
n_seen_total = len(pooled)
n_wrong_total = sum(1 for p in pooled if not p["correct"])
print(f"[{time.time()-t0:.1f}s] pooled n_seen={n_seen_total} n_wrong={n_wrong_total} "
      f"n_seeds={len(seeds_present)}")


def pooled_auroc(records, method_key):
    y = np.array([0 if p["correct"] else 1 for p in records])
    if len(set(y)) < 2:
        return float("nan")
    s = np.array([p[method_key] for p in records])
    return float(roc_auc_score(y, s))


methods = {"predictive_entropy_token": "pred_entropy",
           "lexical_entropy": "lex_entropy",
           "semantic_entropy": "sem_entropy"}

point_estimate = {name: pooled_auroc(pooled, key) for name, key in methods.items()}
print(f"[{time.time()-t0:.1f}s] point estimate (pooled over {len(seeds_present)} seeds):",
      json.dumps(point_estimate, indent=2))

# Seed-level (cluster) bootstrap: resample seeds with replacement, pool
# their records, recompute AUROC. Repeat B times.
rng = np.random.RandomState(0)
B = 2000
by_seed = {s: [p for p in pooled if p["seed"] == s] for s in seeds_present}
boot = {name: [] for name in methods}
boot_diff = {"semantic_minus_token": [], "semantic_minus_lexical": [], "lexical_minus_token": []}

for b in range(B):
    sample_seeds = rng.choice(seeds_present, size=len(seeds_present), replace=True)
    resampled = []
    for s in sample_seeds:
        resampled.extend(by_seed[s])
    y = np.array([0 if p["correct"] else 1 for p in resampled])
    if len(set(y)) < 2:
        continue
    vals = {}
    for name, key in methods.items():
        sc = np.array([p[key] for p in resampled])
        vals[name] = roc_auc_score(y, sc)
        boot[name].append(vals[name])
    boot_diff["semantic_minus_token"].append(vals["semantic_entropy"] - vals["predictive_entropy_token"])
    boot_diff["semantic_minus_lexical"].append(vals["semantic_entropy"] - vals["lexical_entropy"])
    boot_diff["lexical_minus_token"].append(vals["lexical_entropy"] - vals["predictive_entropy_token"])

summary = {"n_seeds": len(seeds_present), "n_bootstrap_valid": len(boot["semantic_entropy"]),
           "n_seen_total": n_seen_total, "n_wrong_total": n_wrong_total,
           "point_estimate": point_estimate, "bootstrap_ci95": {}, "pairwise_diff": {}}

for name in methods:
    arr = np.array(boot[name])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    summary["bootstrap_ci95"][name] = {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}

for name, arr_list in boot_diff.items():
    arr = np.array(arr_list)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    frac_pos = float((arr > 0).mean())
    summary["pairwise_diff"][name] = {
        "mean_diff": float(arr.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
        "frac_bootstrap_gt_0": frac_pos,
        "significant_at_5pct": bool(lo > 0 or hi < 0),
    }

print(f"[{time.time()-t0:.1f}s] === BOOTSTRAP SUMMARY ===")
print(json.dumps(summary, indent=2))

with open("bootstrap_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[{time.time()-t0:.1f}s] DONE total={time.time()-t0:.1f}s")
