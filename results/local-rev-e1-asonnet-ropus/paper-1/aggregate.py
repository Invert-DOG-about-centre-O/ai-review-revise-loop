"""
Aggregate results across seeds and evaluate a learned combination of the
cheap single-pass signals (logistic regression probe) against the
expensive sampling-based signals, using a held-out split of each seed's
test set to fit the probe (no leakage).
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEEDS = [0, 1, 2]

per_seed_summary = []
probe_aurocs = {"cheap_probe": [], "expensive_probe": [], "all_probe": []}

for seed in SEEDS:
    with open(f"results_seed{seed}.json") as f:
        res = json.load(f)
    with open(f"records_seed{seed}.json") as f:
        recs = json.load(f)

    per_seed_summary.append(res)

    n = len(recs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fit_idx, eval_idx = idx[: n // 2], idx[n // 2:]

    def col(name):
        return np.array([r[name] for r in recs], dtype=float)

    err = np.array([0 if r["correct"] else 1 for r in recs])
    cheap = np.stack([col("mean_token_entropy"), col("max_token_entropy")], axis=1)
    expensive = np.stack([col("lexical_diversity"), col("self_consistency_disagreement"),
                           col("semantic_entropy")], axis=1)
    allfeat = np.hstack([cheap, expensive])

    for name, feats in [("cheap_probe", cheap), ("expensive_probe", expensive), ("all_probe", allfeat)]:
        Xf, yf = feats[fit_idx], err[fit_idx]
        Xe, ye = feats[eval_idx], err[eval_idx]
        if len(np.unique(yf)) < 2 or len(np.unique(ye)) < 2:
            probe_aurocs[name].append(float("nan"))
            continue
        clf = LogisticRegression(max_iter=1000).fit(Xf, yf)
        score = clf.predict_proba(Xe)[:, 1]
        probe_aurocs[name].append(roc_auc_score(ye, score))

print("=== Per-seed summary ===")
for seed, res in zip(SEEDS, per_seed_summary):
    print(f"seed {seed}: greedy_acc={res['greedy_accuracy']:.3f} "
          f"majority_acc={res['majority_accuracy']:.3f} "
          f"cost_ratio={res['cost_ratio']:.2f}x")
    for k, v in res["auroc"].items():
        print(f"    {k}: {v:.3f}")

print("\n=== Mean +/- std across seeds (individual signals) ===")
keys = list(per_seed_summary[0]["auroc"].keys())
agg = {}
for k in keys:
    vals = [res["auroc"][k] for res in per_seed_summary]
    agg[k] = (float(np.mean(vals)), float(np.std(vals)))
    print(f"  {k}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

print("\n=== Learned probes (fit on held-out half of each seed's test set) ===")
agg_probes = {}
for name, vals in probe_aurocs.items():
    v = np.array(vals, dtype=float)
    agg_probes[name] = (float(np.nanmean(v)), float(np.nanstd(v)))
    print(f"  {name}: {np.nanmean(v):.3f} +/- {np.nanstd(v):.3f}  (per-seed: {['%.3f'%x for x in v]})")

acc_mean = np.mean([res["greedy_accuracy"] for res in per_seed_summary])
acc_std = np.std([res["greedy_accuracy"] for res in per_seed_summary])
maj_mean = np.mean([res["majority_accuracy"] for res in per_seed_summary])
cost_mean = np.mean([res["cost_ratio"] for res in per_seed_summary])

summary = {
    "seeds": SEEDS,
    "greedy_accuracy_mean": float(acc_mean), "greedy_accuracy_std": float(acc_std),
    "majority_accuracy_mean": float(maj_mean),
    "cost_ratio_mean": float(cost_mean),
    "individual_signal_auroc": agg,
    "learned_probe_auroc": agg_probes,
}
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary.json")
