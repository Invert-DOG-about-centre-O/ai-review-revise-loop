"""
Revision-round analysis, addressing reviewer requests:
1. Fisher's exact test on crossover EXISTENCE (binary: defined vs None) between
   datasets, which several reviewers noted has far more power than the
   pre-registered Mann-Whitney on the conditional widths. Reported as an
   EXPLORATORY (not confirmatory / not pre-registered) secondary test.
2. Failure-mode breakdown for seeds without a stable crossover: "never touches
   overconfidence at any width" vs "touches it but reverts" (oscillates).
Both are purely descriptive of already-collected data; no new runs, no change
to the single pre-registered confirmatory test or its n.
"""
import json
from scipy.stats import fisher_exact

with open("results.json") as f:
    d = json.load(f)

results = d["raw_results"]
widths = d["widths"]
seeds = d["seeds"]
crossover_widths = d["crossover_widths"]  # {"digits": {seed: width_or_None}, "synthetic": {...}}

# 1. Fisher's exact test on existence (2x2: dataset x has-defined-crossover)
fisher_table = []
exist_counts = {}
for dataset_name in ["digits", "synthetic"]:
    cw = crossover_widths[dataset_name]
    n_defined = sum(1 for v in cw.values() if v is not None)
    n_undefined = sum(1 for v in cw.values() if v is None)
    fisher_table.append([n_defined, n_undefined])
    exist_counts[dataset_name] = {"defined": n_defined, "undefined": n_undefined, "total": len(cw)}

odds_ratio, p_fisher = fisher_exact(fisher_table)

# 2. Failure-mode breakdown: for seeds with NO stable (non-reverting) crossover,
# did bias ever touch >=0 at any width ("touches but reverts") or never ("never touches")?
failure_modes = {"digits": {"never_touches": 0, "touches_but_reverts": 0},
                  "synthetic": {"never_touches": 0, "touches_but_reverts": 0}}
for dataset_name in ["digits", "synthetic"]:
    cw = crossover_widths[dataset_name]
    for seed_str, stable_w in cw.items():
        seed = int(seed_str)
        if stable_w is not None:
            continue  # has a stable crossover, not a failure case
        biases = [r["bias"] for r in results if r["dataset"] == dataset_name and r["seed"] == seed]
        biases_by_width = {r["width"]: r["bias"] for r in results if r["dataset"] == dataset_name and r["seed"] == seed}
        ever_touched = any(biases_by_width[w] >= 0 for w in widths)
        if ever_touched:
            failure_modes[dataset_name]["touches_but_reverts"] += 1
        else:
            failure_modes[dataset_name]["never_touches"] += 1

out = {
    "fisher_exact_crossover_existence": {
        "table_defined_undefined_digits_then_synthetic": fisher_table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_fisher),
        "note": "EXPLORATORY, not pre-registered/confirmatory; tests whether the PROBABILITY of a defined crossover differs by dataset, not its location.",
    },
    "existence_counts": exist_counts,
    "failure_mode_breakdown_among_undefined_crossover_seeds": failure_modes,
}

d.setdefault("analysis", {})
d["analysis"]["revision_round_exploratory"] = out
with open("results.json", "w") as f:
    json.dump(d, f, indent=2)

print(json.dumps(out, indent=2))
