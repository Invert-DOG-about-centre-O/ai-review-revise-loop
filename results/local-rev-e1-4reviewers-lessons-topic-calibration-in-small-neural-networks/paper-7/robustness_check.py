import json
import numpy as np
from scipy.stats import wilcoxon

d = json.load(open("results.json"))
out = {}
for name in ["digits", "blobs"]:
    rows = [r for r in d["per_seed_results"] if r["dataset"] == name]
    diffs = np.array([r["ece_single_cal"] - r["ece_ensemble_cal"] for r in rows])
    idx = np.argmax(np.abs(diffs))
    diffs_loo = np.delete(diffs, idx)
    w, p = wilcoxon(diffs_loo)
    rng = np.random.RandomState(0)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out[name] = dict(
        full_mean=float(diffs.mean()),
        loo_dropped_seed=int(rows[idx]["seed"]),
        loo_mean=float(diffs_loo.mean()),
        loo_wilcoxon_p=float(p),
        bootstrap_ci95=[float(lo), float(hi)],
    )
    print(name, out[name])

with open("robustness_results.json", "w") as f:
    json.dump(out, f, indent=2)
