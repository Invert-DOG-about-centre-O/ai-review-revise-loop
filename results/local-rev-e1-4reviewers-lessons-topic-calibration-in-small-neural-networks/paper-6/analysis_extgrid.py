"""Analysis of raw_results_extgrid.json -- same tests as analysis.py, run on
the grid-ceiling-corrected primary sweep (T* fit with [0.1,500] grid for all
320 cells), per round-3 reviewer requests to propagate the fix through the
full pipeline rather than leaving it as a 100-run diagnostic."""
import json
import collections
import numpy as np
from scipy import stats

with open("raw_results_extgrid.json") as f:
    data = json.load(f)

results = data["results"]
widths = data["config"]["widths"]
by_ds = collections.defaultdict(list)
for r in results:
    by_ds[r["dataset"]].append(r)


def permutation_spearman(x, y, n_perm=10000, seed=0):
    rho_obs, _ = stats.spearmanr(x, y)
    rng = np.random.RandomState(seed)
    count = 0
    for i in range(n_perm):
        yp = rng.permutation(y)
        r, _ = stats.spearmanr(x, yp)
        if abs(r) >= abs(rho_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return rho_obs, p


out = {"per_cell_summary": {}, "primary_tests": {}, "pinning": {}}

for ds, rows in by_ds.items():
    out["per_cell_summary"][ds] = {}
    for w in widths:
        cell = [r for r in rows if r["width"] == w]
        def summ(key):
            vals = np.array([c[key] for c in cell])
            return float(vals.mean())
        out["per_cell_summary"][ds][str(w)] = {
            "acc": summ("acc"), "ece_pre": summ("ece_pre"), "ece_post": summ("ece_post"),
            "T_star": summ("T_star"),
            "n_pinned_orig_grid20": int(sum(c["pinned_orig_grid20"] for c in cell)),
        }

alpha_bonf = 0.05 / 6
for ds, rows in by_ds.items():
    log2w = np.array([np.log2(r["width"]) for r in rows])
    ece_pre = np.array([r["ece_pre"] for r in rows])
    T_star = np.array([r["T_star"] for r in rows])
    rho1, p1 = permutation_spearman(log2w, ece_pre, seed=1)
    rho2, p2 = permutation_spearman(log2w, T_star, seed=2)
    out["primary_tests"][ds] = {
        "H1_width_vs_ece_pre": {"rho": rho1, "p": p1},
        "H2_width_vs_Tstar_extgrid": {"rho": rho2, "p": p2},
    }
    worse = sum(1 for r in rows if r["ece_post"] > r["ece_pre"])
    out["primary_tests"][ds]["ts_made_worse"] = {"count": worse, "n": len(rows)}

for ds, rows in by_ds.items():
    pinned_by_w = collections.Counter()
    for r in rows:
        if r["pinned_orig_grid20"]:
            pinned_by_w[r["width"]] += 1
    out["pinning"][ds] = dict(pinned_by_w)

with open("analysis_extgrid_results.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
