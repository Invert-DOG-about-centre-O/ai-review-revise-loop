"""
Reviewer-requested check: width and test accuracy are confounded in the main
sweep (accuracy climbs steeply with width, esp. on digits). This checks
whether the H1 width-vs-ece_raw correlation is really a width effect or an
accuracy effect, via (a) partial Spearman correlation of width vs ece_raw
controlling for acc_test, and (b) the raw width-ece_raw correlation within
the single cell where H1 was significant (blobs/label_smooth).
"""
import json

import numpy as np
from scipy import stats

with open("raw_results.json") as f:
    data = json.load(f)
results = data["results"]
datasets = sorted(set(r["dataset"] for r in results))
conditions = data["config"]["conditions"]

def partial_spearman(x, y, z):
    # partial correlation of x,y controlling for z, via rank residuals
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    def resid(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx_resid = resid(rx, rz)
    ry_resid = resid(ry, rz)
    r, p = stats.pearsonr(rx_resid, ry_resid)
    return r, p

out = {}
for ds in datasets:
    for cond in conditions:
        rows = [r for r in results if r["dataset"] == ds and r["condition"] == cond]
        w = np.array([r["width"] for r in rows], dtype=float)
        e = np.array([r["ece_raw"] for r in rows])
        a = np.array([r["acc_test"] for r in rows])
        rho_raw, p_raw = stats.spearmanr(w, e)
        rho_acc_e, p_acc_e = stats.spearmanr(a, e)
        r_partial, p_partial = partial_spearman(w, e, a)
        out[f"{ds}__{cond}"] = {
            "spearman_width_ece_raw": float(rho_raw),
            "spearman_acc_ece_raw": float(rho_acc_e),
            "partial_corr_width_ece_given_acc": float(r_partial),
            "partial_p": float(p_partial),
        }

with open("accuracy_control_results.json", "w") as f:
    json.dump(out, f, indent=1)

for k, v in out.items():
    print(f"{k}: width-ece rho={v['spearman_width_ece_raw']:.3f}  acc-ece rho={v['spearman_acc_ece_raw']:.3f}  "
          f"partial(width|acc)={v['partial_corr_width_ece_given_acc']:.3f} p={v['partial_p']:.3g}")
