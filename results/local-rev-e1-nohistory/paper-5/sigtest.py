"""Paired significance tests on the existing 5-seed revision_results.json
(no retraining needed -- addresses reviewer's point that no significance
test was reported for the main-run KL/ECE gap)."""
import json
import numpy as np
from scipy import stats

d = json.load(open("revision_results.json"))
per_seed = d["per_seed"]

kl_raw = np.array([r["kl_raw"] for r in per_seed])
kl_nll = np.array([r["kl_scaled_nll"] for r in per_seed])
kl_kl = np.array([r["kl_scaled_kl"] for r in per_seed])
ece_raw = np.array([r["ece_raw"] for r in per_seed])
ece_nll = np.array([r["ece_scaled_nll"] for r in per_seed])
ece_kl = np.array([r["ece_scaled_kl"] for r in per_seed])

def paired(a, b, name):
    diff = b - a
    t, p = stats.ttest_rel(a, b)
    wt, wp = stats.wilcoxon(a, b) if len(a) >= 5 else (None, None)
    print(f"{name}: mean diff={diff.mean():.5f} (n={len(a)}), paired t={t:.3f}, p={p:.4f}, "
          f"wilcoxon p={wp:.4f}" if wp is not None else f"{name}: mean diff={diff.mean():.5f}, t={t:.3f}, p={p:.4f}")
    return dict(mean_diff=float(diff.mean()), t=float(t), p=float(p),
                wilcoxon_p=(float(wp) if wp is not None else None))

out = {}
out["kl_raw_vs_nll"] = paired(kl_raw, kl_nll, "KL raw vs NLL-scaled")
out["kl_raw_vs_kl"] = paired(kl_raw, kl_kl, "KL raw vs KL-scaled")
out["ece_raw_vs_nll"] = paired(ece_raw, ece_nll, "ECE raw vs NLL-scaled")
out["ece_raw_vs_kl"] = paired(ece_raw, ece_kl, "ECE raw vs KL-scaled")

with open("sigtest_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
