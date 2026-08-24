"""
Post-hoc analysis of the coarse sweep: quantify how strongly the algorithmic
exposure-bias parameter (alpha) predicts final polarization (variance),
compared against the confidence-threshold parameter (epsilon), using
threshold-free descriptors (Pearson r, OLS slope, and total variation of the
curve) rather than an arbitrary cutoff-based classification.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_coarse.csv")))
for r in rows:
    for k in ("alpha", "epsilon", "variance_mean", "variance_std", "extremeness_mean", "amp_ratio_mean"):
        r[k] = float(r[k])

summary = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

colors = {("mixing", 0.15): "#1f77b4", ("mixing", 0.35): "#d62728",
          ("smallworld", 0.15): "#2ca02c", ("smallworld", 0.35): "#9467bd"}

for topology in ["mixing", "smallworld"]:
    for epsilon in [0.15, 0.35]:
        sub = sorted([r for r in rows if r["topology"] == topology and r["epsilon"] == epsilon],
                     key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in sub])
        v = np.array([r["variance_mean"] for r in sub])
        amp = np.array([r["amp_ratio_mean"] for r in sub])
        # Pearson correlation and OLS slope of variance vs alpha
        corr = float(np.corrcoef(a, v)[0, 1])
        slope, intercept = np.polyfit(a, v, 1)
        total_variation = float(np.sum(np.abs(np.diff(v))))
        v_range = float(v.max() - v.min())
        # same regression, but for amp_ratio vs alpha (sanity check mechanism works)
        amp_corr = float(np.corrcoef(a, amp)[0, 1])
        key = f"{topology}_eps{epsilon}"
        summary[key] = {
            "pearson_r_alpha_vs_variance": round(corr, 4),
            "ols_slope_alpha_vs_variance": round(float(slope), 5),
            "variance_total_variation": round(total_variation, 4),
            "variance_range": round(v_range, 4),
            "variance_baseline_alpha0": round(float(v[a == 0][0]), 4),
            "pearson_r_alpha_vs_amp_ratio": round(amp_corr, 4),
        }
        ax = axes[0]
        ax.plot(a, v, marker="o", ms=3, lw=1.3, color=colors[(topology, epsilon)],
                label=f"{topology}, eps={epsilon}")
        ax2 = axes[1]
        ax2.plot(a, amp, marker="o", ms=3, lw=1.3, color=colors[(topology, epsilon)],
                 label=f"{topology}, eps={epsilon}")

calib = json.load(open("calibration.json"))
axes[1].axhline(calib["target_ratio"], color="gray", ls="--", lw=1,
                 label=f"real-world target ({calib['target_ratio']}x)")
axes[1].axvline(calib["calibrated_alpha"], color="gray", ls=":", lw=1)

axes[0].set_xlabel("algorithmic exposure bias (alpha)")
axes[0].set_ylabel("final opinion variance")
axes[0].set_title("Polarization vs. algorithmic bias strength")
axes[0].legend(fontsize=8)

axes[1].set_xlabel("algorithmic exposure bias (alpha)")
axes[1].set_ylabel("same-side / opposite-side exposure ratio")
axes[1].set_title("Exposure amplification vs. bias strength")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("figure.png", dpi=140)

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

for k, v in summary.items():
    print(k, v)
