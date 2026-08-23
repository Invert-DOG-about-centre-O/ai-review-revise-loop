import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

with open("traces.json") as f:
    traces = json.load(f)

labels = {
    "STATIC_HONEST": "Static honest (alpha=0)",
    "STATIC_SYCOPHANT": "Static sycophant (alpha=1)",
    "APPROVAL_ONLY": "Approval-only training",
    "REG_lambda0.5_q0.3": "Accuracy-regularized (lambda=0.5, q=0.3)",
    "TRANSPARENCY_0.8": "Transparency mitigation (0.8)",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name, lab in labels.items():
    d = traces[name]
    axes[0].plot(d["alpha"], label=lab, linewidth=1.2)
    axes[1].plot(d["acc_roll"], label=lab, linewidth=1.2)

axes[0].set_title("Sycophancy dial (alpha) over training")
axes[0].set_xlabel("round")
axes[0].set_ylabel("alpha")
axes[1].set_title("Rolling accuracy (window=100)")
axes[1].set_xlabel("round")
axes[1].set_ylabel("accuracy")
axes[1].axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="AI evidence accuracy (0.80)")
axes[0].legend(fontsize=7, loc="center right")
axes[1].legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("results_plot.png", dpi=140)
print("saved")
