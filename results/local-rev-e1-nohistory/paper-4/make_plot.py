import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    out = json.load(f)

schemes = ["accuracy_oracle", "approval_flat", "approval_confidence"]
labels = {
    "accuracy_oracle": "Accuracy-oracle (verified reward)",
    "approval_flat": "Approval-flat (pure agreement)",
    "approval_confidence": "Approval-confidence-modulated",
}
colors = {"accuracy_oracle": "#4C72B0", "approval_flat": "#C44E52", "approval_confidence": "#DD8452"}

fig, ax = plt.subplots(figsize=(6, 4.2))
for scheme in schemes:
    q = out["results"][scheme]["by_confidence_quartile"]
    x = [(item["c_lo"] + item["c_hi"]) / 2 for item in q]
    y = [item["sycophancy_rate"] for item in q]
    ax.plot(x, y, marker="o", label=labels[scheme], color=colors[scheme], linewidth=2)

ax.set_xlabel("User-expressed confidence c (quartile midpoint)")
ax.set_ylabel("Sycophancy rate\n(P(defer to user) | own belief disagrees)")
ax.set_ylim(-0.02, 1.05)
ax.set_title("Sycophancy rate vs. user confidence, by reward scheme")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("sycophancy_vs_confidence.png", dpi=150)
print("saved plot")
