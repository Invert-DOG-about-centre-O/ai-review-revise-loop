import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    data = json.load(f)

histories = data["histories"]

label_map = {
    "naive_rlhf": "Naive RLHF",
    "robust_median_agg": "Robust median agg. (w/ outlier raters)",
    "sycophancy_penalty": "Sycophancy L2 penalty on pressure weight",
    "precommit_answer": "Pre-commit answer (hide stance 50% of steps)",
    "naive_rlhf_with_outliers": "Naive RLHF (w/ outlier raters)",
}
colors = {
    "naive_rlhf": "#4C72B0",
    "robust_median_agg": "#DD8452",
    "sycophancy_penalty": "#55A868",
    "precommit_answer": "#8172B2",
    "naive_rlhf_with_outliers": "#C44E52",
}

fig, ax = plt.subplots(figsize=(7, 4.5))
for key, hist in histories.items():
    iters = [h["iter"] for h in hist]
    g = [h["g"] for h in hist]
    ax.plot(iters, g, label=label_map[key], color=colors[key], linewidth=2)

ax.set_xlabel("Training iteration")
ax.set_ylabel("Learned pressure weight  g\n(higher = more sycophantic)")
ax.set_title("Emergence of sycophancy during REINFORCE training\non simulated annotator approval")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("sycophancy_training_curves.png", dpi=150)
print("saved sycophancy_training_curves.png")
