import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    R = json.load(f)

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
acc = [R["alpha_sweep"][str(a)]["final_accuracy_mean"] for a in alphas]
acc_std = [R["alpha_sweep"][str(a)]["final_accuracy_std"] for a in alphas]
syco = [R["alpha_sweep"][str(a)]["final_sycophancy_mean"] for a in alphas]
syco_std = [R["alpha_sweep"][str(a)]["final_sycophancy_std"] for a in alphas]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.errorbar(alphas, acc, yerr=acc_std, marker="o", capsize=3, color="#2b6cb0", label="Accuracy")
ax.errorbar(alphas, syco, yerr=syco_std, marker="s", capsize=3, color="#c53030", label="Sycophancy rate")
ax.axhline(R["supervised_oracle_baseline"]["final_accuracy_mean"], color="gray", ls="--", lw=1, label="Oracle SFT accuracy")
ax.set_xlabel("Rater sycophancy bias (alpha)")
ax.set_ylabel("Rate")
ax.set_title("Effect of rater bias on trained policy")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.85)

n_raters = [1, 3, 5, 9]
acc2 = [R["rater_aggregation"][str(n)]["final_accuracy_mean"] for n in n_raters]
acc2_std = [R["rater_aggregation"][str(n)]["final_accuracy_std"] for n in n_raters]
syco2 = [R["rater_aggregation"][str(n)]["final_sycophancy_mean"] for n in n_raters]
syco2_std = [R["rater_aggregation"][str(n)]["final_sycophancy_std"] for n in n_raters]

ax = axes[1]
ax.errorbar(n_raters, acc2, yerr=acc2_std, marker="o", capsize=3, color="#2b6cb0", label="Accuracy")
ax.errorbar(n_raters, syco2, yerr=syco2_std, marker="s", capsize=3, color="#c53030", label="Sycophancy rate")
ax.axhline(R["supervised_oracle_baseline"]["final_accuracy_mean"], color="gray", ls="--", lw=1, label="Oracle SFT accuracy")
ax.set_xlabel("Number of aggregated raters (alpha=0.75)")
ax.set_title("Effect of rater aggregation")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.85)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("saved results.png")
