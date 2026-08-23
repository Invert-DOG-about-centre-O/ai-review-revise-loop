import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    data = json.load(f)["results"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

order = ["random_0.0", "calibrated_0.0", "bridging_0.5",
         "sycophantic_0.0", "sycophantic_0.25", "sycophantic_0.5",
         "sycophantic_0.75", "sycophantic_1.0"]
labels = ["random", "calibrated", "bridging(0.5)",
          "syco(0.0)", "syco(0.25)", "syco(0.5)", "syco(0.75)", "syco(1.0)"]

for key, lab in zip(order, labels):
    curve = data[key]["var_curve"]
    axes[0].plot(curve, label=lab)
axes[0].set_title("Opinion variance over time")
axes[0].set_xlabel("round")
axes[0].set_ylabel("variance")
axes[0].legend(fontsize=7)

for key, lab in zip(order, labels):
    curve = data[key]["bc_curve"]
    axes[1].plot(curve, label=lab)
axes[1].axhline(5/9, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_title("Bimodality coefficient over time")
axes[1].set_xlabel("round")
axes[1].set_ylabel("bimodality coeff.")

s_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
final_var = [data[f"sycophantic_{s}"]["final_var_mean"] for s in s_vals]
final_var_std = [data[f"sycophantic_{s}"]["final_var_std"] for s in s_vals]
final_bc = [data[f"sycophantic_{s}"]["final_bc_mean"] for s in s_vals]
final_ext = [data[f"sycophantic_{s}"]["final_extremity_mean"] for s in s_vals]
axes[2].errorbar(s_vals, final_var, yerr=final_var_std, marker="o", label="variance")
axes[2].plot(s_vals, final_bc, marker="s", label="bimodality coeff.")
axes[2].plot(s_vals, final_ext, marker="^", label="mean |opinion|")
axes[2].set_title("Final-round metrics vs. sycophancy strength s")
axes[2].set_xlabel("sycophancy strength s")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results_plot.png", dpi=140)
print("Saved results_plot.png")
