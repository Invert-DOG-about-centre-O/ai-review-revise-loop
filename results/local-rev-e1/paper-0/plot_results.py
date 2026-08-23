import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    results = json.load(f)

alphas = [0.0, 0.3, 0.6, 0.9, 1.0]
baseline = results["none"]["between_gap_mean"]

pers = [results[f"personalized_a{a}"]["between_gap_mean"] for a in alphas]
shared = [results[f"shared_a{a}"]["between_gap_mean"] for a in alphas]
pers_err = [results[f"personalized_a{a}"]["between_gap_std"] for a in alphas]
shared_err = [results[f"shared_a{a}"]["between_gap_std"] for a in alphas]

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.axhline(baseline, color="gray", linestyle="--", label="No-AI baseline")
ax.errorbar(alphas, pers, yerr=pers_err, marker="o", label="Personalized AI (no cross-talk)")
ax.errorbar(alphas, shared, yerr=shared_err, marker="s", label="Shared AI (single global model)")
ax.set_xlabel("Sycophancy coefficient (alpha)")
ax.set_ylabel("Between-community opinion gap")
ax.set_title("AI-mediated cross-community opinion leakage")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results_plot.png", dpi=150)
print("saved results_plot.png")
