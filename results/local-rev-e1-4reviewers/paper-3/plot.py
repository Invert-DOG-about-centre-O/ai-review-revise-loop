import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis_summary.json") as f:
    s = json.load(f)

Ks = sorted(int(k) for k in s["tradeoff"].keys())
sem = [s["tradeoff"][str(k)]["sem_ent_auroc_mean"] for k in Ks]
sem_std = [s["tradeoff"][str(k)]["sem_ent_auroc_std"] for k in Ks]
sc = [s["tradeoff"][str(k)]["sc_auroc_mean"] for k in Ks]
sc_std = [s["tradeoff"][str(k)]["sc_auroc_std"] for k in Ks]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.errorbar(Ks, sem, yerr=sem_std, marker="o", label="semantic entropy")
ax.errorbar(Ks, sc, yerr=sc_std, marker="s", label="self-consistency")
ax.axhline(s["auroc"]["neg_mean_logp (1 pass, greedy)"], color="gray",
           linestyle="--", label="neg mean logp (1 pass)")
ax.axhline(s["auroc"]["first_token_entropy (1 pass)"], color="lightgray",
           linestyle=":", label="first-token entropy (1 pass)")
ax.set_xlabel("K (number of sampled generations)")
ax.set_ylabel("AUROC (detecting wrong answers)")
ax.set_title("Cost (K samples) vs. error-detection AUROC")
ax.legend(fontsize=8)
ax.set_ylim(0.45, 0.7)

ax2 = axes[1]
x = [0, 1]
labels = ["logp_conf\n(1 pass)", "self-consistency\n(K=8)"]
eces = [s["calibration"]["logp_conf = exp(mean_logp) (1 pass)"]["ece"],
        s["calibration"]["self_consistency_conf (K=8 samples)"]["ece"]]
briers = [s["calibration"]["logp_conf = exp(mean_logp) (1 pass)"]["brier"],
          s["calibration"]["self_consistency_conf (K=8 samples)"]["brier"]]
w = 0.35
ax2.bar([i - w / 2 for i in x], eces, width=w, label="ECE")
ax2.bar([i + w / 2 for i in x], briers, width=w, label="Brier")
ax2.set_xticks(x, labels)
ax2.set_title("Calibration quality (lower = better)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results.png", dpi=140)
print("Saved results.png")
