"""Sensitivity check: does score-choice-dominates ranking hold under a different
noise/overlap regime (lower class overlap, no label noise -> should push the base
model toward overconfidence instead of underconfidence)?"""
import json
import numpy as np
import experiment as E

E.N_SEEDS = 5


def run_regime(sigma, label_noise, tag):
    E.NOISE_SIGMA = sigma
    E.LABEL_NOISE = label_noise
    runs = [E.run_seed(seed) for seed in range(5)]
    methods = ["raw_softmax", "temp_scaled", "ensemble", "ensemble_temp_scaled"]
    summary = {}
    for m in methods:
        ece_vals = [r[m]["ece"] for r in runs]
        T_vals = [r[m]["T"] for r in runs]
        lac = [r[m]["LAC"]["avg_set_size"] for r in runs]
        aps = [r[m]["APS"]["avg_set_size"] for r in runs]
        summary[m] = {
            "ece_mean": float(np.mean(ece_vals)),
            "T_mean": float(np.mean(T_vals)),
            "lac_mean": float(np.mean(lac)),
            "aps_mean": float(np.mean(aps)),
        }
    lac_spread = max(summary[m]["lac_mean"] for m in methods) - min(summary[m]["lac_mean"] for m in methods)
    aps_spread = max(summary[m]["aps_mean"] for m in methods) - min(summary[m]["aps_mean"] for m in methods)
    score_gap = np.mean([summary[m]["aps_mean"] - summary[m]["lac_mean"] for m in methods])
    print(f"[{tag}] sigma={sigma} label_noise={label_noise}")
    for m in methods:
        s = summary[m]
        print(f"  {m}: ECE={s['ece_mean']:.4f} T={s['T_mean']:.3f} LAC={s['lac_mean']:.4f} APS={s['aps_mean']:.4f}")
    print(f"  LAC spread across calib methods: {lac_spread:.4f}; APS spread: {aps_spread:.4f}; "
          f"mean LAC->APS score-switch gap: {score_gap:.4f}")
    return {"tag": tag, "sigma": sigma, "label_noise": label_noise, "summary": summary,
            "lac_spread": lac_spread, "aps_spread": aps_spread, "score_gap": score_gap}


if __name__ == "__main__":
    out = []
    out.append(run_regime(1.8, 0.12, "original"))
    out.append(run_regime(0.8, 0.0, "low_overlap_no_label_noise"))
    with open("sensitivity_results.json", "w") as f:
        json.dump(out, f, indent=2)
