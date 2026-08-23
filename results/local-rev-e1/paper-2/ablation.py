"""
Ablation sweeps over (a) the diversity-injection rate and (b) the retrain
frequency, holding everything else fixed at simulate.py's defaults.
Uses fewer seeds than the main experiment to stay within the compute budget.
"""
import json
import time
import numpy as np
import simulate as sim

SEEDS = [0, 1, 2]


def run_diversity_sweep():
    rows = []
    for eps in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        sim.DIVERSITY_EPS = eps
        vals_var, vals_div, vals_extreme = [], [], []
        for seed in SEEDS:
            res = sim.run_condition("adaptive_diverse", seed)
            vals_var.append(res["final_opinion_var"])
            vals_div.append(res["overall_content_diversity"])
            vals_extreme.append(res["final_frac_extreme"])
        rows.append({
            "diversity_eps": eps,
            "final_opinion_var_mean": float(np.mean(vals_var)),
            "final_opinion_var_std": float(np.std(vals_var)),
            "overall_content_diversity_mean": float(np.mean(vals_div)),
            "final_frac_extreme_mean": float(np.mean(vals_extreme)),
        })
        print(f"eps={eps:.1f} var={rows[-1]['final_opinion_var_mean']:.4f} "
              f"diversity={rows[-1]['overall_content_diversity_mean']:.3f} "
              f"frac_extreme={rows[-1]['final_frac_extreme_mean']:.3f}")
    sim.DIVERSITY_EPS = 0.3  # restore default
    return rows


def run_retrain_freq_sweep():
    rows = []
    for freq in [1, 2, 4, 10, 40]:
        sim.RETRAIN_EVERY = freq
        vals_var, vals_div, vals_extreme = [], [], []
        for seed in SEEDS:
            res = sim.run_condition("adaptive", seed)
            vals_var.append(res["final_opinion_var"])
            vals_div.append(res["overall_content_diversity"])
            vals_extreme.append(res["final_frac_extreme"])
        rows.append({
            "retrain_every": freq,
            "final_opinion_var_mean": float(np.mean(vals_var)),
            "final_opinion_var_std": float(np.std(vals_var)),
            "overall_content_diversity_mean": float(np.mean(vals_div)),
            "final_frac_extreme_mean": float(np.mean(vals_extreme)),
        })
        print(f"retrain_every={freq:2d} var={rows[-1]['final_opinion_var_mean']:.4f} "
              f"diversity={rows[-1]['overall_content_diversity_mean']:.3f} "
              f"frac_extreme={rows[-1]['final_frac_extreme_mean']:.3f}")
    sim.RETRAIN_EVERY = 4  # restore default
    return rows


def main():
    t0 = time.time()
    print("=== Diversity-injection sweep (adaptive_diverse condition) ===")
    div_rows = run_diversity_sweep()
    print("\n=== Retrain-frequency sweep (adaptive condition) ===")
    freq_rows = run_retrain_freq_sweep()
    with open("ablation_results.json", "w") as f:
        json.dump({"diversity_sweep": div_rows, "retrain_freq_sweep": freq_rows}, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
