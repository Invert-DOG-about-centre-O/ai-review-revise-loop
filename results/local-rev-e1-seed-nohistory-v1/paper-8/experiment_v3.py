"""
Follow-up experiments for round-2 review of the AI-leakage paper.

Addresses the three remaining review weaknesses:
  (a) broadcast control was only 5 seeds and not paired-t-tested vs baseline
      -> re-run broadcast at 20 seeds with a paired t-test vs baseline.
  (b) most of Table 1 (11 conditions) still rests on 5 seeds only
      -> re-run the full 11-condition sweep at 20 seeds so every row has the
         same statistical footing, and report per-row SD.
  (c) no informal hypothesis for why alpha~=0.6 minimizes the personalized
      gap -> not a code question, addressed in prose in the paper, but we
      additionally record the *rate* of the U-shape's within-run advisor
      drift to support that discussion (advisor state variance at each alpha).
"""
import json
import time
import numpy as np
from scipy import stats

from experiment_v2 import run_many, RNG_SEED, N, N_PER_COMMUNITY  # reuse identical mechanics


def main():
    t0 = time.time()
    out = {}

    # (a) broadcast vs baseline, 20 seeds, paired t-test
    n_seeds = 20
    baseline_gaps, _ = run_many("none", 0.0, n_seeds)
    broadcast_gaps, broadcast_wvars = run_many("broadcast", 0.0, n_seeds)
    t_b, p_b = stats.ttest_rel(baseline_gaps, broadcast_gaps)
    out["broadcast_20seed"] = {
        "gap_mean": float(broadcast_gaps.mean()), "gap_std": float(broadcast_gaps.std()),
        "within_var_mean": float(broadcast_wvars.mean()),
        "leakage_index": float(baseline_gaps.mean() - broadcast_gaps.mean()),
        "baseline_gap_mean_20seed": float(baseline_gaps.mean()),
        "paired_t_vs_baseline": float(t_b), "paired_p_vs_baseline": float(p_b),
        "n_seeds": n_seeds,
    }
    print("broadcast 20-seed vs baseline:", out["broadcast_20seed"])

    # (b) full 11-condition sweep at 20 seeds (same conditions as original Table 1)
    conditions = [("none", 0.0)] + [("personalized", a) for a in [0.0, 0.3, 0.6, 0.9, 1.0]] \
        + [("shared", a) for a in [0.0, 0.3, 0.6, 0.9, 1.0]]
    table = {}
    for mode, alpha in conditions:
        gaps, wvars = run_many(mode, alpha, n_seeds)
        key = "none" if mode == "none" else f"{mode}_a{alpha}"
        table[key] = {
            "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
            "within_var_mean": float(wvars.mean()),
            "leakage_index": float(baseline_gaps.mean() - gaps.mean()),
        }
        print(key, table[key])
    out["full_sweep_20seed"] = table

    out["wall_time_s"] = time.time() - t0
    with open("results_v3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
