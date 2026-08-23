"""
Follow-up experiments for round-3 review of the AI-leakage paper.

Addresses the three round-3 weaknesses/questions:
  (a) broadcast-vs-baseline p=0.053 was right at the significance boundary
      -> re-run at 60 seeds (vs 20) to sharpen the estimate (review Q1).
  (b) most Table-1 rows lacked their own paired significance test vs baseline
      -> add a per-row paired t-test (same seeds as baseline) for the full
      11-condition, 20-seed sweep (review weakness 4).
  (c) scope limitations (esp. zero inter-community edges) untested across
      3 revisions -> cheapest one to test given sub-minute runtimes: sweep
      inter-community edge probability away from exactly 0 and see whether
      the leakage effect (shared alpha=0.9 vs baseline) and the broadcast
      null result survive weak direct connectivity (review Q3).
"""
import json
import time
import numpy as np
from scipy import stats

import experiment_v2 as ev2
from experiment_v2 import run_many, RNG_SEED, N, N_PER_COMMUNITY


def main():
    t0 = time.time()
    out = {}

    # (a) broadcast vs baseline, 60 seeds, paired t-test (sharpening round-3's 20-seed p=0.053)
    n_seeds_big = 60
    baseline_gaps_60, _ = run_many("none", 0.0, n_seeds_big)
    broadcast_gaps_60, broadcast_wvars_60 = run_many("broadcast", 0.0, n_seeds_big)
    t_b, p_b = stats.ttest_rel(baseline_gaps_60, broadcast_gaps_60)
    out["broadcast_60seed"] = {
        "gap_mean": float(broadcast_gaps_60.mean()), "gap_std": float(broadcast_gaps_60.std()),
        "baseline_gap_mean": float(baseline_gaps_60.mean()),
        "leakage_index": float(baseline_gaps_60.mean() - broadcast_gaps_60.mean()),
        "paired_t_vs_baseline": float(t_b), "paired_p_vs_baseline": float(p_b),
        "n_seeds": n_seeds_big,
    }
    print("broadcast 60-seed vs baseline:", out["broadcast_60seed"])

    # (b) full 11-condition sweep at 20 seeds, WITH a per-row paired t-test vs baseline
    n_seeds = 20
    baseline_gaps_20, _ = run_many("none", 0.0, n_seeds)
    conditions = [("personalized", a) for a in [0.0, 0.3, 0.6, 0.9, 1.0]] \
        + [("shared", a) for a in [0.0, 0.3, 0.6, 0.9, 1.0]]
    table = {}
    for mode, alpha in conditions:
        gaps, wvars = run_many(mode, alpha, n_seeds)
        t_stat, p_val = stats.ttest_rel(baseline_gaps_20, gaps)
        key = f"{mode}_a{alpha}"
        table[key] = {
            "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
            "leakage_index": float(baseline_gaps_20.mean() - gaps.mean()),
            "paired_t_vs_baseline": float(t_stat), "paired_p_vs_baseline": float(p_val),
        }
        print(key, table[key])
    out["full_sweep_20seed_with_tests"] = table
    out["baseline_20seed"] = {"gap_mean": float(baseline_gaps_20.mean()), "gap_std": float(baseline_gaps_20.std())}

    # (c) inter-community edge probability robustness: does leakage survive
    # weak *direct* connectivity between communities (not just via the AI)?
    n_seeds_c = 10
    inter_ps = [0.0, 0.01, 0.05]
    inter_results = {}
    orig_inter_p = ev2.INTER_P
    try:
        for ip in inter_ps:
            ev2.INTER_P = ip
            base_g, _ = run_many("none", 0.0, n_seeds_c)
            shared_g, _ = run_many("shared", 0.9, n_seeds_c)
            broad_g, _ = run_many("broadcast", 0.0, n_seeds_c)
            inter_results[str(ip)] = {
                "baseline_gap_mean": float(base_g.mean()),
                "shared_a0.9_gap_mean": float(shared_g.mean()),
                "shared_a0.9_leakage": float(base_g.mean() - shared_g.mean()),
                "broadcast_gap_mean": float(broad_g.mean()),
                "broadcast_leakage": float(base_g.mean() - broad_g.mean()),
                "n_seeds": n_seeds_c,
            }
            print(f"inter_p={ip}:", inter_results[str(ip)])
    finally:
        ev2.INTER_P = orig_inter_p
    out["inter_community_edge_robustness"] = inter_results

    out["wall_time_s"] = time.time() - t0
    with open("results_v4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
