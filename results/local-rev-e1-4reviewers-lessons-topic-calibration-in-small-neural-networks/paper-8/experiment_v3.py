"""
v3 additions addressing round-2 review feedback:
  - denser width grid (add 8, 32) to pin down the T*=1 crossing point instead
    of extrapolating between width 4 and width 16
  - a convergence check at width 2: train 5x longer (400 vs 80 epochs) to test
    whether underconfidence is a capacity effect or just undertraining
  - proper Welch's t-test (not just CI-overlap eyeballing) between adjacent
    widths' T* distributions
  - t-distribution (not normal) critical value for 95% CIs at n=10
"""
import json
import time
import numpy as np
from scipy import stats
from experiment_v2 import run_instance, D, NUM_CLASSES

if __name__ == "__main__":
    t0 = time.time()
    seeds = list(range(10))

    # 1. Denser width grid to pin the crossover
    dense_widths = [8, 32]
    bayes_acc, res_dense = run_instance(data_seed=0, widths=dense_widths, seeds=seeds, ls_widths=[])
    print(f"[{time.time()-t0:.1f}s] dense width sweep done")

    # 2. Convergence check: width 2, 400 epochs instead of 80
    from experiment_v2 import run_instance as ri
    _, res_long = run_instance(data_seed=0, widths=[], seeds=[], ls_widths=[])  # no-op warm import
    # directly call with epochs override
    bayes_acc_l, res_w2_long = run_instance(data_seed=0, widths=[2], seeds=seeds, ls_widths=[], epochs=400)
    print(f"[{time.time()-t0:.1f}s] width-2 long-training check done")

    out = dict(
        dense=dict(widths=dense_widths, seeds=seeds, results=res_dense),
        w2_long=dict(width=2, epochs=400, seeds=seeds, results=res_w2_long),
        total_time_s=time.time() - t0,
    )
    with open("results_v3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s")

    # Print quick summary
    T_dense = {}
    for w in dense_widths:
        vals = [r["T_star"] for r in res_dense if r["width"] == w]
        T_dense[w] = (np.mean(vals), np.std(vals))
    print("Dense widths T*:", T_dense)

    vals_w2_80 = None  # filled from results_v2.json below
    vals_w2_400 = [r["T_star"] for r in res_w2_long if r["width"] == 2]
    print("Width 2, 400 epochs T*: mean=%.3f sd=%.3f" % (np.mean(vals_w2_400), np.std(vals_w2_400)))

    # Load v2 results for comparisons (t-tests between adjacent widths, t-based CI)
    with open("results_v2.json") as f:
        v2 = json.load(f)
    main_res = v2["main"]["results"]
    widths_main = [2, 4, 16, 64, 256]
    T_by_width = {w: [r["T_star"] for r in main_res if r["width"] == w and r["label_smoothing"] == 0.0] for w in widths_main}

    print("\nWelch t-tests between adjacent widths (T*):")
    ordered = [2, 4, 8, 16, 32, 64, 256]
    all_T = dict(T_by_width)
    all_T[8] = [r["T_star"] for r in res_dense if r["width"] == 8]
    all_T[32] = [r["T_star"] for r in res_dense if r["width"] == 32]
    for a, b in zip(ordered[:-1], ordered[1:]):
        tstat, pval = stats.ttest_ind(all_T[a], all_T[b], equal_var=False)
        print(f"  width {a} vs {b}: t={tstat:.2f}, p={pval:.2e}")

    print("\nt-distribution 95%% CI half-widths (n=10, t_crit=%.3f):" % stats.t.ppf(0.975, df=9))
    for w in widths_main:
        vals = T_by_width[w]
        sd = np.std(vals, ddof=1)
        half = stats.t.ppf(0.975, df=9) * sd / np.sqrt(len(vals))
        print(f"  width {w}: mean={np.mean(vals):.3f} sd={sd:.3f} half-width={half:.3f}")

    # Fix the ECE-under-LS fold-change claim (reviewer 2 caught "6-8x" error)
    ls_res = [r for r in main_res if r["label_smoothing"] == 0.1]
    noLS_res = [r for r in main_res if r["label_smoothing"] == 0.0]
    print("\nECE fold-change under label smoothing, per width:")
    for w in widths_main:
        ece_noLS = np.mean([r["ece"] for r in noLS_res if r["width"] == w])
        ece_LS = np.mean([r["ece"] for r in ls_res if r["width"] == w])
        print(f"  width {w}: {ece_noLS:.4f} -> {ece_LS:.4f}  ({ece_LS/ece_noLS:.1f}x)")
