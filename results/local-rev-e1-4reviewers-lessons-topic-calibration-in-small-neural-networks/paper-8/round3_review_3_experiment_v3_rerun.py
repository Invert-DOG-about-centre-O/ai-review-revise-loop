"""
Reviewer re-run of experiment_v3.py, writing to a separate output file so as
not to clobber the original results_v3.json.
"""
import json
import time
import numpy as np
from scipy import stats
from experiment_v2 import run_instance, D, NUM_CLASSES

if __name__ == "__main__":
    t0 = time.time()
    seeds = list(range(10))

    dense_widths = [8, 32]
    bayes_acc, res_dense = run_instance(data_seed=0, widths=dense_widths, seeds=seeds, ls_widths=[])
    print(f"[{time.time()-t0:.1f}s] dense width sweep done")

    bayes_acc_l, res_w2_long = run_instance(data_seed=0, widths=[2], seeds=seeds, ls_widths=[], epochs=400)
    print(f"[{time.time()-t0:.1f}s] width-2 long-training check done")

    out = dict(
        dense=dict(widths=dense_widths, seeds=seeds, results=res_dense),
        w2_long=dict(width=2, epochs=400, seeds=seeds, results=res_w2_long),
        total_time_s=time.time() - t0,
    )
    with open("round3_review_3_results_v3_rerun.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s")

    T_dense = {}
    for w in dense_widths:
        vals = [r["T_star"] for r in res_dense if r["width"] == w]
        T_dense[w] = (np.mean(vals), np.std(vals))
    print("Dense widths T*:", T_dense)

    vals_w2_400 = [r["T_star"] for r in res_w2_long if r["width"] == 2]
    print("Width 2, 400 epochs T*: mean=%.3f sd=%.3f" % (np.mean(vals_w2_400), np.std(vals_w2_400)))

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

    # Also print w2-400 vs w2-80 test, since paper reports p=0.29
    vals_w2_80 = T_by_width[2]
    tstat, pval = stats.ttest_ind(vals_w2_80, vals_w2_400, equal_var=False)
    print(f"\nWidth 2: 80ep vs 400ep: mean80={np.mean(vals_w2_80):.3f} mean400={np.mean(vals_w2_400):.3f} t={tstat:.2f} p={pval:.3f}")

    ls_res = [r for r in main_res if r["label_smoothing"] == 0.1]
    noLS_res = [r for r in main_res if r["label_smoothing"] == 0.0]
    print("\nECE fold-change under label smoothing, per width:")
    for w in widths_main:
        ece_noLS = np.mean([r["ece"] for r in noLS_res if r["width"] == w])
        ece_LS = np.mean([r["ece"] for r in ls_res if r["width"] == w])
        print(f"  width {w}: {ece_noLS:.4f} -> {ece_LS:.4f}  ({ece_LS/ece_noLS:.1f}x)")
