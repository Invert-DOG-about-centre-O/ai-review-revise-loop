import numpy as np
from scipy import stats
import json, time
from followup_experiments import run, SEEDS

if __name__ == "__main__":
    t0 = time.time()

    # Joint robustness test (reviewer question 1): combine sparsity AND shape
    # mismatch in a single external corpus, rather than testing each axis alone.
    joint_results = {}
    for pool_size in [30, 5]:
        for shape in ["wide", "bimodal"]:
            for strategy, kw in [("calibrated", {}), ("sycophantic", {"s": 1.0})]:
                vs, bs, es = [], [], []
                for seed in range(SEEDS):
                    var, bc, ext = run(strategy, seed, pool_source="external",
                                        pool_size=pool_size, corpus_shape=shape, **kw)
                    vs.append(var); bs.append(bc); es.append(ext)
                joint_results[f"{strategy}_pool{pool_size}_{shape}"] = {
                    "var": vs, "bc": bs, "ext": es
                }

    # Significance test for each joint condition
    sig = {}
    for pool_size in [30, 5]:
        for shape in ["wide", "bimodal"]:
            cal = joint_results[f"calibrated_pool{pool_size}_{shape}"]["var"]
            syc = joint_results[f"sycophantic_pool{pool_size}_{shape}"]["var"]
            t, p = stats.ttest_ind(syc, cal, equal_var=False)
            sig[f"pool{pool_size}_{shape}"] = {"t": float(t), "p_raw": float(p)}

    elapsed = time.time() - t0

    with open("joint_results.json", "w") as f:
        json.dump({"joint": joint_results, "sig": sig}, f, indent=2)

    print(f"Joint experiment done in {elapsed:.1f}s")
    for k, d in joint_results.items():
        print(f"{k:32s} var={np.mean(d['var']):.3f}+/-{np.std(d['var']):.3f}  "
              f"bc={np.mean(d['bc']):.3f}+/-{np.std(d['bc']):.3f}")
    print("\nSignificance (sycophantic vs calibrated, variance):")
    for k, v in sig.items():
        print(f"{k:20s} t={v['t']:.3f} p_raw={v['p_raw']:.4g}")
