import numpy as np
from scipy import stats
import json, time

N = 300
T = 150
ALPHA = 0.3
GAMMA = 0.05
EPS = 0.35
SEEDS = 20

def bimodality_coeff(x):
    n = len(x)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)
    return (skew**2 + 1) / (kurt + 3*(n-1)**2/((n-2)*(n-3)))

def run(strategy, seed, s=0.0, b=0.5, pool_source="self", pool_size=None, eps=EPS):
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(0, 0.25, N), -1, 1)
    if pool_source == "external":
        ext_pool = np.clip(rng.normal(0, 0.25, pool_size or N), -1, 1)
    for t in range(T):
        pool = ext_pool if pool_source == "external" else x
        if strategy in ("random", "calibrated"):
            idx = rng.integers(0, len(pool), N)
            r = pool[idx]
        elif strategy == "sycophantic":
            do_syco = rng.random(N) < s
            idx_rand = rng.integers(0, len(pool), N)
            r_rand = pool[idx_rand]
            nearest_idx = np.argmin(np.abs(pool[None, :] - x[:, None]), axis=1)
            r_near = pool[nearest_idx]
            r = np.where(do_syco, r_near, r_rand)
        elif strategy == "bridging":
            do_bridge = rng.random(N) < b
            idx_rand = rng.integers(0, len(pool), N)
            r_rand = pool[idx_rand]
            target = -x
            nearest_idx = np.argmin(np.abs(pool[None, :] - target[:, None]), axis=1)
            r_opp = pool[nearest_idx]
            r = np.where(do_bridge, r_opp, r_rand)
        else:
            raise ValueError(strategy)
        diff = r - x
        within = np.abs(diff) <= eps
        x_new = np.where(within, x + ALPHA * diff, x - GAMMA * np.sign(diff) * np.abs(diff))
        x = np.clip(x_new, -1, 1)
    var = np.var(x)
    bc = bimodality_coeff(x)
    ext = np.mean(np.abs(x))
    return var, bc, ext

def sweep(strategy, param_name, param_vals, seeds=SEEDS, **kwargs):
    out = {}
    for pv in param_vals:
        vs, bs, es = [], [], []
        for seed in range(seeds):
            k = dict(kwargs)
            k[param_name] = pv
            var, bc, ext = run(strategy, seed, **k)
            vs.append(var); bs.append(bc); es.append(ext)
        out[str(pv)] = {"var": vs, "bc": bs, "ext": es}
    return out

if __name__ == "__main__":
    t0 = time.time()
    results = {}

    results["calibrated"] = sweep("calibrated", "s", [0.0])  # placeholder param, unused
    results["bridging"] = sweep("bridging", "b", [0.5])
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        vs, bs, es = [], [], []
        for seed in range(SEEDS):
            var, bc, ext = run("sycophantic", seed, s=s)
            vs.append(var); bs.append(bc); es.append(ext)
        results[f"sycophantic_{s}"] = {str(s): {"var": vs, "bc": bs, "ext": es}}

    # Pairwise significance tests: every condition vs calibrated, all 3 metrics, Bonferroni corrected
    calib = results["calibrated"]["0.0"]
    conditions = ["bridging", "sycophantic_0.0", "sycophantic_0.25", "sycophantic_0.5",
                  "sycophantic_0.75", "sycophantic_1.0"]
    metrics = ["var", "bc", "ext"]
    n_tests = len(conditions) * len(metrics)
    sig_tests = {}
    for cond in conditions:
        key = list(results[cond].keys())[0]
        data = results[cond][key]
        sig_tests[cond] = {}
        for m in metrics:
            tstat, p = stats.ttest_ind(data[m], calib[m], equal_var=False)
            p_corr = min(p * n_tests, 1.0)
            sig_tests[cond][m] = {"t": float(tstat), "p_raw": float(p), "p_bonferroni": float(p_corr)}

    # Decoupled / sparse-pool experiment: does freezing survive when content source
    # is NOT the same population (external fixed corpus), and when the pool is sparse?
    decouple_results = {}
    for pool_size in [300, 30, 5]:
        for strategy, kw in [("calibrated", {}), ("sycophantic", {"s": 1.0})]:
            vs, bs, es = [], [], []
            for seed in range(SEEDS):
                var, bc, ext = run(strategy, seed, pool_source="external", pool_size=pool_size, **kw)
                vs.append(var); bs.append(bc); es.append(ext)
            decouple_results[f"{strategy}_pool{pool_size}"] = {"var": vs, "bc": bs, "ext": es}

    elapsed_main = time.time() - t0

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("sig_tests.json", "w") as f:
        json.dump(sig_tests, f, indent=2)
    with open("decouple_results.json", "w") as f:
        json.dump(decouple_results, f, indent=2)

    print(f"Main sweep + sig tests + decouple experiment done in {elapsed_main:.1f}s")
    print("\n=== Table 1 (mean +/- SD) ===")
    for cond in ["calibrated"] + conditions:
        key = list(results[cond].keys())[0]
        d = results[cond][key]
        print(f"{cond:20s} var={np.mean(d['var']):.3f}+/-{np.std(d['var']):.3f}  "
              f"bc={np.mean(d['bc']):.3f}+/-{np.std(d['bc']):.3f}  "
              f"ext={np.mean(d['ext']):.3f}+/-{np.std(d['ext']):.3f}")

    print("\n=== Significance vs calibrated (Welch t, Bonferroni-corrected p, n_tests=%d) ===" % n_tests)
    for cond in conditions:
        for m in metrics:
            r = sig_tests[cond][m]
            print(f"{cond:20s} {m:4s} t={r['t']:.2f} p_corr={r['p_bonferroni']:.4g}")

    print("\n=== Decoupled / sparse external-pool experiment (var mean+/-SD) ===")
    for k, d in decouple_results.items():
        print(f"{k:30s} var={np.mean(d['var']):.3f}+/-{np.std(d['var']):.3f}  "
              f"bc={np.mean(d['bc']):.3f}+/-{np.std(d['bc']):.3f}")
