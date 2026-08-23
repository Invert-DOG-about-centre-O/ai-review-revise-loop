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

def make_external_pool(rng, pool_size, shape):
    if shape == "same":
        return np.clip(rng.normal(0, 0.25, pool_size), -1, 1)
    if shape == "wide":
        return np.clip(rng.normal(0, 0.6, pool_size), -1, 1)
    if shape == "bimodal":
        half = pool_size // 2
        rest = pool_size - half
        a = rng.normal(-0.6, 0.15, half)
        b = rng.normal(0.6, 0.15, rest)
        return np.clip(np.concatenate([a, b]), -1, 1)
    raise ValueError(shape)

def run(strategy, seed, s=0.0, b=0.5, pool_source="self", pool_size=None, eps=EPS, corpus_shape="same"):
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(0, 0.25, N), -1, 1)
    if pool_source == "external":
        ext_pool = make_external_pool(rng, pool_size or N, corpus_shape)
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

if __name__ == "__main__":
    t0 = time.time()

    # Experiment A: does freezing survive when external corpus is MORE dispersed / bimodal
    # than the agents' own initial distribution (reviewer question 2)?
    corpus_results = {}
    for shape in ["same", "wide", "bimodal"]:
        for strategy, kw in [("calibrated", {}), ("sycophantic", {"s": 1.0})]:
            vs, bs, es = [], [], []
            for seed in range(SEEDS):
                var, bc, ext = run(strategy, seed, pool_source="external", pool_size=300,
                                    corpus_shape=shape, **kw)
                vs.append(var); bs.append(bc); es.append(ext)
            corpus_results[f"{strategy}_{shape}"] = {"var": vs, "bc": bs, "ext": es}

    # Experiment B: is sycophantic(0.75) vs calibrated on variance a power issue?
    # Re-run with 60 seeds instead of 20 (reviewer question 3).
    power_results = {}
    N_SEEDS_POWER = 60
    for strategy, kw in [("calibrated", {}), ("sycophantic", {"s": 0.75})]:
        vs, bs, es = [], [], []
        for seed in range(N_SEEDS_POWER):
            var, bc, ext = run(strategy, seed, **kw)
            vs.append(var); bs.append(bc); es.append(ext)
        power_results[strategy] = {"var": vs, "bc": bs, "ext": es}
    t_var, p_var = stats.ttest_ind(power_results["sycophantic"]["var"], power_results["calibrated"]["var"], equal_var=False)
    power_results["_test"] = {"n_seeds": N_SEEDS_POWER, "t_var": float(t_var), "p_var_raw": float(p_var)}

    elapsed = time.time() - t0

    with open("followup_results.json", "w") as f:
        json.dump({"corpus_shape": corpus_results, "power_075": power_results}, f, indent=2)

    print(f"Follow-up experiments done in {elapsed:.1f}s")
    print("\n=== Corpus-shape robustness (external pool, size=300) ===")
    for k, d in corpus_results.items():
        print(f"{k:28s} var={np.mean(d['var']):.3f}+/-{np.std(d['var']):.3f}  "
              f"bc={np.mean(d['bc']):.3f}+/-{np.std(d['bc']):.3f}")

    print("\n=== Power check: sycophantic(0.75) vs calibrated, variance, n_seeds=%d ===" % N_SEEDS_POWER)
    print(f"calibrated   var={np.mean(power_results['calibrated']['var']):.3f}+/-{np.std(power_results['calibrated']['var']):.3f}")
    print(f"syco(0.75)   var={np.mean(power_results['sycophantic']['var']):.3f}+/-{np.std(power_results['sycophantic']['var']):.3f}")
    print(f"t={power_results['_test']['t_var']:.3f} p_raw={power_results['_test']['p_var_raw']:.4g}")
