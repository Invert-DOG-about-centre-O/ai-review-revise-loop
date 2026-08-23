import numpy as np, json, time
from simulation import run

SEEDS = 10

if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for eps in [0.15, 0.25, 0.35, 0.5, 0.7]:
        for strategy, kw in [("calibrated", {}), ("sycophantic", {"s": 1.0})]:
            vs, bs = [], []
            for seed in range(SEEDS):
                var, bc, ext = run(strategy, seed, eps=eps, **kw)
                vs.append(var); bs.append(bc)
            out[f"{strategy}_eps{eps}"] = {"var": vs, "bc": bs}
    with open("ablation_eps_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Ablation done in {time.time()-t0:.1f}s")
    for eps in [0.15, 0.25, 0.35, 0.5, 0.7]:
        c = out[f"calibrated_eps{eps}"]
        s = out[f"sycophantic_eps{eps}"]
        print(f"eps={eps:.2f}  calib var={np.mean(c['var']):.3f}+/-{np.std(c['var']):.3f} bc={np.mean(c['bc']):.3f}  |  "
              f"syco var={np.mean(s['var']):.3f}+/-{np.std(s['var']):.3f} bc={np.mean(s['bc']):.3f}")
