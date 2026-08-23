"""Robustness ablation: sweep the bounded-confidence threshold (eps) for the
random/calibrated baseline vs. fully sycophantic (s=1.0) recommender."""
import json
import time
import numpy as np
import simulation as sim

def main():
    t0 = time.time()
    eps_values = [0.15, 0.25, 0.35, 0.5, 0.7]
    n_seeds = 10
    out = {}
    for eps in eps_values:
        sim.CONFIDENCE_EPS = eps
        for strategy, param in [("calibrated", 0.0), ("sycophantic", 1.0)]:
            key = f"eps{eps}_{strategy}"
            runs = [sim.run_simulation(strategy, param, seed=1000 + s) for s in range(n_seeds)]
            fv = np.array([r["final_var"] for r in runs])
            fb = np.array([r["final_bc"] for r in runs])
            out[key] = {"eps": eps, "strategy": strategy,
                        "final_var_mean": float(fv.mean()), "final_var_std": float(fv.std(ddof=1)),
                        "final_bc_mean": float(fb.mean()), "final_bc_std": float(fb.std(ddof=1))}
            print(f"eps={eps:.2f} {strategy:>11s}: var={fv.mean():.4f}+-{fv.std(ddof=1):.4f} "
                  f"bc={fb.mean():.4f}+-{fb.std(ddof=1):.4f}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    with open("ablation_eps_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
