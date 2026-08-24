"""
Round-2 review question 3: does the weak alpha effect depend on influence
being similarity-gated (bounded confidence)? Test a second update rule where
the SAME exposure-biased partner sampling is used, but influence is NOT gated
by epsilon -- every sampled partner exerts mu*(x_j-x_i) influence regardless
of distance. If alpha's effect on variance grows much larger here than in the
gated (Deffuant-Weisbuch) model, that confirms the gating mechanism (not the
exposure-bias mechanism itself) is what mutes alpha's effect.
"""
import numpy as np
import json
import time
import csv
from experiment import mixing_mask, small_world_mask, write_csv

t0 = time.time()


def run_ungated(N, T, alpha, mu, topology, seed, k=10, p_rewire=0.1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=N)
    if topology == "mixing":
        reach = mixing_mask(N)
    else:
        reach = small_world_mask(N, k=k, p=p_rewire, rng=rng)

    same_side_count = 0
    opp_side_count = 0
    for t in range(T):
        diff = np.abs(x[:, None] - x[None, :])
        w = np.exp(alpha * (1 - diff / 2.0))
        w = np.where(reach, w, 0.0)
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = w / row_sums
        cum = np.cumsum(probs, axis=1)
        r = rng.random(size=(N, 1))
        partner = (cum < r).sum(axis=1)
        partner = np.clip(partner, 0, N - 1)

        active = np.abs(x) > 0.05
        same = (np.sign(x[active]) == np.sign(x[partner][active]))
        same_side_count += int(same.sum())
        opp_side_count += int((~same).sum())

        # NO epsilon gate: every sampled partner influences the focal agent.
        d = x[partner] - x
        x = np.clip(x + mu * d, -1, 1)

    ratio = same_side_count / max(opp_side_count, 1)
    return {"variance": float(np.var(x)), "amp_ratio": ratio}


def sweep_ungated(alphas, topologies, seeds, N=200, T=150, mu=0.3):
    rows = []
    for topology in topologies:
        for alpha in alphas:
            vs, rs = [], []
            for seed in seeds:
                out = run_ungated(N, T, alpha, mu, topology, seed)
                vs.append(out["variance"])
                rs.append(out["amp_ratio"])
            rows.append({
                "topology": topology, "alpha": alpha, "n_seeds": len(seeds),
                "variance_mean": float(np.mean(vs)), "variance_std": float(np.std(vs)),
                "amp_ratio_mean": float(np.mean(rs)),
            })
            print(f"[ungated] topology={topology} alpha={alpha:.2f} "
                  f"var={rows[-1]['variance_mean']:.4f} amp={rows[-1]['amp_ratio_mean']:.3f}")
    return rows


if __name__ == "__main__":
    alphas = [round(a, 2) for a in np.arange(0.0, 6.01, 0.5)]
    rows = sweep_ungated(alphas, ["mixing", "smallworld"], seeds=[1, 2, 3, 4, 5], N=200, T=150, mu=0.3)
    write_csv(rows, "results_ungated.csv")

    summary = {}
    for topology in ["mixing", "smallworld"]:
        sub = sorted([r for r in rows if r["topology"] == topology], key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in sub])
        v = np.array([r["variance_mean"] for r in sub])
        r_obs = float(np.corrcoef(a, v)[0, 1])
        summary[topology] = {
            "r_alpha_variance": round(r_obs, 4),
            "variance_range": round(float(v.max() - v.min()), 4),
            "variance_at_alpha0": round(float(v[0]), 4),
            "variance_at_alpha6": round(float(v[-1]), 4),
        }
    with open("ungated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"TOTAL ungated experiment time: {time.time()-t0:.1f}s")
