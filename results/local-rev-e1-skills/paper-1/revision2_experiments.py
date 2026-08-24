"""
Round-3 revision experiments addressing round-3 review questions:
 (1) Denser N sweep (N in {50,75,100,150,200}) at alpha in {0,6}, both epsilons,
     mixing topology, to locate where/whether the epsilon-dominance margin at
     eps=0.35 shrinks or reverses as N shrinks (flagged: N=100/eps=0.35 nearly
     erased the margin -- is this a boundary case or a trend?).
 (2) Calibration cutoff sensitivity: recompute the amplification ratio with
     |x_i| cutoffs of 0.0, 0.05 (main-paper default), and 0.15, and re-find the
     calibrated alpha at each, to see how much the headline comparison moves.
 (3) Partial-gating model: replace the hard epsilon cutoff with a smooth
     logistic gate of the same effective width, and rerun the alpha sweep, to
     test whether a strong alpha effect emerges between the "no gate" (ungated
     ablation, already in the paper: consensus, alpha irrelevant) and "hard
     gate" (main model: weak alpha) extremes.
"""
import csv
import json
import time
import numpy as np
from experiment import mixing_mask, small_world_mask, write_csv

t0 = time.time()


def run_sim_cutoff_variants(N, T, alpha, epsilon, mu, topology, seed, cutoffs):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=N)
    reach = mixing_mask(N) if topology == "mixing" else small_world_mask(N, rng=rng)
    same_ct = {c: 0 for c in cutoffs}
    opp_ct = {c: 0 for c in cutoffs}
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
        for c in cutoffs:
            active = np.abs(x) > c
            same = (np.sign(x[active]) == np.sign(x[partner][active]))
            same_ct[c] += int(same.sum())
            opp_ct[c] += int((~same).sum())
        d = x[partner] - x
        within = np.abs(d) < epsilon
        update = np.where(within, mu * d, 0.0)
        x = np.clip(x + update, -1, 1)
    return {c: same_ct[c] / max(opp_ct[c], 1) for c in cutoffs}, float(np.var(x))


def run_sim_soft_gate(N, T, alpha, epsilon, mu, topology, seed, steepness):
    """Same as run_simulation but with a logistic soft gate instead of a hard
    |d|<epsilon cutoff: influence weight = sigmoid(steepness*(epsilon-|d|)),
    so it's ~1 well inside epsilon, ~0 well outside, with a smooth transition
    of controllable width (steepness) instead of a discontinuity."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=N)
    reach = mixing_mask(N) if topology == "mixing" else small_world_mask(N, rng=rng)
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
        d = x[partner] - x
        gate = 1.0 / (1.0 + np.exp(-steepness * (epsilon - np.abs(d))))
        update = gate * mu * d
        x = np.clip(x + update, -1, 1)
    return float(np.var(x))


# ---------------------------------------------------------------------------
# (1) Denser N sweep
# ---------------------------------------------------------------------------
N_values = [50, 75, 100, 150, 200]
n_rows = []
for N_val in N_values:
    for epsilon in [0.15, 0.35]:
        for alpha in [0.0, 6.0]:
            variances = []
            for seed in [1, 2, 3, 4, 5]:
                rng = np.random.default_rng(seed)
                x = rng.uniform(-1, 1, size=N_val)
                reach = mixing_mask(N_val)
                for t in range(150):
                    diff = np.abs(x[:, None] - x[None, :])
                    w = np.exp(alpha * (1 - diff / 2.0))
                    w = np.where(reach, w, 0.0)
                    row_sums = w.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    probs = w / row_sums
                    cum = np.cumsum(probs, axis=1)
                    r = rng.random(size=(N_val, 1))
                    partner = (cum < r).sum(axis=1)
                    partner = np.clip(partner, 0, N_val - 1)
                    d = x[partner] - x
                    within = np.abs(d) < epsilon
                    update = np.where(within, 0.3 * d, 0.0)
                    x = np.clip(x + update, -1, 1)
                variances.append(float(np.var(x)))
            n_rows.append({"N": N_val, "epsilon": epsilon, "alpha": alpha,
                            "variance_mean": float(np.mean(variances))})
            print(f"[Nsweep] N={N_val} eps={epsilon} alpha={alpha} var={np.mean(variances):.4f}")
write_csv(n_rows, "results_nsweep.csv")

n_summary = {}
for N_val in N_values:
    for epsilon in [0.15, 0.35]:
        pass
for N_val in N_values:
    v15_0 = [r for r in n_rows if r["N"] == N_val and r["epsilon"] == 0.15 and r["alpha"] == 0.0][0]["variance_mean"]
    v35_0 = [r for r in n_rows if r["N"] == N_val and r["epsilon"] == 0.35 and r["alpha"] == 0.0][0]["variance_mean"]
    v15_6 = [r for r in n_rows if r["N"] == N_val and r["epsilon"] == 0.15 and r["alpha"] == 6.0][0]["variance_mean"]
    v35_6 = [r for r in n_rows if r["N"] == N_val and r["epsilon"] == 0.35 and r["alpha"] == 6.0][0]["variance_mean"]
    eps_shift = abs(v35_0 - v15_0)
    alpha_range_eps35 = abs(v35_6 - v35_0)
    alpha_range_eps15 = abs(v15_6 - v15_0)
    ratio = eps_shift / max(alpha_range_eps35, 1e-6)
    n_summary[f"N={N_val}"] = {
        "epsilon_shift_at_alpha0": round(eps_shift, 4),
        "alpha_range_eps0.35": round(alpha_range_eps35, 4),
        "alpha_range_eps0.15": round(alpha_range_eps15, 4),
        "epsilon_dominance_ratio_eps0.35": round(ratio, 2),
    }
print("[N sweep summary]", json.dumps(n_summary, indent=2))
print(f"N sweep done at {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# (2) Calibration cutoff sensitivity
# ---------------------------------------------------------------------------
cutoffs = [0.0, 0.05, 0.15]
fine_alphas = [round(a, 3) for a in np.arange(0.0, 2.01, 0.1)]
cutoff_rows_by_alpha = {c: [] for c in cutoffs}
for alpha in fine_alphas:
    ratio_acc = {c: [] for c in cutoffs}
    var_acc = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        ratios, var = run_sim_cutoff_variants(200, 150, alpha, 0.35, 0.3, "mixing", seed, cutoffs)
        for c in cutoffs:
            ratio_acc[c].append(ratios[c])
        var_acc.append(var)
    for c in cutoffs:
        cutoff_rows_by_alpha[c].append({"alpha": alpha, "ratio_mean": float(np.mean(ratio_acc[c])),
                                          "variance_mean": float(np.mean(var_acc))})
    print(f"[cutoff] alpha={alpha:.2f} ratios={[round(np.mean(ratio_acc[c]),3) for c in cutoffs]}")

calib_cutoff_summary = {}
alpha0_var = cutoff_rows_by_alpha[0.05][0]["variance_mean"]
for c in cutoffs:
    rows = cutoff_rows_by_alpha[c]
    best = min(rows, key=lambda r: abs(r["ratio_mean"] - 1.4))
    calib_cutoff_summary[f"cutoff_{c}"] = {
        "calibrated_alpha": best["alpha"],
        "achieved_ratio": round(best["ratio_mean"], 4),
        "variance_at_calibration": round(best["variance_mean"], 4),
        "variance_shift_from_alpha0": round(abs(best["variance_mean"] - alpha0_var), 4),
    }
print("[calib cutoff sensitivity]", json.dumps(calib_cutoff_summary, indent=2))
print(f"cutoff sensitivity done at {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# (3) Partial (soft) gating model
# ---------------------------------------------------------------------------
soft_alphas = [round(a, 2) for a in np.arange(0.0, 6.01, 1.0)]
soft_rows = []
for steepness, tag in [(50.0, "steep"), (10.0, "soft"), (3.0, "verysoft")]:
    for epsilon in [0.15, 0.35]:
        variances = []
        for alpha in soft_alphas:
            vs = [run_sim_soft_gate(200, 150, alpha, epsilon, 0.3, "mixing", seed, steepness) for seed in [1, 2, 3, 4, 5]]
            v_mean = float(np.mean(vs))
            soft_rows.append({"steepness_tag": tag, "steepness": steepness, "epsilon": epsilon,
                               "alpha": alpha, "variance_mean": v_mean})
            variances.append(v_mean)
        print(f"[softgate] tag={tag} eps={epsilon} variances={[round(v,4) for v in variances]}")
write_csv(soft_rows, "results_softgate.csv")

soft_summary = {}
for steepness, tag in [(50.0, "steep"), (10.0, "soft"), (3.0, "verysoft")]:
    for epsilon in [0.15, 0.35]:
        sub = sorted([r for r in soft_rows if r["steepness_tag"] == tag and r["epsilon"] == epsilon], key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in sub])
        v = np.array([r["variance_mean"] for r in sub])
        r_obs = float(np.corrcoef(a, v)[0, 1]) if np.std(v) > 1e-12 else 0.0
        soft_summary[f"{tag}_eps{epsilon}"] = {"r": round(r_obs, 4), "range": round(float(v.max() - v.min()), 4),
                                                 "var_at_alpha0": round(float(v[0]), 4)}
print("[softgate summary]", json.dumps(soft_summary, indent=2))
print(f"soft-gate experiment done at {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
out = {
    "n_sweep": n_summary,
    "calibration_cutoff_sensitivity": calib_cutoff_summary,
    "soft_gate": soft_summary,
}
with open("revision2_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"TOTAL revision2 time: {time.time()-t0:.1f}s")
