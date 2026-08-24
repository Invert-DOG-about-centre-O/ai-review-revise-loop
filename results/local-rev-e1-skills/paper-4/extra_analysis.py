"""
Additional experiments for the revision, addressing round-1 review weaknesses:
(A) Firm-level decomposition of the diversity-mandate effect (is the index drop
    mechanically driven by averaging in the myopic firm's low prices, or does the
    Q-learning firm itself price lower?)
(B) Hyperparameter sensitivity of the intervention ranking (alpha, beta) beyond
    the single Calvano-style configuration used in the main comparison.
(C) Seed-count / power sensitivity for the two non-significant interventions,
    using the existing n=25 matched-seed data (no new simulation needed).
"""
import json
import time
import numpy as np
from scipy import stats as sstats
import collusion_sim as cs

PERIODS = 150000
SEEDS25 = list(range(25))
SEEDS_SMALL = list(range(10))

out = {}

# ---------------------------------------------------------------------------
# (A) Firm-level decomposition: baseline vs diversity, per-firm collusion index
# ---------------------------------------------------------------------------
def run_sim_perfirm(seed, condition):
    """Copy of cs.run_sim but returns firm-1-only and firm-2-only tail indices."""
    import random, math
    rng = random.Random(seed)
    M = cs.M
    is_audit = False
    n_opp_states = cs.M
    Q1 = np.zeros((M, n_opp_states, M))
    if condition != "diversity":
        Q2 = np.zeros((M, n_opp_states, M))
    a1 = rng.randrange(M)
    a2 = rng.randrange(M)
    s1_opp, s2_opp = a2, a1
    price_hist = np.zeros(PERIODS, dtype=np.int16)
    price_hist2 = np.zeros(PERIODS, dtype=np.int16)
    for t in range(PERIODS):
        eps = math.exp(-cs.BETA * t)
        if rng.random() < eps:
            act1 = rng.randrange(M)
        else:
            act1 = int(np.argmax(Q1[a1, s1_opp]))
        if condition == "diversity":
            act2 = int(np.argmax(cs.PROFIT_MAT[:, a1]))
        else:
            if rng.random() < eps:
                act2 = rng.randrange(M)
            else:
                act2 = int(np.argmax(Q2[a2, s2_opp]))
        p1_idx, p2_idx = act1, act2
        r1 = cs.PROFIT_MAT[p1_idx, p2_idx]
        r2 = cs.PROFIT_MAT[p2_idx, p1_idx]
        best_next1 = np.max(Q1[p1_idx, p2_idx])
        Q1[a1, s1_opp, act1] += cs.ALPHA * (r1 + cs.DELTA * best_next1 - Q1[a1, s1_opp, act1])
        if condition != "diversity":
            best_next2 = np.max(Q2[p2_idx, p1_idx])
            Q2[a2, s2_opp, act2] += cs.ALPHA * (r2 + cs.DELTA * best_next2 - Q2[a2, s2_opp, act2])
        a1, a2 = p1_idx, p2_idx
        s1_opp, s2_opp = p2_idx, p1_idx
        price_hist[t] = p1_idx
        price_hist2[t] = p2_idx
    tail = PERIODS // 10
    idx1 = cs.collusion_index(cs.PRICE_GRID[price_hist[-tail:]].mean())
    idx2 = cs.collusion_index(cs.PRICE_GRID[price_hist2[-tail:]].mean())
    return idx1, idx2

print("=== (A) Firm-level decomposition (n=25, main market) ===")
t0 = time.time()
decomp = {"baseline": {"firm1": [], "firm2": []}, "diversity": {"firm1": [], "firm2": []}}
for cond in ["baseline", "diversity"]:
    for seed in SEEDS25:
        i1, i2 = run_sim_perfirm(seed, cond)
        decomp[cond]["firm1"].append(i1)
        decomp[cond]["firm2"].append(i2)
    print(cond, "done", time.time() - t0)

for cond in decomp:
    f1 = np.array(decomp[cond]["firm1"])
    f2 = np.array(decomp[cond]["firm2"])
    print(f"{cond}: firm1(learner) mean={f1.mean():.4f}  firm2 mean={f2.mean():.4f}")

t, p = sstats.ttest_rel(np.array(decomp["diversity"]["firm1"]), np.array(decomp["baseline"]["firm1"]))
print(f"Learner-only (firm1) diversity vs baseline: diff={np.array(decomp['diversity']['firm1']).mean()-np.array(decomp['baseline']['firm1']).mean():+.4f} paired-t p={p:.2e}")

out["firm_decomposition"] = {
    "baseline_firm1_mean": float(np.array(decomp["baseline"]["firm1"]).mean()),
    "baseline_firm2_mean": float(np.array(decomp["baseline"]["firm2"]).mean()),
    "diversity_firm1_mean": float(np.array(decomp["diversity"]["firm1"]).mean()),
    "diversity_firm2_mean": float(np.array(decomp["diversity"]["firm2"]).mean()),
    "learner_only_paired_t_p": float(p),
    "learner_only_diff": float(np.array(decomp["diversity"]["firm1"]).mean() - np.array(decomp["baseline"]["firm1"]).mean()),
}

# ---------------------------------------------------------------------------
# (B) Hyperparameter sensitivity: alt ALPHA and alt BETA, n=10 seeds, main market
# ---------------------------------------------------------------------------
print("\n=== (B) Hyperparameter sensitivity (n=10, main market) ===")
CONDS = ["baseline", "diversity", "transparency", "audit_explore", "audit_enforce"]
orig_alpha, orig_delta, orig_beta = cs.ALPHA, cs.DELTA, cs.BETA

configs = {
    "alpha_hi": dict(ALPHA=0.30, DELTA=0.95, BETA=4e-5),
    "beta_slow": dict(ALPHA=0.15, DELTA=0.95, BETA=1e-5),
}

hp_results = {}
for cfg_name, cfg in configs.items():
    cs.ALPHA, cs.DELTA, cs.BETA = cfg["ALPHA"], cfg["DELTA"], cfg["BETA"]
    cfg_out = {}
    base_vals = None
    for cond in CONDS:
        vals = np.array([cs.run_sim(seed=s, periods=PERIODS, condition=cond)["collusion_index_tail"] for s in SEEDS_SMALL])
        cfg_out[cond] = vals
        if cond == "baseline":
            base_vals = vals
    print(f"-- config {cfg_name} (alpha={cfg['ALPHA']}, beta={cfg['BETA']}) --")
    row = {}
    for cond in CONDS:
        vals = cfg_out[cond]
        if cond == "baseline":
            print(f"  baseline mean={vals.mean():.4f}")
            row["baseline"] = float(vals.mean())
            continue
        t, p = sstats.ttest_rel(vals, base_vals)
        p_corr = min(p * 4, 1.0)
        print(f"  {cond:15s} mean={vals.mean():.4f} diff={vals.mean()-base_vals.mean():+.4f} p={p:.2e} bonf={p_corr:.2e}")
        row[cond] = {"mean": float(vals.mean()), "diff": float(vals.mean() - base_vals.mean()), "p": float(p), "p_bonf": float(p_corr)}
    hp_results[cfg_name] = row
    print(f"  elapsed {time.time()-t0:.1f}s")

cs.ALPHA, cs.DELTA, cs.BETA = orig_alpha, orig_delta, orig_beta
out["hyperparam_sensitivity"] = hp_results

# ---------------------------------------------------------------------------
# (C) Seed-count / power sensitivity using existing n=25 results (no new sim)
# ---------------------------------------------------------------------------
print("\n=== (C) Power vs seed count (subsampled from existing n=25 data) ===")
with open("results.json") as f:
    res = json.load(f)

def by_cond(records):
    d = {}
    for r in records:
        d.setdefault(r["condition"], {})[r["seed"]] = r["collusion_index_tail"]
    return d

main = by_cond(res["main"])
seeds_sorted = sorted(main["baseline"].keys())
base_full = np.array([main["baseline"][s] for s in seeds_sorted])

power_out = {}
rng = np.random.default_rng(0)
for cond in ["transparency", "audit_explore", "diversity", "audit_enforce"]:
    cond_full = np.array([main[cond][s] for s in seeds_sorted])
    diffs_full = cond_full - base_full
    row = {}
    for n in [5, 10, 15, 20, 25]:
        ps = []
        for trial in range(200):
            idx = rng.choice(len(diffs_full), size=n, replace=False) if n <= len(diffs_full) else rng.choice(len(diffs_full), size=n, replace=True)
            sub = diffs_full[idx]
            if sub.std(ddof=1) == 0:
                continue
            t, p = sstats.ttest_1samp(sub, 0.0)
            ps.append(p)
        row[n] = float(np.mean(np.array(ps) < 0.05)) if ps else None
    power_out[cond] = row
    print(f"{cond:15s} fraction of resamples significant at p<.05, by n: " + ", ".join(f"n={n}:{v:.2f}" for n, v in row.items()))

out["power_by_n"] = power_out

with open("extra_analysis_output.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved extra_analysis_output.json. Total time:", time.time() - t0)
