"""
Runs the full experiment battery for the socio-technical governance study of
algorithmic (Q-learning) tacit collusion, and writes results to results.json.
Deterministic given the seed list.
"""
import json
import time
import numpy as np
from scipy import stats as sstats
import collusion_sim as cs

PERIODS = 150000
N_SEEDS = 25
SEEDS = list(range(N_SEEDS))

MAIN_CONDITIONS = ["baseline", "diversity", "transparency", "audit_explore", "audit_enforce"]

results = {"meta": {}, "main": [], "threshold_ablation": [], "market_robustness": []}

t_start = time.time()

results["meta"] = {
    "P_NASH": cs.P_NASH,
    "P_MONOPOLY": cs.P_MONOPOLY,
    "PRICE_GRID": cs.PRICE_GRID.tolist(),
    "M": cs.M,
    "PERIODS": PERIODS,
    "N_SEEDS": N_SEEDS,
    "ALPHA": cs.ALPHA,
    "DELTA": cs.DELTA,
    "BETA": cs.BETA,
}

# ---------------------------------------------------------------------------
# 1. Main comparison: baseline vs 4 interventions, matched seeds
# ---------------------------------------------------------------------------
print("=== Main comparison ===")
for cond in MAIN_CONDITIONS:
    for seed in SEEDS:
        r = cs.run_sim(seed=seed, periods=PERIODS, condition=cond)
        results["main"].append(r)
    elapsed = time.time() - t_start
    print(f"{cond}: done ({elapsed:.1f}s elapsed)")

# ---------------------------------------------------------------------------
# 2. Threshold sensitivity ablation for audit_enforce (matched seed count)
# ---------------------------------------------------------------------------
print("=== Threshold ablation (audit_enforce) ===")
for theta in [0.3, 0.7]:
    for seed in SEEDS:
        r = cs.run_sim(seed=seed, periods=PERIODS, condition="audit_enforce", audit_theta=theta)
        r["audit_theta"] = theta
        results["threshold_ablation"].append(r)
    elapsed = time.time() - t_start
    print(f"theta={theta}: done ({elapsed:.1f}s elapsed)")

# ---------------------------------------------------------------------------
# 3. Market-structure robustness: less-differentiated market (higher MU -> more
#    price competition sensitivity), fewer seeds (secondary robustness axis)
# ---------------------------------------------------------------------------
print("=== Market robustness (MU=0.5) ===")
N_SEEDS_ROBUST = 15
SEEDS_ROBUST = list(range(N_SEEDS_ROBUST))

# monkey-patch demand params for a structurally different market and recompute grid
import importlib
cs.MU = 0.5
cs.P_NASH = cs.nash_price()
cs.P_MONOPOLY = cs.monopoly_price()
cs.PRICE_GRID = np.linspace(
    cs.P_NASH - cs.XI * (cs.P_MONOPOLY - cs.P_NASH),
    cs.P_MONOPOLY + cs.XI * (cs.P_MONOPOLY - cs.P_NASH),
    cs.M,
)
PROFIT_MAT2 = np.zeros((cs.M, cs.M))
for i in range(cs.M):
    for j in range(cs.M):
        PROFIT_MAT2[i, j] = (cs.PRICE_GRID[i] - cs.COST) * cs.demand(cs.PRICE_GRID[i], cs.PRICE_GRID[j])[0]
cs.PROFIT_MAT = PROFIT_MAT2
cs.NASH_IDX = int(np.argmin(np.abs(cs.PRICE_GRID - cs.P_NASH)))

results["meta"]["robust_market"] = {"MU": cs.MU, "P_NASH": cs.P_NASH, "P_MONOPOLY": cs.P_MONOPOLY}

for cond in MAIN_CONDITIONS:
    for seed in SEEDS_ROBUST:
        r = cs.run_sim(seed=seed, periods=PERIODS, condition=cond)
        results["market_robustness"].append(r)
    elapsed = time.time() - t_start
    print(f"[robust] {cond}: done ({elapsed:.1f}s elapsed)")

total_time = time.time() - t_start
results["meta"]["total_runtime_sec"] = total_time
print("Total runtime:", total_time, "s")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results.json")
