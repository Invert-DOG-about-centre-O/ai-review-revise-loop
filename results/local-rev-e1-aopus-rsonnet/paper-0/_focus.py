import numpy as np
from sim import run
TRAP = 2.0
def fw(logs, key): return float(np.mean([lg[key] for lg in logs[-10:]]))
def ps(alloc, seeds, key, **kw):
    return np.array([fw(run(alloc, seed=s, feat_noise=TRAP, **kw), key) for s in seeds])
SEEDS = list(range(16))
print("THRESH robustness (16 seeds)", flush=True)
for th in [-0.8, -0.3, 0.0, 0.3]:
    g05 = ps("greedy", SEEDS, "rate_gap", seed_B_frac=0.05, seed_thresh=th).mean()
    g50 = ps("greedy", SEEDS, "rate_gap", seed_B_frac=0.5, seed_thresh=th).mean()
    print(f"th={th:+.1f} gap@0.05={g05:.3f} gap@0.5={g50:.3f} slope={g50-g05:+.3f}", flush=True)
print("BETA reversal (16 seeds)", flush=True)
for beta in [2.0, 4.0, 6.0, 8.0]:
    qa = ps("ucb", SEEDS, "qual_appr", beta=beta).mean()
    ut = ps("ucb", SEEDS, "util", beta=beta).mean()
    rg = ps("ucb", SEEDS, "rate_gap", beta=beta).mean()
    print(f"beta={beta} rate_gap={rg:.3f} qual_appr={qa:.3f} util={ut:.3f}", flush=True)
