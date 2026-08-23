"""Revision experiments: significance tests over more seeds, extended UCB beta
sweep (does the free lunch break?), and robustness of the seed_B_frac finding to
the seed-region restriction threshold. Reuses run() from sim.py."""
import numpy as np
from sim import run

TRAP = 2.0

def final_window(logs, key):
    return float(np.mean([lg[key] for lg in logs[-10:]]))

def per_seed(allocator, seeds, key, **kw):
    return np.array([final_window(run(allocator, seed=s, feat_noise=TRAP, **kw), key)
                     for s in seeds])

def paired_boot(a, b, nboot=20000, rng=None):
    """Paired bootstrap p-value (two-sided) and mean diff a-b over paired seeds."""
    rng = rng or np.random.default_rng(0)
    d = a - b
    n = len(d)
    obs = d.mean()
    idx = rng.integers(0, n, size=(nboot, n))
    boot = d[idx].mean(axis=1)
    # two-sided p: prob bootstrap mean crosses 0 relative to observed
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return obs, float(min(p, 1.0))

SEEDS = list(range(32))
print(f"=== Significance: allocators, trap regime, N={len(SEEDS)} seeds ===")
alloc = {}
for a in ["greedy", "eps_explore", "ucb", "dp_parity"]:
    alloc[a] = {k: per_seed(a, SEEDS, k) for k in
                ["rate_gap", "tpr_gap", "qual_appr", "util", "calB"]}
    m = alloc[a]
    print(f"{a:12s} rate_gap={m['rate_gap'].mean():.3f}+/-{m['rate_gap'].std():.3f}  "
          f"tpr_gap={m['tpr_gap'].mean():.3f}+/-{m['tpr_gap'].std():.3f}  "
          f"qual_appr={m['qual_appr'].mean():.3f}  util={m['util'].mean():.3f}+/-{m['util'].std():.3f}  "
          f"calB={m['calB'].mean():.3f}")

print("\n=== Paired bootstrap tests (diff = X - Y, 20000 resamples) ===")
def report(x, y, key):
    d, p = paired_boot(alloc[x][key], alloc[y][key])
    print(f"{key:9s}  {x} - {y}: diff={d:+.4f}  p={p:.4f}")
report("greedy", "eps_explore", "rate_gap")
report("greedy", "ucb", "rate_gap")
report("greedy", "eps_explore", "tpr_gap")
report("ucb", "eps_explore", "rate_gap")     # UCB vs eps: significant?
report("ucb", "eps_explore", "util")          # UCB's utility edge over eps
report("ucb", "eps_explore", "qual_appr")
report("ucb", "greedy", "qual_appr")
report("dp_parity", "greedy", "calB")         # parity does NOT fix calibration
report("dp_parity", "greedy", "rate_gap")

print(f"\n=== Extended UCB beta sweep (does the free lunch break?), N={len(SEEDS)} ===")
print(f"{'beta':>5s} {'rate_gap':>10s} {'tpr_gap':>10s} {'qual_appr':>10s} {'util':>14s}")
for beta in [0.0, 1.2, 2.0, 3.0, 4.0, 6.0, 8.0]:
    rg = per_seed("ucb", SEEDS, "rate_gap", beta=beta)
    tg = per_seed("ucb", SEEDS, "tpr_gap", beta=beta)
    qa = per_seed("ucb", SEEDS, "qual_appr", beta=beta)
    ut = per_seed("ucb", SEEDS, "util", beta=beta)
    print(f"{beta:>5} {rg.mean():>10.3f} {tg.mean():>10.3f} {qa.mean():>10.3f} "
          f"{ut.mean():>8.3f}+/-{ut.std():.3f}")

print(f"\n=== Robustness of seed_B_frac finding to restriction threshold, N={len(SEEDS)} ===")
print("(rate_gap at seed_B_frac 0.05 vs 0.5; positive slope = 'more biased data hurts')")
print(f"{'thresh':>7s} {'gap@0.05':>9s} {'gap@0.5':>9s} {'slope':>8s}")
for th in [-0.8, -0.3, 0.0, 0.3]:
    g05 = per_seed("greedy", SEEDS, "rate_gap", seed_B_frac=0.05, seed_thresh=th).mean()
    g50 = per_seed("greedy", SEEDS, "rate_gap", seed_B_frac=0.5, seed_thresh=th).mean()
    print(f"{th:>7} {g05:>9.3f} {g50:>9.3f} {g50 - g05:>+8.3f}")
