"""
Follow-up requested by round-2 review (question 2): how does the myopic firm's
observed index (0.155 under diversity) compare to what two purely-myopic
best-responders converge to when paired against each other (no learning at
all)? This contextualizes whether 0.155 reflects residual softening from
facing a learner, or is just what myopic best-response converges to anyway.
"""
import time
import random
import numpy as np
import collusion_sim as cs

PERIODS = 150000
SEEDS = list(range(10))


def run_both_myopic(seed):
    rng = random.Random(seed)
    M = cs.M
    a1 = rng.randrange(M)
    a2 = rng.randrange(M)
    price_hist1 = np.zeros(PERIODS, dtype=np.int16)
    price_hist2 = np.zeros(PERIODS, dtype=np.int16)
    for t in range(PERIODS):
        act1 = int(np.argmax(cs.PROFIT_MAT[:, a2]))
        act2 = int(np.argmax(cs.PROFIT_MAT[:, a1]))
        a1, a2 = act1, act2
        price_hist1[t] = a1
        price_hist2[t] = a2
    tail = PERIODS // 10
    avg_price = cs.PRICE_GRID[np.concatenate([price_hist1[-tail:], price_hist2[-tail:]])].mean()
    return cs.collusion_index(avg_price)


t0 = time.time()
vals = [run_both_myopic(s) for s in SEEDS]
print("both-myopic index per seed:", [f"{v:.4f}" for v in vals])
print("both-myopic mean:", np.mean(vals), "elapsed", time.time() - t0)
