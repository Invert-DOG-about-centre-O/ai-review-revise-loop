"""
Q-learning Bertrand pricing simulation with socio-technical governance interventions.

Implements a Calvano-style (2020) repeated logit-demand Bertrand duopoly where both
firms use tabular Q-learning to set prices, and studies whether cheap, deployable
"socio-technical" governance interventions reduce the algorithmic tacit-collusion
that emerges under a plain baseline.

All randomness is seeded; every run in this file is fully deterministic given its seed.
"""
import random
import math
import json
import time
import numpy as np
from scipy import stats as sstats

# ---------------------------------------------------------------------------
# Demand / profit model (Calvano et al. 2020 logit demand)
# ---------------------------------------------------------------------------
A = 2.0        # quality index (both firms symmetric)
A0 = 0.0       # outside-option quality
MU = 0.25      # horizontal differentiation / logit scale
COST = 1.0     # marginal cost


def demand(p1, p2):
    e1 = math.exp((A - p1) / MU)
    e2 = math.exp((A - p2) / MU)
    e0 = math.exp((A0) / MU)
    denom = e1 + e2 + e0
    return e1 / denom, e2 / denom


def profits(p1, p2):
    q1, q2 = demand(p1, p2)
    return (p1 - COST) * q1, (p2 - COST) * q2


def nash_price():
    # symmetric Bertrand-Nash: firm i best-responds to firm j's price, solved by
    # fixed point on a fine grid (standard for this model, see Calvano et al. 2020).
    grid = np.linspace(COST, COST + 3.0, 4000)
    p = 1.5
    for _ in range(200):
        # best response of firm 1 to p2=p on fine grid
        profs = [ (pp - COST) * demand(pp, p)[0] for pp in grid ]
        p_new = grid[int(np.argmax(profs))]
        if abs(p_new - p) < 1e-6:
            p = p_new
            break
        p = p_new
    return p


def monopoly_price():
    grid = np.linspace(COST, COST + 3.0, 4000)
    # symmetric joint-profit maximizing price (both firms set same price)
    profs = [ 2 * (pp - COST) * demand(pp, pp)[0] for pp in grid ]
    return grid[int(np.argmax(profs))]


P_NASH = nash_price()
P_MONOPOLY = monopoly_price()

# ---------------------------------------------------------------------------
# Price grid (Calvano-style: extend beyond [Nash, Monopoly] by xi on each side)
# ---------------------------------------------------------------------------
XI = 0.1
M = 15  # number of discrete price points
PRICE_GRID = np.linspace(
    P_NASH - XI * (P_MONOPOLY - P_NASH),
    P_MONOPOLY + XI * (P_MONOPOLY - P_NASH),
    M,
)

# precompute profit matrix profit_mat[i,j] = profit of firm playing action i when
# opponent plays action j (symmetric, so used for both firms)
PROFIT_MAT = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        PROFIT_MAT[i, j] = (PRICE_GRID[i] - COST) * demand(PRICE_GRID[i], PRICE_GRID[j])[0]

PROFIT_MIN = PROFIT_MAT.min()
PROFIT_MAX = PROFIT_MAT.max()

# ---------------------------------------------------------------------------
# Q-learning agent config (Calvano-style)
# ---------------------------------------------------------------------------
ALPHA = 0.15     # learning rate
DELTA = 0.95     # discount factor
BETA = 4e-5      # exploration decay rate: eps_t = exp(-BETA * t)


def collusion_index(avg_price):
    return (avg_price - P_NASH) / (P_MONOPOLY - P_NASH)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
NASH_IDX = int(np.argmin(np.abs(PRICE_GRID - P_NASH)))


def run_sim(seed, periods, condition="baseline", audit_theta=0.5, audit_window=1000,
            audit_boost_eps=0.5, audit_boost_len=1000, transparency_bins=3):
    """
    condition in {"baseline", "audit_explore", "audit_enforce", "diversity", "transparency"}
    Returns dict of summary stats.
    """
    rng = random.Random(seed)

    is_audit = condition in ("audit_explore", "audit_enforce")

    # state encoding depends on condition:
    #  - baseline / audit_* / diversity: state = (last p1 idx, last p2 idx) -> full transparency
    #  - transparency: state = (last own idx, coarse bin of opponent idx)
    if condition == "transparency":
        n_opp_states = transparency_bins
    else:
        n_opp_states = M

    Q1 = np.zeros((M, n_opp_states, M))
    if condition != "diversity":
        Q2 = np.zeros((M, n_opp_states, M))

    # init state: random starting prices
    a1 = rng.randrange(M)
    a2 = rng.randrange(M)

    def opp_state(idx):
        if condition == "transparency":
            return int(idx * transparency_bins // M)
        return idx

    s1_opp = opp_state(a2)
    s2_opp = opp_state(a1)

    price_hist = np.zeros(periods, dtype=np.int16)
    price_hist2 = np.zeros(periods, dtype=np.int16)

    audit_active_until = -1
    n_audits = 0

    for t in range(periods):
        eps = math.exp(-BETA * t)
        audited_now = is_audit and t <= audit_active_until
        if audited_now and condition == "audit_explore":
            eps = audit_boost_eps

        # --- firm 1 action (own last price a1, opponent-state s1_opp) ---
        if audited_now and condition == "audit_enforce":
            act1 = NASH_IDX
        elif rng.random() < eps:
            act1 = rng.randrange(M)
        else:
            act1 = int(np.argmax(Q1[a1, s1_opp]))

        # --- firm 2 action ---
        if condition == "diversity":
            # myopic best response to firm 1's last observed price (no learning, no memory)
            act2 = int(np.argmax(PROFIT_MAT[:, a1]))
        elif audited_now and condition == "audit_enforce":
            act2 = NASH_IDX
        else:
            eps2 = eps
            if rng.random() < eps2:
                act2 = rng.randrange(M)
            else:
                act2 = int(np.argmax(Q2[a2, s2_opp]))

        p1_idx, p2_idx = act1, act2
        r1 = PROFIT_MAT[p1_idx, p2_idx]
        r2 = PROFIT_MAT[p2_idx, p1_idx]

        new_s1_opp = opp_state(p2_idx)
        new_s2_opp = opp_state(p1_idx)

        # Q-learning updates
        best_next1 = np.max(Q1[p1_idx, new_s1_opp])
        Q1[a1, s1_opp, act1] += ALPHA * (r1 + DELTA * best_next1 - Q1[a1, s1_opp, act1])

        if condition != "diversity":
            best_next2 = np.max(Q2[p2_idx, new_s2_opp])
            Q2[a2, s2_opp, act2] += ALPHA * (r2 + DELTA * best_next2 - Q2[a2, s2_opp, act2])

        a1, a2 = p1_idx, p2_idx
        s1_opp, s2_opp = new_s1_opp, new_s2_opp

        price_hist[t] = p1_idx
        price_hist2[t] = p2_idx

        # --- audit governance layer ---
        if is_audit and t >= audit_window and t > audit_active_until:
            window_prices = PRICE_GRID[price_hist[t - audit_window + 1: t + 1]]
            avg_p = window_prices.mean()
            idx = collusion_index(avg_p)
            if idx > audit_theta:
                audit_active_until = t + audit_boost_len
                n_audits += 1

    tail = periods // 10  # last 10% of periods
    avg_price_tail = PRICE_GRID[np.concatenate([price_hist[-tail:], price_hist2[-tail:]])].mean()
    idx_tail = collusion_index(avg_price_tail)

    avg_price_last1000 = PRICE_GRID[np.concatenate([price_hist[-1000:], price_hist2[-1000:]])].mean()
    idx_last1000 = collusion_index(avg_price_last1000)

    return {
        "seed": seed,
        "condition": condition,
        "collusion_index_tail": float(idx_tail),
        "collusion_index_last1000": float(idx_last1000),
        "avg_price_tail": float(avg_price_tail),
        "n_audits": n_audits,
    }


if __name__ == "__main__":
    print("P_NASH =", P_NASH, "P_MONOPOLY =", P_MONOPOLY)
    t0 = time.time()
    r = run_sim(seed=0, periods=20000, condition="baseline")
    print("timing test (20000 periods):", time.time() - t0, "s ->", r)
