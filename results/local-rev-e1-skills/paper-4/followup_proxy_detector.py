"""
Follow-up requested by round-3 review (weakness 2 / question 2): the audit trigger in
the main paper assumes the regulator observes the true (demand-model) Nash price P_NASH
in real time. Here we test a demand-model-free proxy: the regulator instead estimates
p_hat_nash as the 5th percentile of realized prices during the first 5000 periods (the
early high-exploration phase, before agents have learned to coordinate), and uses that
estimate (instead of the true P_NASH) both to build its rolling collusion-index trigger
and to define the enforced "competitive" price during a cooldown window (it forces both
firms to p_hat_nash rather than the true P_NASH). We compare this proxy-audit condition
against the true-Nash audit_enforce condition and against baseline, n=25, main market.
"""
import time
import math
import random
import numpy as np
from scipy import stats as sstats
import collusion_sim as cs

PERIODS = 150000
SEEDS25 = list(range(25))
EARLY_WINDOW = 5000
AUDIT_THETA = 0.5
AUDIT_WINDOW = 1000
AUDIT_BOOST_LEN = 1000


def run_sim_proxy_audit(seed):
    rng = random.Random(seed)
    M = cs.M
    Q1 = np.zeros((M, M, M))
    Q2 = np.zeros((M, M, M))
    a1 = rng.randrange(M)
    a2 = rng.randrange(M)
    s1_opp, s2_opp = a2, a1
    price_hist = np.zeros(PERIODS, dtype=np.int16)
    price_hist2 = np.zeros(PERIODS, dtype=np.int16)

    p_hat_nash = None
    hat_nash_idx = None
    audit_active_until = -1
    n_audits = 0

    for t in range(PERIODS):
        eps = math.exp(-cs.BETA * t)
        audited_now = t <= audit_active_until

        if audited_now and p_hat_nash is not None:
            act1 = hat_nash_idx
        elif rng.random() < eps:
            act1 = rng.randrange(M)
        else:
            act1 = int(np.argmax(Q1[a1, s1_opp]))

        if audited_now and p_hat_nash is not None:
            act2 = hat_nash_idx
        elif rng.random() < eps:
            act2 = rng.randrange(M)
        else:
            act2 = int(np.argmax(Q2[a2, s2_opp]))

        p1_idx, p2_idx = act1, act2
        r1 = cs.PROFIT_MAT[p1_idx, p2_idx]
        r2 = cs.PROFIT_MAT[p2_idx, p1_idx]
        best_next1 = np.max(Q1[p1_idx, p2_idx])
        Q1[a1, s1_opp, act1] += cs.ALPHA * (r1 + cs.DELTA * best_next1 - Q1[a1, s1_opp, act1])
        best_next2 = np.max(Q2[p2_idx, p1_idx])
        Q2[a2, s2_opp, act2] += cs.ALPHA * (r2 + cs.DELTA * best_next2 - Q2[a2, s2_opp, act2])
        a1, a2 = p1_idx, p2_idx
        s1_opp, s2_opp = p2_idx, p1_idx
        price_hist[t] = p1_idx
        price_hist2[t] = p2_idx

        # regulator estimates p_hat_nash once, right after the early exploration window
        if t == EARLY_WINDOW - 1:
            early_prices = cs.PRICE_GRID[np.concatenate([price_hist[:EARLY_WINDOW], price_hist2[:EARLY_WINDOW]])]
            p_hat_nash = float(np.percentile(early_prices, 5))
            hat_nash_idx = int(np.argmin(np.abs(cs.PRICE_GRID - p_hat_nash)))

        if p_hat_nash is not None and t >= EARLY_WINDOW + AUDIT_WINDOW and t > audit_active_until:
            window_prices = cs.PRICE_GRID[np.concatenate([
                price_hist[t - AUDIT_WINDOW + 1: t + 1], price_hist2[t - AUDIT_WINDOW + 1: t + 1]])]
            avg_p = window_prices.mean()
            idx_hat = (avg_p - p_hat_nash) / (cs.P_MONOPOLY - p_hat_nash)
            if idx_hat > AUDIT_THETA:
                audit_active_until = t + AUDIT_BOOST_LEN
                n_audits += 1

    tail = PERIODS // 10
    avg_price_tail = cs.PRICE_GRID[np.concatenate([price_hist[-tail:], price_hist2[-tail:]])].mean()
    idx_tail = cs.collusion_index(avg_price_tail)  # evaluated with TRUE P_NASH, for fair comparison
    return idx_tail, p_hat_nash, n_audits


t0 = time.time()
proxy_vals, hats, audits = [], [], []
for s in SEEDS25:
    idx, hat, na = run_sim_proxy_audit(s)
    proxy_vals.append(idx)
    hats.append(hat)
    audits.append(na)
proxy_vals = np.array(proxy_vals)
print("proxy audit done", time.time() - t0, "mean idx=", proxy_vals.mean(),
      "mean p_hat_nash=", np.mean(hats), "true P_NASH=", cs.P_NASH, "mean n_audits=", np.mean(audits))

base = np.array([cs.run_sim(seed=s, periods=PERIODS, condition="baseline")["collusion_index_tail"] for s in SEEDS25])
true_audit = np.array([cs.run_sim(seed=s, periods=PERIODS, condition="audit_enforce")["collusion_index_tail"] for s in SEEDS25])
print("baseline mean=", base.mean(), "true-Nash audit_enforce mean=", true_audit.mean())

t1, p1 = sstats.ttest_rel(proxy_vals, base)
t2, p2 = sstats.ttest_rel(proxy_vals, true_audit)
print(f"proxy-audit vs baseline: diff={proxy_vals.mean()-base.mean():+.4f} p={p1:.2e}")
print(f"proxy-audit vs true-Nash audit: diff={proxy_vals.mean()-true_audit.mean():+.4f} p={p2:.2e}")
print("total elapsed", time.time() - t0)
