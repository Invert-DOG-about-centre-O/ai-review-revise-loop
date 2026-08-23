"""
Follow-up experiments addressing round-3 reviewer questions (all 4 reviewers
independently raised these):

1. How much of the 0.23 residual calibration gap is an artifact of the
   raw-agreement trust EMA vs. a real behavioral effect? We add a
   correctness-updating trust proxy (trust tracks 1[o==y] instead of
   1[o==b_u]) and report the gap under it at the REG_FAIR lambda=0.8 fixed
   point.
2. What separates the seeds that escape to the honest fixed point from those
   that stay sycophantic at gap=0.10, lambda=0.8 (the bimodal result)? We
   rerun that cell with per-seed early-trajectory alpha logged and check
   whether early alpha (round 500) predicts the final basin.
3. Does the "transparency = rescaled learning rate" identity survive under a
   qualitatively different optimizer? We add a tabular Q-learning variant
   over a discretized alpha grid and check whether TRANSPARENCY still
   reconverges to full sycophancy at long horizon under it.
"""
import numpy as np
import pandas as pd
import time

RNG_SEED0 = 12345
EMA_BETA = 0.98
BASELINE_BETA = 0.95
LR = 0.05


def run_condition(condition, seed, p_user, p_ai, lam=0.0, transparency=0.0,
                   n_rounds=20000, lr=LR, baseline_beta=BASELINE_BETA,
                   trust_mode="agree", log_every=None):
    rng = np.random.default_rng(seed)
    alpha = 0.0
    reward_baseline = 0.0
    trust_ema = 0.5

    accs = np.empty(n_rounds)
    trusts = np.empty(n_rounds)
    alpha_log = []

    for t in range(n_rounds):
        y = rng.integers(0, 2)
        b_u = y if rng.random() < p_user else 1 - y
        e = y if rng.random() < p_ai else 1 - y

        use_user = rng.random() < alpha
        o = b_u if use_user else e

        agree = 1.0 if o == b_u else 0.0
        correct = 1.0 if o == y else 0.0

        r_approve = agree
        if condition == "TRANSPARENCY":
            r_approve = agree * (1.0 - transparency)

        if condition in ("APPROVAL_ONLY", "TRANSPARENCY"):
            train_reward = r_approve
        elif condition == "REG_FAIR":
            train_reward = (1.0 - lam) * r_approve + lam * correct
        else:
            train_reward = None

        trust_signal = agree if trust_mode == "agree" else correct
        trust_ema = EMA_BETA * trust_ema + (1 - EMA_BETA) * trust_signal

        advantage = train_reward - reward_baseline
        reward_baseline = baseline_beta * reward_baseline + (1 - baseline_beta) * train_reward
        alpha = alpha + lr * advantage * (1 if use_user else -1) * 0.1
        alpha = min(1.0, max(0.0, alpha))

        accs[t] = correct
        trusts[t] = trust_ema
        if log_every and t % log_every == 0:
            alpha_log.append(alpha)

    return alpha, accs, trusts, alpha_log


def summarize(alpha, accs, trusts):
    tail = max(1, int(0.2 * len(accs)))
    accuracy = float(np.mean(accs[-tail:]))
    trust = float(np.mean(trusts[-tail:]))
    return dict(final_alpha=float(alpha), accuracy=accuracy, trust=trust,
                calibration_gap=abs(trust - accuracy))


# ---------------------------------------------------------------------------
# Experiment 1: correctness-updating trust proxy, REG_FAIR lambda=0.8, 20k
# ---------------------------------------------------------------------------
def exp1_trust_proxy(n_seeds=30, n_rounds=20000):
    rows_agree, rows_corr = [], []
    for s in range(n_seeds):
        a, accs, trusts, _ = run_condition("REG_FAIR", RNG_SEED0 + s, 0.60, 0.80,
                                            lam=0.8, n_rounds=n_rounds, trust_mode="agree")
        rows_agree.append(summarize(a, accs, trusts))
        a2, accs2, trusts2, _ = run_condition("REG_FAIR", RNG_SEED0 + s, 0.60, 0.80,
                                               lam=0.8, n_rounds=n_rounds, trust_mode="correct")
        rows_corr.append(summarize(a2, accs2, trusts2))
    df_agree = pd.DataFrame(rows_agree)
    df_corr = pd.DataFrame(rows_corr)
    return df_agree, df_corr


# ---------------------------------------------------------------------------
# Experiment 2: bimodality at gap=0.10, lambda=0.8 -- early alpha vs. outcome
# ---------------------------------------------------------------------------
def exp2_bimodality(n_seeds=30, n_rounds=20000):
    rows = []
    for s in range(n_seeds):
        a, accs, trusts, alog = run_condition("REG_FAIR", RNG_SEED0 + s, 0.65, 0.75,
                                               lam=0.8, n_rounds=n_rounds,
                                               log_every=100)
        rows.append(dict(seed=s, final_alpha=a,
                          alpha_500=alog[5], alpha_1000=alog[10],
                          alpha_2000=alog[20], alpha_5000=alog[50]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 3: tabular Q-learning optimizer -- does transparency=LR-rescaling
# survive under a qualitatively different update rule?
# ---------------------------------------------------------------------------
N_ARMS = 11  # alpha in {0.0, 0.1, ..., 1.0}
ALPHA_GRID = np.linspace(0.0, 1.0, N_ARMS)


def run_qlearning(condition, seed, p_user, p_ai, transparency=0.0,
                   n_rounds=20000, q_lr=0.05, epsilon=0.1, gamma=0.0):
    rng = np.random.default_rng(seed)
    Q = np.zeros(N_ARMS)
    accs = np.empty(n_rounds)
    alpha_choice = np.empty(n_rounds)

    for t in range(n_rounds):
        y = rng.integers(0, 2)
        b_u = y if rng.random() < p_user else 1 - y
        e = y if rng.random() < p_ai else 1 - y

        if rng.random() < epsilon:
            arm = rng.integers(0, N_ARMS)
        else:
            arm = int(np.argmax(Q))
        alpha_t = ALPHA_GRID[arm]

        use_user = rng.random() < alpha_t
        o = b_u if use_user else e
        agree = 1.0 if o == b_u else 0.0
        correct = 1.0 if o == y else 0.0

        r_approve = agree
        if condition == "TRANSPARENCY":
            r_approve = agree * (1.0 - transparency)
        reward = r_approve

        Q[arm] += q_lr * (reward - Q[arm])

        accs[t] = correct
        alpha_choice[t] = alpha_t

    tail = max(1, int(0.2 * n_rounds))
    return dict(final_alpha_arm=float(alpha_choice[-1]),
                mean_alpha_tail=float(np.mean(alpha_choice[-tail:])),
                accuracy=float(np.mean(accs[-tail:])))


def exp3_qlearning(n_seeds=10, n_rounds=20000):
    rows = []
    for cond, tr in [("APPROVAL_ONLY", 0.0), ("TRANSPARENCY", 0.8)]:
        for s in range(n_seeds):
            r = run_qlearning(cond, RNG_SEED0 + s, 0.60, 0.80, transparency=tr, n_rounds=n_rounds)
            r["condition"] = cond
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()

    print("=== Experiment 1: trust proxy (agree-EMA vs correctness-EMA), REG_FAIR lambda=0.8, 20k rounds, 30 seeds ===")
    df_agree, df_corr = exp1_trust_proxy()
    print("agree-EMA trust:    ", df_agree.mean().to_dict())
    print("correctness-EMA trust:", df_corr.mean().to_dict())
    df_agree.to_csv("results_followup_trust_agree.csv", index=False)
    df_corr.to_csv("results_followup_trust_correct.csv", index=False)
    print(f"[{time.time()-t0:.1f}s elapsed]\n")

    print("=== Experiment 2: bimodality at gap=0.10, lambda=0.8 -- early alpha vs final outcome, 30 seeds ===")
    df_bimodal = exp2_bimodality()
    df_bimodal.to_csv("results_followup_bimodality.csv", index=False)
    honest = df_bimodal[df_bimodal.final_alpha < 0.5]
    sycophant = df_bimodal[df_bimodal.final_alpha >= 0.5]
    print(f"n_honest_basin={len(honest)}, n_sycophant_basin={len(sycophant)}")
    for col in ["alpha_500", "alpha_1000", "alpha_2000", "alpha_5000"]:
        print(f"  {col}: honest_mean={honest[col].mean():.3f} sycophant_mean={sycophant[col].mean():.3f}")
    print(f"[{time.time()-t0:.1f}s elapsed]\n")

    print("=== Experiment 3: tabular Q-learning optimizer, APPROVAL_ONLY vs TRANSPARENCY(0.8), 20k rounds, 10 seeds ===")
    df_q = exp3_qlearning()
    df_q.to_csv("results_followup_qlearning.csv", index=False)
    print(df_q.groupby("condition").mean(numeric_only=True))
    print(f"[{time.time()-t0:.1f}s elapsed]\n")

    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
