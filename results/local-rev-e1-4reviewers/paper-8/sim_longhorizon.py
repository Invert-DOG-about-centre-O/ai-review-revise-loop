"""
Long-horizon re-run (20,000 / 60,000 rounds) generating results_longhorizon.csv.

This is the driver script four independent reviewers flagged as missing: the
underlying run_condition() function supported n_rounds>4000 all along, but no
included script actually called it at long horizon. This script fixes that,
and also tracks trust_ema (matching sim.py's calibration_gap definition,
EMA_BETA=0.98) so the reported calibration-gap numbers are computed the same
way as everywhere else in the paper -- sim_extra.py's run_condition tracked
trust_ema locally but never returned it, so its numbers were never usable for
calibration_gap.

Also runs a lambda-sensitivity sweep for REG_FAIR at three different
p_ai - p_user gaps, addressing the reviewer question about whether the
lambda=0.8 fixed point is specific to gap=0.20.
"""
import numpy as np
import pandas as pd
import time
from scipy import stats

RNG_SEED0 = 12345
EMA_BETA = 0.98
BASELINE_BETA = 0.95
LR = 0.05


def run_condition(condition, seed, p_user, p_ai, lam=0.0, q=1.0, transparency=0.0,
                   n_rounds=20000, lr=LR, baseline_beta=BASELINE_BETA):
    rng = np.random.default_rng(seed)
    alpha = 0.0
    reward_baseline = 0.0
    trust_ema = 0.5

    accs = np.empty(n_rounds)
    trusts = np.empty(n_rounds)
    alphas = np.empty(n_rounds)

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
        elif condition == "REG":
            train_reward = (1 - lam) * r_approve + lam * correct if rng.random() < q else r_approve
        elif condition == "REG_FAIR":
            train_reward = (1.0 - lam) * r_approve + lam * correct
        else:
            train_reward = None

        trust_ema = EMA_BETA * trust_ema + (1 - EMA_BETA) * agree

        advantage = train_reward - reward_baseline
        reward_baseline = baseline_beta * reward_baseline + (1 - baseline_beta) * train_reward
        alpha = alpha + lr * advantage * (1 if use_user else -1) * 0.1
        alpha = min(1.0, max(0.0, alpha))

        accs[t] = correct
        trusts[t] = trust_ema
        alphas[t] = alpha

    return alphas, accs, trusts


def summarize(alphas, accs, trusts):
    tail = max(1, int(0.2 * len(accs)))
    accuracy = float(np.mean(accs[-tail:]))
    trust = float(np.mean(trusts[-tail:]))
    return dict(final_alpha=float(alphas[-1]), accuracy=accuracy, trust=trust,
                calibration_gap=abs(trust - accuracy))


def run_many(condition, n_seeds, p_user=0.60, p_ai=0.80, **kwargs):
    rows = []
    for s in range(n_seeds):
        alphas, accs, trusts = run_condition(condition, RNG_SEED0 + s, p_user, p_ai, **kwargs)
        rows.append(summarize(alphas, accs, trusts))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    rows = []

    def add(name, n_rounds, n_seeds, **kw):
        df = run_many(n_rounds=n_rounds, n_seeds=n_seeds, **kw)
        r = {"condition": name, "n_rounds": n_rounds, "n_seeds": n_seeds}
        for col in df.columns:
            r[f"{col}_mean"] = float(df[col].mean())
            r[f"{col}_std"] = float(df[col].std())
        rows.append(r)
        return df

    # --- 20,000 rounds, 30 seeds ---
    add("Approval_only", 20000, 30, condition="APPROVAL_ONLY")
    for tr in [0.2, 0.5, 0.8]:
        add(f"Transparency_{tr}", 20000, 30, condition="TRANSPARENCY", transparency=tr)
    add("REG_lambda0.5_q1.0", 20000, 30, condition="REG", lam=0.5, q=1.0)
    add("REG_FAIR_lambda0.5", 20000, 30, condition="REG_FAIR", lam=0.5)
    reg_fair_08_20k = add("REG_FAIR_lambda0.8", 20000, 30, condition="REG_FAIR", lam=0.8)

    # --- 60,000 rounds, 10 seeds (stability check) ---
    reg_fair_08_60k = add("REG_FAIR_lambda0.8", 60000, 10, condition="REG_FAIR", lam=0.8)

    # --- lambda-sensitivity sweep for REG_FAIR across the p_ai - p_user gap ---
    for gap_name, p_user, p_ai in [("gap0.10", 0.65, 0.75), ("gap0.20", 0.60, 0.80), ("gap0.30", 0.55, 0.85)]:
        for lam in [0.05, 0.1, 0.2, 0.4, 0.8]:
            add(f"REG_FAIR_{gap_name}_lambda{lam}", 20000, 10, condition="REG_FAIR",
                lam=lam, p_user=p_user, p_ai=p_ai)

    df_out = pd.DataFrame(rows)
    df_out.to_csv("results_longhorizon.csv", index=False)
    print(df_out.to_string(index=False))

    print(f"\nREG_FAIR lambda=0.8 calibration_gap: 20k={reg_fair_08_20k['calibration_gap'].mean():.3f} "
          f"(range {reg_fair_08_20k['calibration_gap'].min():.3f}-{reg_fair_08_20k['calibration_gap'].max():.3f}), "
          f"60k={reg_fair_08_60k['calibration_gap'].mean():.3f} "
          f"(range {reg_fair_08_60k['calibration_gap'].min():.3f}-{reg_fair_08_60k['calibration_gap'].max():.3f})")

    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
