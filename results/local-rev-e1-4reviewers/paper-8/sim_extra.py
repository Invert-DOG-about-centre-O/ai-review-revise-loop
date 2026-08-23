"""
Follow-up experiments addressing reviewer weaknesses on the sycophancy-drift sim.

1. Fair REG comparison: a "REG_FAIR" condition that discounts correctness with a
   single multiplicative factor (like TRANSPARENCY discounts agreement), with
   correctness observed on every round, so both mitigations attenuate their
   respective signal through exactly one knob.
2. Locate the REG q transition point between 0.5 and 1.0 (q in 0.7, 0.9).
3. Robustness of the transparency >> REG ranking to the hill-climb learning rate
   (LR in 0.02, 0.05 (baseline), 0.1) and EMA baseline beta (0.90, 0.95, 0.99).
4. Paired significance tests (Welch t-test across the 30 seeds) for:
   - lambda sweep at q=0.3 (lambda=0.1 vs lambda=0.8) on final_alpha
   - REG_FAIR vs TRANSPARENCY at matched strength 0.8
   - LR robustness: approval-only final_alpha and transparency=0.8 final_alpha
     across LR values
"""
import numpy as np
import pandas as pd
import json
import time
from scipy import stats

RNG_SEED0 = 12345
N_ROUNDS = 4000
N_SEEDS = 30
P_USER = 0.60
P_AI = 0.80
EMA_BETA = 0.98


def run_condition(condition, seed, lam=0.0, q=1.0, transparency=0.0,
                   fixed_alpha=None, n_rounds=N_ROUNDS, lr=0.05, baseline_beta=0.95):
    rng = np.random.default_rng(seed)
    alpha = 0.0 if fixed_alpha is None else fixed_alpha
    reward_baseline = 0.0
    trust_ema = 0.5

    alphas = np.empty(n_rounds)
    accs = np.empty(n_rounds)

    for t in range(n_rounds):
        y = rng.integers(0, 2)
        b_u = y if rng.random() < P_USER else 1 - y
        e = y if rng.random() < P_AI else 1 - y

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
            if rng.random() < q:
                train_reward = (1 - lam) * r_approve + lam * correct
            else:
                train_reward = r_approve
        elif condition == "REG_FAIR":
            # single-knob attenuation: correctness always observed, agreement
            # reward discounted by the SAME strength factor used for lambda,
            # and the correctness bonus added undiminished (mirrors how
            # TRANSPARENCY discounts agreement by exactly one factor).
            train_reward = (1.0 - lam) * r_approve + lam * correct
        else:
            train_reward = None

        trust_ema = EMA_BETA * trust_ema + (1 - EMA_BETA) * agree

        if fixed_alpha is None:
            advantage = train_reward - reward_baseline
            reward_baseline = baseline_beta * reward_baseline + (1 - baseline_beta) * train_reward
            alpha = alpha + lr * advantage * (1 if use_user else -1) * 0.1
            alpha = min(1.0, max(0.0, alpha))

        alphas[t] = alpha
        accs[t] = correct

    return alphas, accs


def summarize(alphas, accs):
    tail = max(1, int(0.2 * len(accs)))
    return dict(final_alpha=float(alphas[-1]), accuracy=float(np.mean(accs[-tail:])))


def run_many(condition, n_seeds=N_SEEDS, **kwargs):
    rows = []
    for s in range(n_seeds):
        alphas, accs = run_condition(condition, RNG_SEED0 + s, **kwargs)
        rows.append(summarize(alphas, accs))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    out = {}

    # --- 1. Fair REG comparison (single-knob discount, q=1.0 always) ---
    for lam in [0.2, 0.5, 0.8]:
        out[f"REG_FAIR_{lam}"] = run_many("REG_FAIR", lam=lam)
    for tr in [0.2, 0.5, 0.8]:
        out[f"TRANSPARENCY_{tr}"] = run_many("TRANSPARENCY", transparency=tr)

    # --- 2. Locate REG q transition (lambda=0.5) ---
    for q in [0.5, 0.7, 0.9, 1.0]:
        out[f"REG_q{q}"] = run_many("REG", lam=0.5, q=q)

    # --- 3. Robustness to LR and baseline_beta ---
    for lr in [0.02, 0.05, 0.1]:
        out[f"APPROVAL_ONLY_lr{lr}"] = run_many("APPROVAL_ONLY", lr=lr)
        out[f"TRANSPARENCY_0.8_lr{lr}"] = run_many("TRANSPARENCY", transparency=0.8, lr=lr)
    for bb in [0.90, 0.95, 0.99]:
        out[f"APPROVAL_ONLY_bb{bb}"] = run_many("APPROVAL_ONLY", baseline_beta=bb)
        out[f"TRANSPARENCY_0.8_bb{bb}"] = run_many("TRANSPARENCY", transparency=0.8, baseline_beta=bb)

    # --- 4. lambda sweep at q=0.3 (original REG condition) for sig test ---
    for lam in [0.1, 0.8]:
        out[f"REG_lambda{lam}_q0.3"] = run_many("REG", lam=lam, q=0.3)

    summary_rows = []
    for name, df in out.items():
        row = {"condition": name}
        for col in df.columns:
            row[f"{col}_mean"] = float(df[col].mean())
            row[f"{col}_std"] = float(df[col].std())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_extra.csv", index=False)
    print(summary_df.to_string(index=False))

    # significance tests
    print("\n--- Significance tests (Welch t-test on final_alpha across 30 seeds) ---")
    t1, p1 = stats.ttest_ind(out["REG_lambda0.1_q0.3"]["final_alpha"],
                              out["REG_lambda0.8_q0.3"]["final_alpha"], equal_var=False)
    print(f"lambda=0.1 vs lambda=0.8 (q=0.3): t={t1:.3f}, p={p1:.2e}")

    t2, p2 = stats.ttest_ind(out["REG_FAIR_0.8"]["final_alpha"],
                              out["TRANSPARENCY_0.8"]["final_alpha"], equal_var=False)
    print(f"REG_FAIR(0.8) vs TRANSPARENCY(0.8) final_alpha: t={t2:.3f}, p={p2:.2e}; "
          f"means {out['REG_FAIR_0.8']['final_alpha'].mean():.3f} vs {out['TRANSPARENCY_0.8']['final_alpha'].mean():.3f}")

    t3, p3 = stats.ttest_ind(out["REG_FAIR_0.8"]["accuracy"],
                              out["TRANSPARENCY_0.8"]["accuracy"], equal_var=False)
    print(f"REG_FAIR(0.8) vs TRANSPARENCY(0.8) accuracy: t={t3:.3f}, p={p3:.2e}; "
          f"means {out['REG_FAIR_0.8']['accuracy'].mean():.3f} vs {out['TRANSPARENCY_0.8']['accuracy'].mean():.3f}")

    for lr in [0.02, 0.1]:
        t, p = stats.ttest_ind(out["TRANSPARENCY_0.8_lr0.05"]["final_alpha"],
                                out[f"TRANSPARENCY_0.8_lr{lr}"]["final_alpha"], equal_var=False)
        print(f"TRANSPARENCY 0.8 final_alpha, LR=0.05 vs LR={lr}: t={t:.3f}, p={p:.2e}; "
              f"means {out['TRANSPARENCY_0.8_lr0.05']['final_alpha'].mean():.3f} vs {out[f'TRANSPARENCY_0.8_lr{lr}']['final_alpha'].mean():.3f}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
