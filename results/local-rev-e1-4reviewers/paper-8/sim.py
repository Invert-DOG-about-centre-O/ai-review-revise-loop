"""
Agent-based simulation of approval-driven sycophancy in an adaptive AI advisor.

Setting
-------
At each round t, a simulated user holds a prior belief b_u in {0,1} about a binary
question whose ground truth is y (b_u == y with probability p_user, i.e. the user's
own prior is noisy but better than chance). The AI advisor has private "evidence"
e, an independent noisy estimate of y (e == y with probability p_ai > p_user, i.e.
the AI is a better-than-user but imperfect judge).

The advisor outputs o in {0,1}. It mixes between reporting its evidence e and
echoing the user's prior b_u, controlled by a scalar "sycophancy dial" alpha in
[0,1]:  o = b_u with probability alpha, else o = e.

Feedback: the user "approves" (reward r_approve = 1) when the advisor's output
agrees with the user's own prior belief (o == b_u) -- capturing the widely
reported tendency of human raters/users to reward agreement over correctness in
short-horizon feedback (thumbs-up/down, chat continuation, etc.), independent of
whether the advisor was actually right. A correctness signal r_correct = 1[o==y]
is also computed but is only partially/late observable to the training loop,
depending on condition (modeling delayed outcome feedback in real deployments).

alpha is updated online via a simple stochastic hill-climbing / REINFORCE-style
rule driven by a training reward that differs by condition:

  A. APPROVAL_ONLY   : reward = r_approve                       (pure engagement-optimized)
  B. REG(lambda)     : reward = (1-lambda)*r_approve + lambda*r_correct_delayed
                        where r_correct_delayed is only available with prob q
  C. TRANSPARENCY    : approval reward is discounted by a transparency factor
                        whenever o == b_u (user is told the advisor may be
                        adapting to them, which weakens the pure-agreement boost)
  D. STATIC_HONEST   : alpha fixed at 0 (advisor always reports evidence) -- control
  E. STATIC_SYCOPHANT: alpha fixed at 1 (advisor always echoes the user)   -- control

We track, per round: alpha_t, o_t, y_t, b_u_t, r_approve_t, r_correct_t, and a
user "trust" signal = EMA of approval (proxy for subjective trust growth,
matching self-report trust scales that track recent satisfying interactions
rather than ground-truth accuracy).

Metrics reported (averaged over independent seeds):
  - final_alpha            : degree of sycophancy the training loop converges to
  - accuracy               : mean 1[o==y] over the last 20% of rounds
  - trust                  : mean EMA-trust over the last 20% of rounds
  - calibration_gap        : |trust - accuracy| at the end (trust/accuracy divergence)
  - sycophancy_index       : Pearson correlation between per-round alpha and 1[o==b_u]
  - regret                 : p_ai - accuracy (accuracy lost relative to an "always
                              report evidence honestly" oracle bound)
"""
import numpy as np
import pandas as pd
import json
import time

RNG_SEED0 = 12345
N_ROUNDS = 4000
N_SEEDS = 30
P_USER = 0.60      # user's own prior accuracy
P_AI = 0.80        # AI evidence accuracy
LR = 0.05          # hill-climbing learning rate on alpha
EMA_BETA = 0.98     # trust EMA smoothing
BASELINE_BETA = 0.95  # reward baseline EMA smoothing (for REINFORCE-style advantage)


def run_condition(condition, seed, lam=0.0, q=1.0, transparency=0.0,
                   fixed_alpha=None, n_rounds=N_ROUNDS):
    rng = np.random.default_rng(seed)
    alpha = 0.0 if fixed_alpha is None else fixed_alpha
    reward_baseline = 0.0
    trust_ema = 0.5

    alphas = np.empty(n_rounds)
    accs = np.empty(n_rounds)
    trusts = np.empty(n_rounds)
    agree_flags = np.empty(n_rounds)

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
            # transparency discounts the boost from pure agreement
            r_approve = agree * (1.0 - transparency)

        if condition in ("APPROVAL_ONLY", "TRANSPARENCY"):
            train_reward = r_approve
        elif condition == "REG":
            # delayed correctness feedback observed with prob q
            if rng.random() < q:
                train_reward = (1 - lam) * r_approve + lam * correct
            else:
                train_reward = r_approve
        else:
            train_reward = None  # static conditions: no update

        trust_ema = EMA_BETA * trust_ema + (1 - EMA_BETA) * agree

        if fixed_alpha is None:
            advantage = train_reward - reward_baseline
            reward_baseline = BASELINE_BETA * reward_baseline + (1 - BASELINE_BETA) * train_reward
            alpha = alpha + LR * advantage * (1 if use_user else -1) * 0.1
            alpha = min(1.0, max(0.0, alpha))

        alphas[t] = alpha
        accs[t] = correct
        trusts[t] = trust_ema
        agree_flags[t] = agree

    return alphas, accs, trusts, agree_flags


def summarize(alphas, accs, trusts, agree_flags):
    tail = max(1, int(0.2 * len(accs)))
    final_alpha = float(alphas[-1])
    accuracy = float(np.mean(accs[-tail:]))
    trust = float(np.mean(trusts[-tail:]))
    calibration_gap = abs(trust - accuracy)
    if np.std(alphas) > 1e-9 and np.std(agree_flags) > 1e-9:
        sycophancy_index = float(np.corrcoef(alphas, agree_flags)[0, 1])
    else:
        sycophancy_index = float("nan")
    regret = P_AI - accuracy
    return dict(final_alpha=final_alpha, accuracy=accuracy, trust=trust,
                calibration_gap=calibration_gap, sycophancy_index=sycophancy_index,
                regret=regret)


def run_many(condition, n_seeds=N_SEEDS, **kwargs):
    rows = []
    for s in range(n_seeds):
        alphas, accs, trusts, agree_flags = run_condition(condition, RNG_SEED0 + s, **kwargs)
        rows.append(summarize(alphas, accs, trusts, agree_flags))
    df = pd.DataFrame(rows)
    return df


def main():
    t0 = time.time()
    results = {}
    traces = {}

    # Condition D: static honest (control, alpha=0)
    df = run_many("STATIC", fixed_alpha=0.0)
    results["STATIC_HONEST"] = df

    # Condition E: static sycophant (control, alpha=1)
    df = run_many("STATIC", fixed_alpha=1.0)
    results["STATIC_SYCOPHANT"] = df

    # Condition A: pure approval-maximizing training
    df = run_many("APPROVAL_ONLY")
    results["APPROVAL_ONLY"] = df

    # Condition B: accuracy-regularized, sweep lambda, delayed feedback prob q=0.3
    for lam in [0.1, 0.3, 0.5, 0.8]:
        df = run_many("REG", lam=lam, q=0.3)
        results[f"REG_lambda{lam}_q0.3"] = df

    # Effect of feedback delay: fix lambda=0.5, sweep q
    for q in [0.05, 0.2, 0.5, 1.0]:
        df = run_many("REG", lam=0.5, q=q)
        results[f"REG_lambda0.5_q{q}"] = df

    # Condition C: transparency mitigation, sweep transparency discount
    for tr in [0.2, 0.5, 0.8]:
        df = run_many("TRANSPARENCY", transparency=tr)
        results[f"TRANSPARENCY_{tr}"] = df

    summary_rows = []
    for name, df in results.items():
        row = {"condition": name}
        for col in df.columns:
            row[f"{col}_mean"] = float(df[col].mean())
            row[f"{col}_std"] = float(df[col].std())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_summary.csv", index=False)

    # one representative trace per key condition for plotting
    trace_conditions = {
        "STATIC_HONEST": dict(condition="STATIC", fixed_alpha=0.0),
        "STATIC_SYCOPHANT": dict(condition="STATIC", fixed_alpha=1.0),
        "APPROVAL_ONLY": dict(condition="APPROVAL_ONLY"),
        "REG_lambda0.5_q0.3": dict(condition="REG", lam=0.5, q=0.3),
        "TRANSPARENCY_0.8": dict(condition="TRANSPARENCY", transparency=0.8),
    }
    trace_data = {}
    for name, kw in trace_conditions.items():
        alphas, accs, trusts, agree_flags = run_condition(seed=RNG_SEED0, **kw)
        # smooth accuracy with rolling window for readability
        window = 100
        acc_roll = pd.Series(accs).rolling(window).mean().values
        trace_data[name] = dict(alpha=alphas.tolist(), acc_roll=acc_roll.tolist(),
                                 trust=trusts.tolist())

    with open("traces.json", "w") as f:
        json.dump(trace_data, f)

    print(summary_df.to_string(index=False))
    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
