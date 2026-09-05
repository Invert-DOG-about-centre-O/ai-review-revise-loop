import numpy as np
import csv, json, sys

P_USER = 0.60
P_AI = 0.80
LR = 0.05
BASELINE_BETA = 0.95
TRUST_BETA = 0.98
N_ROUNDS = 4000
N_SEEDS = 30
STEP_SCALE = 0.1


def run_one(seed, condition, lam=0.0, q=0.0, transparency=0.0, lr=LR,
            baseline_beta=BASELINE_BETA, fallback="approval", baseline_init=0.5,
            return_trace=False, update_rule="hillclimb"):
    rng = np.random.default_rng(seed)
    alpha = 0.5
    baseline = baseline_init
    trust = 0.5
    trust_acc = 0.5
    alphas = np.zeros(N_ROUNDS)
    accs = np.zeros(N_ROUNDS)
    trusts = np.zeros(N_ROUNDS)
    agree = np.zeros(N_ROUNDS)

    static = condition in ("STATIC_HONEST", "STATIC_SYCOPHANT")
    if condition == "STATIC_HONEST":
        alpha = 0.0
    elif condition == "STATIC_SYCOPHANT":
        alpha = 1.0

    for t in range(N_ROUNDS):
        y = rng.random() < 0.5
        b_u = y if rng.random() < P_USER else (not y)
        e = y if rng.random() < P_AI else (not y)

        use_user = rng.random() < alpha
        o = b_u if use_user else e

        r_approve = float(o == b_u)
        r_correct = float(o == y)

        observed = None
        if condition == "APPROVAL_ONLY" or static:
            reward = r_approve
        elif condition == "REG":
            observed = rng.random() < q
            if observed:
                reward = (1 - lam) * r_approve + lam * r_correct
            else:
                if fallback == "approval":
                    reward = r_approve
                else:
                    reward = None
        elif condition == "TRANSPARENCY":
            if o == b_u:
                reward = (1 - transparency) * r_approve
            else:
                reward = r_approve
        else:
            raise ValueError(condition)

        if not static and reward is not None:
            advantage = reward - baseline
            if update_rule == "hillclimb":
                direction = 1.0 if use_user else -1.0
                alpha = alpha + lr * advantage * direction * STEP_SCALE
            elif update_rule == "pg":
                # Standard REINFORCE score-function gradient for a Bernoulli
                # policy pi(use_user)=alpha: d/dalpha log pi = 1/alpha or
                # -1/(1-alpha). Clipped away from 0/1 to avoid blow-up.
                a_safe = min(max(alpha, 1e-3), 1 - 1e-3)
                score = (1.0 / a_safe) if use_user else (-1.0 / (1 - a_safe))
                alpha = alpha + lr * advantage * score * STEP_SCALE
            else:
                raise ValueError(update_rule)
            alpha = min(1.0, max(0.0, alpha))
            baseline = baseline_beta * baseline + (1 - baseline_beta) * reward

        trust = TRUST_BETA * trust + (1 - TRUST_BETA) * r_approve
        # Alternative trust proxy: an EMA of r_correct, but only updated on
        # rounds where correctness is actually revealed (same q as REG's
        # partial observability), to test whether the calibration gap is
        # structural (no correctness channel exists) rather than an artifact
        # of using raw agreement.
        if observed:
            trust_acc = TRUST_BETA * trust_acc + (1 - TRUST_BETA) * r_correct

        alphas[t] = alpha
        accs[t] = r_correct
        trusts[t] = trust
        agree[t] = r_approve

    tail = int(N_ROUNDS * 0.2)
    final_alpha = alphas[-tail:].mean()
    accuracy = accs[-tail:].mean()
    trust_mean = trusts[-tail:].mean()
    calib_gap = abs(trust_mean - accuracy)
    calib_gap_acctrust = abs(trust_acc - accuracy)
    if np.std(alphas) > 1e-9 and np.std(agree) > 1e-9:
        syco_idx = np.corrcoef(alphas, agree)[0, 1]
    else:
        syco_idx = 0.0
    regret = P_AI - accuracy
    out = dict(final_alpha=final_alpha, accuracy=accuracy, trust=trust_mean,
               calibration_gap=calib_gap, calibration_gap_acctrust=calib_gap_acctrust,
               sycophancy_index=syco_idx, regret=regret)
    if return_trace:
        out["alpha_trace"] = alphas
    return out


def run_condition(name, condition, seeds=N_SEEDS, **kwargs):
    rows = [run_one(s, condition, **kwargs) for s in range(seeds)]
    agg = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows])
        agg[k + "_mean"] = vals.mean()
        agg[k + "_std"] = vals.std()
        agg[k + "_vals"] = vals.tolist()
    agg["name"] = name
    return agg


def main():
    results = []
    results.append(run_condition("Static honest", "STATIC_HONEST"))
    results.append(run_condition("Static sycophant", "STATIC_SYCOPHANT"))
    results.append(run_condition("Approval-only", "APPROVAL_ONLY"))

    for lam in [0.1, 0.3, 0.5, 0.8]:
        results.append(run_condition(f"REG lam={lam} q=0.3", "REG", lam=lam, q=0.3))
    for q in [0.05, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        results.append(run_condition(f"REG lam=0.5 q={q}", "REG", lam=0.5, q=q))
    for t in [0.2, 0.5, 0.8]:
        results.append(run_condition(f"Transparency {t}", "TRANSPARENCY", transparency=t))

    # No-fallback REG variant (unobserved rounds skip the alpha update)
    for q in [0.3, 0.5, 1.0]:
        results.append(run_condition(f"REG-noFB lam=0.5 q={q}", "REG", lam=0.5, q=q, fallback="none"))

    # REG-fair: q=1.0 (no missing-label confound), swept over lambda.
    # Promoted to a first-class, saved condition (reviewer request) rather
    # than a one-off manual reconstruction.
    for lam in [0.5, 0.8, 1.0]:
        results.append(run_condition(f"REG-fair lam={lam} q=1.0", "REG", lam=lam, q=1.0))

    with open("results_summary.csv", "w", newline="") as f:
        keys = [k for k in results[0] if not k.endswith("_vals")]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})

    with open("results_raw.json", "w") as f:
        json.dump(results, f)

    for r in results:
        print(r["name"], "alpha=%.3f" % r["final_alpha_mean"], "acc=%.3f" % r["accuracy_mean"],
              "trust=%.3f" % r["trust_mean"], "gap=%.3f" % r["calibration_gap_mean"],
              "syco=%.3f" % r["sycophancy_index_mean"])


if __name__ == "__main__":
    main()
