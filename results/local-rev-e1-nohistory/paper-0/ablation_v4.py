"""
Round-3 review follow-ups:

(D) Fine-grained (delta-lambda=0.01) sweep under the LOGIT (sigmoid) blend,
    mirroring 4.3's fine grid but for the mechanism used in 4.5, to check
    whether the "smooth decline" there is actually a coarse-grid illusion
    that resolves into its own narrow band at fine resolution (round-3
    review question 3).

(E) Sensitivity of the "narrow-band transition" classification to the
    drop-threshold used to flag it (round-3 review question 1 / weakness 2):
    recompute max single-step drop (over delta-lambda=0.01) for both linear
    and sigmoid fine grids, and report the classification under thresholds
    {0.02, 0.03, 0.05} instead of just the one arbitrary cutoff (0.03) used
    before.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RNG_SEED_BASE = 12345
N_SEEDS = 30
SKILL_LEVELS = {"low": 0.55, "medium": 0.70, "high": 0.85}
TRUST_LR = 0.15
TRUST_INIT = 0.5

def load_ai_predictions():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    p_ai = clf.predict_proba(X_test)[:, 1]
    ai_acc = clf.score(X_test, y_test)
    return p_ai, y_test, ai_acc

def acc_curve(difficulty, skill_peak):
    return skill_peak - (skill_peak - 0.5) * (difficulty / difficulty.max())

def human_judgment(rng, y_true, acc, conf_map):
    correct = rng.random(len(y_true)) < acc
    h_belief = np.where(correct, y_true, 1 - y_true)
    h_prob = np.where(correct, 0.5 + 0.5 * conf_map(acc), 0.5 - 0.5 * conf_map(acc))
    return h_belief, h_prob

def run_condition(p_ai, y_test, ai_acc, lam, skill_peak, seed, conf_map, blend):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    r0 = (p_ai > 0.5).astype(int)
    difficulty = 1 - np.abs(p_ai - 0.5) * 2
    acc = acc_curve(difficulty, skill_peak)
    h_belief, h_prob = human_judgment(rng, y_test, acc, conf_map)

    disagree = (h_belief != r0)
    if blend == "linear":
        p_shifted = np.where(disagree, (1 - lam) * p_ai + lam * h_prob, p_ai)
    elif blend == "sigmoid":
        eps = 1e-6
        p_c = np.clip(p_ai, eps, 1 - eps)
        h_c = np.clip(h_prob, eps, 1 - eps)
        logit_ai = np.log(p_c / (1 - p_c))
        logit_h = np.log(h_c / (1 - h_c))
        logit_shift = (1 - lam) * logit_ai + lam * logit_h
        p_shifted = np.where(disagree, 1 / (1 + np.exp(-logit_shift)), p_ai)
    r1 = (p_shifted > 0.5).astype(int)

    trust = TRUST_INIT
    team_decision = np.zeros(n, dtype=int)
    for i in range(n):
        follow_ai = rng.random() < trust
        team_decision[i] = r1[i] if follow_ai else h_belief[i]
        trust = trust + TRUST_LR * (int(r1[i] == y_test[i]) - trust)
    team_acc = (team_decision == y_test).mean()
    return team_acc

def sweep(p_ai, y_test, ai_acc, lambdas, conf_map, blend):
    rows = []
    for skill_name, skill_peak in SKILL_LEVELS.items():
        for lam in lambdas:
            accs = [run_condition(p_ai, y_test, ai_acc, lam, skill_peak, RNG_SEED_BASE + s, conf_map, blend)
                    for s in range(N_SEEDS)]
            rows.append(dict(skill=skill_name, lam=round(float(lam), 3), team_acc_mean=np.mean(accs)))
    return pd.DataFrame(rows)

def max_step_drop(df, skill):
    sub = df[df.skill == skill].sort_values("lam").reset_index(drop=True)
    diffs = sub.team_acc_mean.diff()
    worst_idx = diffs.idxmin()
    return -diffs.min(), sub.lam.iloc[worst_idx - 1], sub.lam.iloc[worst_idx]

def classify(max_drop, thresh):
    return "narrow-band" if max_drop > thresh else "smooth"

def main():
    t0 = time.time()
    p_ai, y_test, ai_acc = load_ai_predictions()

    # (D) fine-grained sigmoid-blend sweep, same resolution/range as 4.3's linear fine grid
    fine_lambdas = np.round(np.arange(0.40, 1.00, 0.01), 3)
    fine_sig_df = sweep(p_ai, y_test, ai_acc, fine_lambdas, conf_map=lambda a: a, blend="sigmoid")
    fine_sig_df.to_csv("results_fine_sigmoid.csv", index=False)

    # for direct comparison, also rerun the linear fine grid over the SAME extended range
    fine_lin_df = sweep(p_ai, y_test, ai_acc, fine_lambdas, conf_map=lambda a: a, blend="linear")

    print("=== (D) Fine-grained (delta=0.01) max single-step drop, linear vs sigmoid blend ===")
    summary = {}
    for skill in SKILL_LEVELS:
        lin_drop, lin_a, lin_b = max_step_drop(fine_lin_df, skill)
        sig_drop, sig_a, sig_b = max_step_drop(fine_sig_df, skill)
        summary[skill] = dict(linear_max_drop=round(float(lin_drop), 4), linear_at=f"{lin_a}->{lin_b}",
                               sigmoid_max_drop=round(float(sig_drop), 4), sigmoid_at=f"{sig_a}->{sig_b}")
        print(f"  {skill}: linear max drop={lin_drop:.4f} at {lin_a}->{lin_b}   "
              f"sigmoid max drop={sig_drop:.4f} at {sig_a}->{sig_b}")

    print("\n=== (E) Classification sensitivity to drop-threshold ===")
    thresh_results = {}
    for thresh in [0.02, 0.03, 0.05]:
        row = {}
        for skill in SKILL_LEVELS:
            lin_drop = summary[skill]["linear_max_drop"]
            sig_drop = summary[skill]["sigmoid_max_drop"]
            row[skill] = dict(linear=classify(lin_drop, thresh), sigmoid=classify(sig_drop, thresh))
        thresh_results[thresh] = row
        print(f"  thresh={thresh}: " + ", ".join(f"{s}: linear={row[s]['linear']}/sigmoid={row[s]['sigmoid']}" for s in SKILL_LEVELS))

    elapsed = time.time() - t0
    with open("run_log_v4.txt", "w") as f:
        f.write(f"elapsed_sec={elapsed:.2f}\n\n")
        f.write("(D) max single-step drop, linear vs sigmoid (delta=0.01, lam in [0.40,0.99]):\n")
        f.write(json.dumps(summary, indent=2) + "\n\n")
        f.write("(E) classification sensitivity across thresholds:\n")
        f.write(json.dumps(thresh_results, indent=2) + "\n")
    print(f"\nElapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
