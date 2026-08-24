"""
Ablation requested by review: is the sharp lambda=0.5 threshold specific to
representing the human's belief as a *binary* label h in {0,1}, or does it
survive if the human instead reports a *continuous* confidence?

Everything is identical to sim.py except one line: instead of blending the
AI's probability toward a hard binary belief, we blend it toward a continuous
human confidence h_prob in (0,1), constructed so a human with the same skill
curve as before is right/wrong at exactly the same rate, but now states a
graded confidence (0.5 + 0.5*acc if their belief matches the truth, else
0.5 - 0.5*acc) rather than a flat 0 or 1.
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
LAMBDAS = np.round(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.0]), 2)
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

def human_judgment_continuous(rng, y_true, difficulty, skill_peak):
    """Same accuracy curve as sim.py, but the human states a continuous
    confidence h_prob in (0,1) instead of a hard binary belief."""
    acc = skill_peak - (skill_peak - 0.5) * (difficulty / difficulty.max())
    correct = rng.random(len(y_true)) < acc
    h_belief = np.where(correct, y_true, 1 - y_true)  # kept for reference/consistency
    h_prob = np.where(correct, 0.5 + 0.5 * acc, 0.5 - 0.5 * acc)
    return h_belief, h_prob, acc

def run_condition(p_ai, y_test, ai_acc, lam, skill_peak, seed):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    r0 = (p_ai > 0.5).astype(int)
    difficulty = np.abs(p_ai - 0.5) * 2
    difficulty = 1 - difficulty
    h_belief, h_prob, _ = human_judgment_continuous(rng, y_test, difficulty, skill_peak)

    disagree = (h_belief != r0)
    p_shifted = np.where(disagree, (1 - lam) * p_ai + lam * h_prob, p_ai)
    r1 = (p_shifted > 0.5).astype(int)
    flipped_to_wrong = (r0 == y_test) & (r1 != y_test)
    flipped_to_right = (r0 != y_test) & (r1 == y_test)

    trust = TRUST_INIT
    followed = np.zeros(n, dtype=bool)
    team_decision = np.zeros(n, dtype=int)
    trust_trace = np.zeros(n)
    for i in range(n):
        follow_ai = rng.random() < trust
        followed[i] = follow_ai
        team_decision[i] = r1[i] if follow_ai else h_belief[i]
        ai_was_right = int(r1[i] == y_test[i])
        trust = trust + TRUST_LR * (ai_was_right - trust)
        trust_trace[i] = trust

    team_acc = (team_decision == y_test).mean()
    overreliance = (followed & (r1 != y_test) & disagree).mean()
    calib_error = abs(trust_trace[-1] - ai_acc)
    harmful_flip_rate = flipped_to_wrong.mean()
    beneficial_flip_rate = flipped_to_right.mean()

    return dict(team_acc=team_acc, overreliance=overreliance, calib_error=calib_error,
                harmful_flip_rate=harmful_flip_rate, beneficial_flip_rate=beneficial_flip_rate)

def main():
    t0 = time.time()
    p_ai, y_test, ai_acc = load_ai_predictions()
    rows = []
    for skill_name, skill_peak in SKILL_LEVELS.items():
        for lam in LAMBDAS:
            for s in range(N_SEEDS):
                seed = RNG_SEED_BASE + s
                res = run_condition(p_ai, y_test, ai_acc, lam, skill_peak, seed)
                res.update(skill=skill_name, lam=lam, seed=s)
                rows.append(res)
    df = pd.DataFrame(rows)
    agg = df.groupby(["skill", "lam"]).agg(
        team_acc_mean=("team_acc", "mean"),
        overreliance_mean=("overreliance", "mean"),
        calib_error_mean=("calib_error", "mean"),
    ).reset_index()
    agg.to_csv("results_continuous_ablation.csv", index=False)
    print(agg.to_string(index=False))

    # step-size diagnostic: largest single-step drop in team_acc_mean vs lambda, per skill
    print("\nMax single-step drop in team_acc_mean between adjacent lambda values:")
    for skill in SKILL_LEVELS:
        sub = agg[agg.skill == skill].sort_values("lam")
        diffs = sub.team_acc_mean.diff().dropna()
        print(f"  {skill}: max drop = {diffs.min():.4f} (binary version had a ~0.13-0.36 single-step drop at lam=0.5)")
    print(f"\nElapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
