"""
Human-AI sycophancy simulation.

Grounds an "AI advisor" in a real logistic-regression classifier trained on the
sklearn/UCI breast-cancer dataset, couples it to a synthetic human with a skill
parameter, an explicit disagreement-triggered sycophantic shift (parameter
lambda), and a reinforcement-learning trust update. Sweeps lambda x skill x
seed for three mechanism variants (binary blend, sigmoid blend, continuous
belief blend) and writes results_ablation.csv.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ALPHA = 0.15          # trust learning rate
K_SIGMOID = 8.0        # sigmoid steepness for the sigmoid-blend ablation
SKILLS = {"low": 0.55, "medium": 0.70, "high": 0.85}
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.0]
N_SEEDS = 30
VARIANTS = ["binary", "sigmoid", "continuous"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_ai():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    p_ai = clf.predict_proba(X_test)[:, 1]
    r0 = (p_ai > 0.5).astype(int)
    ai_acc = (r0 == y_test).mean()
    return p_ai, y_test, ai_acc


def run_condition(p_ai, y_test, ai_acc, skill, lam, variant, seed):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    r0 = (p_ai > 0.5).astype(int)
    difficulty = 1.0 - np.abs(2.0 * p_ai - 0.5)
    human_acc = skill - (skill - 0.5) * difficulty
    human_correct = rng.random(n) < human_acc
    h = np.where(human_correct, y_test, 1 - y_test)

    disagree = h != r0
    p_shifted = p_ai.copy()

    if variant == "binary":
        p_shifted[disagree] = (1 - lam) * p_ai[disagree] + lam * h[disagree]
    elif variant == "sigmoid":
        pull = sigmoid(K_SIGMOID * (lam - 0.5))
        p_shifted[disagree] = (1 - pull) * p_ai[disagree] + pull * h[disagree]
    elif variant == "continuous":
        mag = np.abs(rng.normal(0.3, 0.15, size=n))
        h_conf = np.clip(np.where(h == 1, 0.5 + mag, 0.5 - mag), 0.0, 1.0)
        p_shifted[disagree] = (1 - lam) * p_ai[disagree] + lam * h_conf[disagree]
    else:
        raise ValueError(variant)

    r1 = (p_shifted > 0.5).astype(int)

    trust = 0.5
    team_decision = np.empty(n, dtype=int)
    follow_draws = rng.random(n)
    for i in range(n):
        follow_ai = follow_draws[i] < trust
        decision = r1[i] if follow_ai else h[i]
        team_decision[i] = decision
        correct = int(decision == y_test[i])
        trust = trust + ALPHA * (correct - trust)

    team_acc = (team_decision == y_test).mean()
    overreliance = np.mean((team_decision != y_test) & (h == y_test))
    calib_error = abs(trust - ai_acc)
    harmful_flip = np.mean(disagree & (r1 != r0) & (r0 == y_test) & (r1 != y_test))
    beneficial_flip = np.mean(disagree & (r1 != r0) & (r0 != y_test) & (r1 == y_test))

    return {
        "variant": variant,
        "skill": skill,
        "lambda": lam,
        "seed": seed,
        "team_acc": team_acc,
        "overreliance": overreliance,
        "calib_error": calib_error,
        "harmful_flip": harmful_flip,
        "beneficial_flip": beneficial_flip,
        "final_trust": trust,
    }


def main():
    p_ai, y_test, ai_acc = train_ai()
    print(f"AI-alone held-out accuracy: {ai_acc:.4f} on {len(y_test)} cases")

    rows = []
    for variant in VARIANTS:
        for skill_name, skill_val in SKILLS.items():
            for lam in LAMBDAS:
                for seed in range(N_SEEDS):
                    r = run_condition(p_ai, y_test, ai_acc, skill_val, lam, variant, seed)
                    r["skill_name"] = skill_name
                    rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv("results_ablation.csv", index=False)
    print(f"Total runs: {len(df)}")

    summary = df.groupby(["variant", "skill_name", "lambda"]).agg(
        team_acc=("team_acc", "mean"),
        overreliance=("overreliance", "mean"),
        calib_error=("calib_error", "mean"),
    ).reset_index()
    summary.to_csv("results_summary.csv", index=False)

    for variant in VARIANTS:
        print(f"\n=== variant={variant} ===")
        for skill_name in ["low", "medium", "high"]:
            sub = summary[(summary.variant == variant) & (summary.skill_name == skill_name)]
            print(f"-- skill={skill_name} --")
            print(sub[["lambda", "team_acc", "overreliance", "calib_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
