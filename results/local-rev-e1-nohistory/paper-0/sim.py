"""
Simulation study: how AI sycophancy interacts with adaptive human trust
in a multi-round human-AI decision-making loop.

Grounding: a real logistic-regression classifier trained on the sklearn
breast-cancer dataset supplies realistic, well-calibrated base predictions
p_AI(x) for each task. We then simulate a two-round advice protocol between
this AI and a synthetic human of varying skill, where the AI's *sycophancy
level* lambda controls how much it shifts its stated round-2 recommendation
toward the human's stated belief when the human disagrees. The human's trust
in the AI evolves via a simple reinforcement rule (increase after being
right, decrease after being wrong) and determines, probabilistically,
whether the human follows the AI's final recommendation or their own
judgment.

Outputs: results.csv (raw per-condition aggregates), results_seed_level.csv
(per-seed replicate results for error bars / stats), and three PNG figures,
plus console summary text saved to run_log.txt.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RNG_SEED_BASE = 12345
N_SEEDS = 30                # independent replicate runs (fresh human noise draws)
LAMBDAS = np.round(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.0]), 2)
SKILL_LEVELS = {"low": 0.55, "medium": 0.70, "high": 0.85}  # human's own peak accuracy on easy tasks
TRUST_LR = 0.15             # trust update learning rate
TRUST_INIT = 0.5

def load_ai_predictions():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    p_ai = clf.predict_proba(X_test)[:, 1]  # AI's calibrated P(class=1)
    ai_acc = clf.score(X_test, y_test)
    return p_ai, y_test, ai_acc

def human_judgment(rng, y_true, difficulty, skill_peak):
    """Simulated human forms an independent binary belief.
    Accuracy interpolates from skill_peak (easy tasks) down to 0.5 (hardest tasks)."""
    acc = skill_peak - (skill_peak - 0.5) * (difficulty / difficulty.max())
    correct = rng.random(len(y_true)) < acc
    human_belief = np.where(correct, y_true, 1 - y_true)
    return human_belief, acc

def run_condition(p_ai, y_test, ai_acc, lam, skill_peak, seed):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    r0 = (p_ai > 0.5).astype(int)
    difficulty = np.abs(p_ai - 0.5) * 2  # 0 = ambiguous, 1 = confident
    difficulty = 1 - difficulty          # invert: 0=easy(confident AI), 1=hard(ambiguous) -> use for human acc scaling
    h_belief, _ = human_judgment(rng, y_test, difficulty, skill_peak)

    disagree = (h_belief != r0)
    # sycophantic shift only on disagreement: AI blends its probability toward human's stated belief
    p_shifted = np.where(disagree, (1 - lam) * p_ai + lam * h_belief, p_ai)
    r1 = (p_shifted > 0.5).astype(int)
    flipped_to_wrong = (r0 == y_test) & (r1 != y_test)   # harmful sycophantic flip
    flipped_to_right = (r0 != y_test) & (r1 == y_test)   # beneficial correction

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
    overreliance = (followed & (r1 != y_test) & disagree).mean()  # followed a sycophantic wrong flip
    calib_error = abs(trust_trace[-1] - ai_acc)
    harmful_flip_rate = flipped_to_wrong.mean()
    beneficial_flip_rate = flipped_to_right.mean()
    follow_rate = followed.mean()

    return dict(
        team_acc=team_acc,
        overreliance=overreliance,
        calib_error=calib_error,
        harmful_flip_rate=harmful_flip_rate,
        beneficial_flip_rate=beneficial_flip_rate,
        follow_rate=follow_rate,
        final_trust=trust_trace[-1],
    )

def main():
    t0 = time.time()
    p_ai, y_test, ai_acc = load_ai_predictions()
    print(f"AI base classifier test accuracy: {ai_acc:.4f}  (n_test={len(y_test)})")

    rows = []
    for skill_name, skill_peak in SKILL_LEVELS.items():
        for lam in LAMBDAS:
            for s in range(N_SEEDS):
                seed = RNG_SEED_BASE + s
                res = run_condition(p_ai, y_test, ai_acc, lam, skill_peak, seed)
                res.update(skill=skill_name, skill_peak=skill_peak, lam=lam, seed=s, ai_acc=ai_acc)
                rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv("results_seed_level.csv", index=False)

    agg = df.groupby(["skill", "lam"]).agg(
        team_acc_mean=("team_acc", "mean"), team_acc_std=("team_acc", "std"),
        overreliance_mean=("overreliance", "mean"), overreliance_std=("overreliance", "std"),
        calib_error_mean=("calib_error", "mean"), calib_error_std=("calib_error", "std"),
        harmful_flip_mean=("harmful_flip_rate", "mean"),
        beneficial_flip_mean=("beneficial_flip_rate", "mean"),
        follow_rate_mean=("follow_rate", "mean"),
        final_trust_mean=("final_trust", "mean"),
    ).reset_index()
    agg.to_csv("results.csv", index=False)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"low": "#7570b3", "medium": "#1b9e77", "high": "#d95f02"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for skill in SKILL_LEVELS:
        sub = agg[agg.skill == skill]
        axes[0].errorbar(sub.lam, sub.team_acc_mean, yerr=sub.team_acc_std / np.sqrt(N_SEEDS),
                          marker="o", label=skill, color=colors[skill])
        axes[1].errorbar(sub.lam, sub.overreliance_mean, yerr=sub.overreliance_std / np.sqrt(N_SEEDS),
                          marker="o", label=skill, color=colors[skill])
        axes[2].errorbar(sub.lam, sub.calib_error_mean, yerr=sub.calib_error_std / np.sqrt(N_SEEDS),
                          marker="o", label=skill, color=colors[skill])
    axes[0].axhline(ai_acc, ls="--", c="gray", lw=1, label="AI-alone acc")
    axes[0].set_title("Team accuracy vs sycophancy")
    axes[0].set_xlabel("sycophancy level λ"); axes[0].set_ylabel("team accuracy")
    axes[1].set_title("Overreliance rate vs sycophancy")
    axes[1].set_xlabel("sycophancy level λ"); axes[1].set_ylabel("P(follow harmful flip)")
    axes[2].set_title("Final trust-calibration error vs sycophancy")
    axes[2].set_xlabel("sycophancy level λ"); axes[2].set_ylabel("|trust - true AI acc|")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_main.png", dpi=140)

    fig2, ax = plt.subplots(figsize=(5.5, 4.2))
    for skill in SKILL_LEVELS:
        sub = agg[agg.skill == skill]
        ax.plot(sub.lam, sub.harmful_flip_mean, marker="o", color=colors[skill], label=f"{skill} harmful")
        ax.plot(sub.lam, sub.beneficial_flip_mean, marker="s", ls="--", color=colors[skill], label=f"{skill} beneficial")
    ax.set_xlabel("sycophancy level λ"); ax.set_ylabel("flip rate")
    ax.set_title("Harmful vs beneficial AI answer flips")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("fig_flips.png", dpi=140)

    elapsed = time.time() - t0
    summary = {
        "ai_acc": ai_acc,
        "n_test": int(len(y_test)),
        "n_seeds": N_SEEDS,
        "lambdas": LAMBDAS.tolist(),
        "elapsed_sec": elapsed,
    }
    with open("run_log.txt", "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n\n")
        f.write(agg.to_string(index=False))
    print(agg.to_string(index=False))
    print(f"\nDone in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
