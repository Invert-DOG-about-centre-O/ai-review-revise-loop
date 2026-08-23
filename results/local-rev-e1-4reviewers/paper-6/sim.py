"""
Agent-based simulation of a human-AI trust/sycophancy feedback loop.

Each simulated "dyad" consists of a human user with an evolving trust level T
and an AI advisor with a fixed sycophancy level s. Over many rounds, both
parties give an answer to a binary task; the AI's stated answer is a mix of
its private (competence-driven) judgement and the user's stated prior belief,
controlled by s. The user decides whether to follow the AI based on current
trust and the AI's expressed confidence, then trust is updated from the
observed outcome (ground truth is revealed each round, as in many real
decision-support settings, e.g. diagnosis-then-biopsy, forecast-then-outcome).

We compare sycophancy levels s in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and, as an
ablation, whether the AI's expressed confidence is calibrated (tracks its
true internal correctness probability) or decoupled/inflated (sycophantic
answers are stated with the same high confidence as genuine ones).
"""
import numpy as np
import json
import time

RNG_SEED = 0
N_DYADS = 400          # Monte Carlo dyads per condition
N_ROUNDS = 60          # interactions per dyad
AI_COMPETENCE = 0.78   # P(AI's private judgement is correct)
USER_COMPETENCE = 0.55 # P(user's own prior belief is correct)
LR = 0.12              # trust update learning rate
SYCOPHANCY_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_condition(s, calibrated_confidence, rng):
    """Run N_DYADS dyads of N_ROUNDS rounds for sycophancy level s.
    Returns dict of per-round arrays (mean over dyads) and summary scalars.
    """
    trust_traj = np.zeros((N_DYADS, N_ROUNDS))
    acc_traj = np.zeros((N_DYADS, N_ROUNDS))
    overreliance = np.zeros((N_DYADS, N_ROUNDS))  # followed AI, AI wrong
    underreliance = np.zeros((N_DYADS, N_ROUNDS)) # ignored AI, AI right, user wrong

    T = np.full(N_DYADS, 0.5)  # initial trust

    for t in range(N_ROUNDS):
        ground_truth = rng.integers(0, 2, size=N_DYADS)

        # user's private belief: correct w.p. USER_COMPETENCE
        user_correct_mask = rng.random(N_DYADS) < USER_COMPETENCE
        user_belief = np.where(user_correct_mask, ground_truth, 1 - ground_truth)

        # AI's private judgement: correct w.p. AI_COMPETENCE
        ai_correct_mask = rng.random(N_DYADS) < AI_COMPETENCE
        ai_private = np.where(ai_correct_mask, ground_truth, 1 - ground_truth)

        # sycophancy: when AI's private judgement disagrees with the user's
        # stated belief, the AI defers to the user with probability s
        disagree = ai_private != user_belief
        defer = disagree & (rng.random(N_DYADS) < s)
        ai_stated = np.where(defer, user_belief, ai_private)
        deferred_and_wrong = defer & (ai_stated != ground_truth)

        # expressed confidence: calibrated version tracks true correctness;
        # decoupled version states high confidence even when deferring
        true_correct = (ai_stated == ground_truth).astype(float)
        if calibrated_confidence:
            conf = np.where(true_correct == 1, 0.85, 0.55) + rng.normal(0, 0.03, N_DYADS)
        else:
            # sycophantic deferrals are stated just as confidently as genuine answers
            conf = np.full(N_DYADS, 0.85) + rng.normal(0, 0.03, N_DYADS)
        conf = np.clip(conf, 0.0, 1.0)

        # user's decision to follow AI: logistic in (trust, confidence)
        p_follow = sigmoid(3.0 * (T - 0.5) + 2.0 * (conf - 0.5))
        follow = rng.random(N_DYADS) < p_follow

        final_answer = np.where(follow, ai_stated, user_belief)
        outcome_correct = (final_answer == ground_truth).astype(float)

        # trust update from realized outcome of the AI's stated answer
        ai_was_right = (ai_stated == ground_truth).astype(float)
        T = T + LR * follow * (ai_was_right - T)          # reinforced when relied on
        T = T + LR * (1 - follow) * 0.3 * (ai_was_right - T)  # weak update when not relied on
        T = np.clip(T, 0.0, 1.0)

        trust_traj[:, t] = T
        acc_traj[:, t] = outcome_correct
        overreliance[:, t] = follow & (ai_stated != ground_truth)
        underreliance[:, t] = (~follow) & (ai_stated == ground_truth) & (user_belief != ground_truth)

    summary = {
        "s": s,
        "calibrated_confidence": calibrated_confidence,
        "final_trust_mean": float(trust_traj[:, -10:].mean()),
        "final_trust_std": float(trust_traj[:, -10:].mean(axis=1).std()),
        "mean_accuracy": float(acc_traj.mean()),
        "late_accuracy": float(acc_traj[:, -10:].mean()),
        "overreliance_rate": float(overreliance.mean()),
        "underreliance_rate": float(underreliance.mean()),
        "trust_trajectory": trust_traj.mean(axis=0).tolist(),
        "accuracy_trajectory": acc_traj.mean(axis=0).tolist(),
    }
    return summary

def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    results = []
    for calibrated in [True, False]:
        for s in SYCOPHANCY_LEVELS:
            res = run_condition(s, calibrated, rng)
            results.append(res)
            print(f"calibrated={calibrated} s={s:.1f} "
                  f"final_trust={res['final_trust_mean']:.3f} "
                  f"late_acc={res['late_accuracy']:.3f} "
                  f"overreliance={res['overreliance_rate']:.4f} "
                  f"underreliance={res['underreliance_rate']:.4f}")

    with open("results.json", "w") as f:
        json.dump({
            "config": {
                "N_DYADS": N_DYADS, "N_ROUNDS": N_ROUNDS,
                "AI_COMPETENCE": AI_COMPETENCE, "USER_COMPETENCE": USER_COMPETENCE,
                "LR": LR, "SYCOPHANCY_LEVELS": SYCOPHANCY_LEVELS,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nTotal runtime: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
