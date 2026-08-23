"""
Extension experiments addressing reviewer concerns on v1:
1. Multi-seed variance/CI for the headline s=0 vs s=1 gap (calibrated confidence).
2. Competence-crossing sensitivity sweep: does the direction of the sycophancy
   effect depend on AI_COMPETENCE > USER_COMPETENCE, or is it a generic effect?
Reuses run_condition from sim.py unmodified.
"""
import numpy as np
import json
import time
import sim as base

N_SEEDS = 20

def multi_seed_gap():
    finals = {0.0: [], 1.0: []}
    overs = {0.0: [], 1.0: []}
    accs = {0.0: [], 1.0: []}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for s in [0.0, 1.0]:
            res = base.run_condition(s, True, rng)
            finals[s].append(res["final_trust_mean"])
            overs[s].append(res["overreliance_rate"])
            accs[s].append(res["late_accuracy"])
    out = {}
    for s in [0.0, 1.0]:
        out[s] = {
            "final_trust_mean_across_seeds": float(np.mean(finals[s])),
            "final_trust_std_across_seeds": float(np.std(finals[s])),
            "late_acc_mean_across_seeds": float(np.mean(accs[s])),
            "late_acc_std_across_seeds": float(np.std(accs[s])),
            "overreliance_mean_across_seeds": float(np.mean(overs[s])),
            "overreliance_std_across_seeds": float(np.std(overs[s])),
        }
    gap_trust = np.array(finals[1.0]) - np.array(finals[0.0])
    gap_acc = np.array(accs[1.0]) - np.array(accs[0.0])
    out["gap_trust_mean"] = float(np.mean(gap_trust))
    out["gap_trust_std"] = float(np.std(gap_trust))
    out["gap_acc_mean"] = float(np.mean(gap_acc))
    out["gap_acc_std"] = float(np.std(gap_acc))
    return out

def competence_crossing():
    """Sweep AI_COMPETENCE, USER_COMPETENCE combinations at s=0 and s=1
    to test whether the sign of the sycophancy effect is generic or
    depends on AI being more competent than the user."""
    combos = [
        ("AI>user (base)", 0.78, 0.55),
        ("equal competence", 0.65, 0.65),
        ("user>AI (reversed)", 0.55, 0.78),
    ]
    results = []
    for label, ai_c, user_c in combos:
        base.AI_COMPETENCE = ai_c
        base.USER_COMPETENCE = user_c
        rng = np.random.default_rng(0)
        row = {"label": label, "AI_COMPETENCE": ai_c, "USER_COMPETENCE": user_c}
        for s in [0.0, 1.0]:
            res = base.run_condition(s, True, rng)
            row[f"final_trust_s{s}"] = res["final_trust_mean"]
            row[f"late_acc_s{s}"] = res["late_accuracy"]
        row["trust_delta_s1_minus_s0"] = row["final_trust_s1.0"] - row["final_trust_s0.0"]
        row["acc_delta_s1_minus_s0"] = row["late_acc_s1.0"] - row["late_acc_s0.0"]
        results.append(row)
    # restore defaults
    base.AI_COMPETENCE = 0.78
    base.USER_COMPETENCE = 0.55
    return results

def main():
    t0 = time.time()
    gap = multi_seed_gap()
    cross = competence_crossing()
    out = {"multi_seed_gap": gap, "competence_crossing": cross}
    with open("results_extra.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nExtra runtime: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
