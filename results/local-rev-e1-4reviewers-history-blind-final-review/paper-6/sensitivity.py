import numpy as np
from sim import run_condition

# 1. Multi-seed variance for headline s=0 vs s=1 gap (calibrated confidence)
SEEDS = list(range(20))
for s in [0.0, 1.0]:
    trusts, accs, overs = [], [], []
    for seed in SEEDS:
        r = run_condition(s, False, seed=seed)
        trusts.append(r["final_trust_mean"])
        accs.append(r["late_acc_mean"])
        overs.append(r["overreliance"])
    print(f"s={s}: trust {np.mean(trusts):.3f}+/-{np.std(trusts):.3f}, "
          f"acc {np.mean(accs):.3f}+/-{np.std(accs):.3f}, "
          f"overrel {np.mean(overs):.3f}+/-{np.std(overs):.3f}")

# significance of s=0 vs s=1 gap via seed-paired differences
trust0 = [run_condition(0.0, False, seed=sd)["final_trust_mean"] for sd in SEEDS]
trust1 = [run_condition(1.0, False, seed=sd)["final_trust_mean"] for sd in SEEDS]
diffs = np.array(trust0) - np.array(trust1)
print(f"trust gap s0-s1: mean={diffs.mean():.3f} std={diffs.std():.3f} "
      f"min={diffs.min():.3f} max={diffs.max():.3f} (all seeds positive: {np.all(diffs>0)})")

print()
print("=== Competence crossover sensitivity ===")
for ai_comp, user_comp, label in [
    (0.78, 0.55, "AI>User (baseline)"),
    (0.55, 0.78, "AI<User (reversed)"),
    (0.65, 0.65, "AI=User"),
]:
    row = []
    for s in [0.0, 0.5, 1.0]:
        accs = [run_condition(s, False, seed=sd, ai_comp=ai_comp, user_comp=user_comp)["late_acc_mean"]
                for sd in range(10)]
        row.append(np.mean(accs))
    print(f"{label}: s=0 acc={row[0]:.3f}, s=0.5 acc={row[1]:.3f}, s=1 acc={row[2]:.3f}, "
          f"monotonic_decline={row[0] > row[1] > row[2]}")

print()
print("=== LR / slope sensitivity (qualitative monotonicity check) ===")
for lr in [0.06, 0.12, 0.24]:
    for slope_t in [1.5, 3.0, 6.0]:
        vals = [run_condition(s, False, seed=0, lr=lr, slope_t=slope_t)["final_trust_mean"]
                for s in [0.0, 0.4, 1.0]]
        mono = vals[0] > vals[1] > vals[2]
        print(f"lr={lr}, slope_t={slope_t}: trust(s=0,0.4,1)={[round(v,3) for v in vals]} monotonic={mono}")

print()
print("=== Confidence-decoupling overreliance, 20-seed mean (re-verified against current sim.py) ===")
for s in [0.0, 0.4, 1.0]:
    cal = [run_condition(s, False, seed=sd)["overreliance"] for sd in SEEDS]
    dec = [run_condition(s, True, seed=sd)["overreliance"] for sd in SEEDS]
    print(f"s={s}: calibrated {np.mean(cal):.3f}+/-{np.std(cal):.3f}  "
          f"decoupled {np.mean(dec):.3f}+/-{np.std(dec):.3f}")

print()
print("=== Equal-competence 'mild decline': testing trust-update-asymmetry vs confidence-selection explanations ===")
for label, kwargs in [("weak_mult=0.3 (default)", dict()), ("weak_mult=1.0 (no asymmetric discount)", dict(weak_mult=1.0))]:
    accs = []
    for s in [0.0, 0.5, 1.0]:
        vals = [run_condition(s, False, seed=sd, ai_comp=0.65, user_comp=0.65, **kwargs)["late_acc_mean"]
                for sd in range(10)]
        accs.append(np.mean(vals))
    print(f"{label}: acc(s=0,0.5,1)={[round(v,3) for v in accs]}")

cal_s0 = np.mean([run_condition(0.0, False, seed=sd, ai_comp=0.65, user_comp=0.65)["late_acc_mean"] for sd in range(10)])
dec_s0 = np.mean([run_condition(0.0, True, seed=sd, ai_comp=0.65, user_comp=0.65)["late_acc_mean"] for sd in range(10)])
print(f"equal-competence s=0: calibrated conf acc={cal_s0:.3f} vs decoupled (uninformative) conf acc={dec_s0:.3f} "
      f"(user_comp=0.65 floor)")

print()
print("=== Equal-competence: calibrated vs decoupled confidence across the FULL s sweep (new) ===")
for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    cal = np.mean([run_condition(s, False, seed=sd, ai_comp=0.65, user_comp=0.65)["late_acc_mean"] for sd in range(10)])
    dec = np.mean([run_condition(s, True, seed=sd, ai_comp=0.65, user_comp=0.65)["late_acc_mean"] for sd in range(10)])
    print(f"s={s}: calibrated={cal:.3f} decoupled={dec:.3f} gap={cal-dec:+.3f}")

print()
print("=== Paired t-test: baseline s=0 vs s=1 trust gap across 20 seeds (new) ===")
from scipy import stats
trust0 = [run_condition(0.0, False, seed=sd)["final_trust_mean"] for sd in range(20)]
trust1 = [run_condition(1.0, False, seed=sd)["final_trust_mean"] for sd in range(20)]
t_stat, p_val = stats.ttest_rel(trust0, trust1)
print(f"t={t_stat:.3f}, p={p_val:.2e}")
