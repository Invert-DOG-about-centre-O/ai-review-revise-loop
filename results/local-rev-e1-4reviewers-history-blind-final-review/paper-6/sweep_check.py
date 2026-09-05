import numpy as np
from sim import run_condition

print("=== Equal-competence: calibrated vs decoupled confidence across full s sweep ===")
S_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for s in S_VALS:
    cal = np.mean([run_condition(s, False, seed=sd, ai_comp=0.65, user_comp=0.65)["late_acc_mean"] for sd in range(10)])
    dec = np.mean([run_condition(s, True, seed=sd, ai_comp=0.65, user_comp=0.65)["late_acc_mean"] for sd in range(10)])
    print(f"s={s}: calibrated={cal:.3f} decoupled={dec:.3f} gap={cal-dec:+.3f}")

print()
print("=== Paired t-test: baseline s=0 vs s=1 trust gap across 20 seeds ===")
from scipy import stats
trust0 = [run_condition(0.0, False, seed=sd)["final_trust_mean"] for sd in range(20)]
trust1 = [run_condition(1.0, False, seed=sd)["final_trust_mean"] for sd in range(20)]
t, p = stats.ttest_rel(trust0, trust1)
print(f"t={t:.3f}, p={p:.2e}")
