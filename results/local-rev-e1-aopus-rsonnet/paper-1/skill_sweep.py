"""
Skill sweep (revision r3): the round-3 review's top weakness is that the
'skill, not shift-size, drives transfer degradation' claim rests on only TWO
skill regimes. Here we sweep in-domain sharpness tau_in over an intermediate
grid while holding the SHIFT RATIO ~constant (tau_out = 1.56 * tau_in), so any
monotone trend in transfer degradation is attributable to model skill, not to a
bigger distributional gap. Reuses experiment_robust's model/calibration logic.
Also runs a non-parametric bootstrap on the 5 paired ECATS-minus-global ECE
differences to corroborate the t-test (review Q2).
"""
import time, json
import numpy as np
from scipy import stats
import experiment_robust as E

t0 = time.time()

# --- (1) Skill sweep: matched shift ratio 1.56x, 3 seeds each ---
RATIO = 1.56
TAUS = [0.45, 0.6, 0.8, 1.0]
SEEDS = [0, 1, 2]
sweep = []
for tau_in in TAUS:
    tau_out = round(tau_in * RATIO, 3)
    runs = [E.run_one(s, tau_in, tau_out) for s in SEEDS]
    acc = np.mean([r["in_domain"]["acc"] for r in runs])
    Tg = np.mean([r["T_global"] for r in runs])
    raw = np.mean([r["shift"]["raw_ece"] for r in runs])
    glob = np.mean([r["shift"]["glob_ece"] for r in runs])
    red = 100.0 * (1 - glob / raw)
    id_raw = np.mean([r["in_domain"]["raw_ece"] for r in runs])
    id_glob = np.mean([r["in_domain"]["glob_ece"] for r in runs])
    id_red = 100.0 * (1 - id_glob / id_raw)
    row = {"tau_in": tau_in, "tau_out": tau_out, "acc": acc, "T_global": Tg,
           "shift_raw_ece": raw, "shift_glob_ece": glob,
           "shift_reduction_pct": red, "indomain_reduction_pct": id_red}
    sweep.append(row)
    print(f"tau_in={tau_in} tau_out={tau_out} acc={acc:.3f} T*={Tg:.2f} "
          f"shift_red={red:.1f}% (raw {raw:.3f}->glob {glob:.3f})  "
          f"[t={time.time()-t0:.0f}s]")

# --- (2) Bootstrap the paired ECATS-minus-global ECE differences (review Q2) ---
d = json.load(open("results_robust.json"))
def boot_paired(regime, split, n_boot=10000, seed=0):
    runs = d[regime]["runs"]
    g = np.array([r[split]["glob_ece"] for r in runs])
    a = np.array([r[split]["adap_ece"] for r in runs])
    x = a - g
    r = np.random.default_rng(seed)
    means = np.array([r.choice(x, size=len(x), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return x.mean(), lo, hi
print("\n=== Bootstrap on paired ECATS-minus-global ECE differences (n=5 seeds) ===")
for regime, lab in [("regime_A_overconfident", "A"), ("regime_B_higher_skill", "B")]:
    for split in ("in_domain", "shift"):
        m, lo, hi = boot_paired(regime, split)
        excl = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"regime {lab} {split:9s} mean={m:+.5f} 95%CI=[{lo:+.5f},{hi:+.5f}] ({excl})")

json.dump({"skill_sweep": sweep, "runtime_s": time.time() - t0},
          open("skill_sweep.json", "w"), indent=2)
print(f"\nTOTAL {time.time()-t0:.1f}s")
