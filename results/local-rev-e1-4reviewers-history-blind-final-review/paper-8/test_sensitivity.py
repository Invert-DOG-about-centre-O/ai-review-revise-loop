import numpy as np
from sim import run_condition

for binit in [0.0, 0.5]:
    r = run_condition(f"TRANSPARENCY 0.8 binit={binit}", "TRANSPARENCY", transparency=0.8, baseline_init=binit, seeds=15)
    print("transparency0.8 binit=%.1f alpha=%.3f acc=%.3f" % (binit, r["final_alpha_mean"], r["accuracy_mean"]))
    r2 = run_condition(f"REG q1 binit={binit}", "REG", lam=0.5, q=1.0, baseline_init=binit, seeds=15)
    print("REGq1 binit=%.1f alpha=%.3f acc=%.3f" % (binit, r2["final_alpha_mean"], r2["accuracy_mean"]))
