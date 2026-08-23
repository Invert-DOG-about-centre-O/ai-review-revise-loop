import numpy as np
from scipy import stats
from sim import run_condition

# Significance tests
lam01 = run_condition("lam0.1", "REG", lam=0.1, q=0.3)
lam08 = run_condition("lam0.8", "REG", lam=0.8, q=0.3)
t, p = stats.ttest_ind(lam01["final_alpha_vals"], lam08["final_alpha_vals"])
print("lambda 0.1 vs 0.8 (q=0.3) alpha t-test: t=%.3f p=%.4g" % (t, p))

q05 = run_condition("q0.5", "REG", lam=0.5, q=0.5)
q10 = run_condition("q1.0", "REG", lam=0.5, q=1.0)
t2, p2 = stats.ttest_ind(q05["final_alpha_vals"], q10["final_alpha_vals"])
print("q 0.5 vs 1.0 (lam=0.5) alpha t-test: t=%.3f p=%.4g" % (t2, p2))

tr05 = run_condition("tr0.5", "TRANSPARENCY", transparency=0.5)
tr08 = run_condition("tr0.8", "TRANSPARENCY", transparency=0.8)
t3, p3 = stats.ttest_ind(tr05["final_alpha_vals"], tr08["final_alpha_vals"])
print("transparency 0.5 vs 0.8 alpha t-test: t=%.3f p=%.4g" % (t3, p3))

t4, p4 = stats.ttest_ind(tr08["accuracy_vals"], q10["accuracy_vals"])
print("accuracy: transparency0.8 vs REG(lam0.5,q1.0) t-test: t=%.3f p=%.4g" % (t4, p4))

# REG-fair (q=1.0) headline point: report mean+-std (30 seeds) and a t-test
# against transparency=0.8, since reviewers asked for the same statistical
# weight given to this comparison as the other headline ones.
regfair08 = run_condition("regfair0.8", "REG", lam=0.8, q=1.0)
regfair10 = run_condition("regfair1.0", "REG", lam=1.0, q=1.0)
t5, p5 = stats.ttest_ind(regfair08["final_alpha_vals"], tr08["final_alpha_vals"])
t6, p6 = stats.ttest_ind(regfair08["accuracy_vals"], tr08["accuracy_vals"])
print("REG-fair lam=0.8,q=1.0: alpha=%.3f+-%.3f acc=%.3f+-%.3f (n=30)"
      % (regfair08["final_alpha_mean"], regfair08["final_alpha_std"],
         regfair08["accuracy_mean"], regfair08["accuracy_std"]))
print("REG-fair lam=1.0,q=1.0: alpha=%.3f+-%.3f acc=%.3f+-%.3f (n=30)"
      % (regfair10["final_alpha_mean"], regfair10["final_alpha_std"],
         regfair10["accuracy_mean"], regfair10["accuracy_std"]))
print("REG-fair(lam0.8) vs transparency0.8: alpha t=%.3f p=%.4g | accuracy t=%.3f p=%.4g" % (t5, p5, t6, p6))

# Bonferroni correction across the 6 headline t-tests above (alpha_fwer=0.05)
pvals = [p, p2, p3, p4, p5, p6]
alpha_fwer = 0.05
bonf_thresh = alpha_fwer / len(pvals)
print("\nBonferroni: %d headline tests, per-test threshold=%.4g; all significant at corrected level: %s"
      % (len(pvals), bonf_thresh, all(pv < bonf_thresh for pv in pvals)))

# Hyperparameter sensitivity: LR and baseline_beta grid on the two headline conditions
print("\n--- hyperparam sensitivity ---")
for lr in [0.02, 0.05, 0.1]:
    for bb in [0.90, 0.95, 0.99]:
        rtr = run_condition("tr", "TRANSPARENCY", transparency=0.8, lr=lr, baseline_beta=bb, seeds=15)
        rreg = run_condition("reg", "REG", lam=0.5, q=1.0, lr=lr, baseline_beta=bb, seeds=15)
        rappr = run_condition("appr", "APPROVAL_ONLY", lr=lr, baseline_beta=bb, seeds=15)
        print("lr=%.2f bb=%.2f | approval acc=%.3f | transparency0.8 acc=%.3f alpha=%.3f | REG(q1) acc=%.3f alpha=%.3f"
              % (lr, bb, rappr["accuracy_mean"], rtr["accuracy_mean"], rtr["final_alpha_mean"],
                 rreg["accuracy_mean"], rreg["final_alpha_mean"]))
