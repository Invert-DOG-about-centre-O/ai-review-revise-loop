"""
Follow-up requested by round-3 review (question 1 / weakness 3): the audit+exploration-boost
"backfire" under slow exploration decay (beta=1e-5) was only established at n=10
(bonf p=1.9e-2). Re-run baseline and audit_explore at beta=1e-5 with n=25 (matching
the main comparison) to check whether the backfire survives at full sample size,
exactly analogous to the alpha_hi audit_enforce follow-up already in the paper.
"""
import time
import numpy as np
from scipy import stats as sstats
import collusion_sim as cs

PERIODS = 150000
SEEDS25 = list(range(25))

cs.ALPHA, cs.DELTA, cs.BETA = 0.15, 0.95, 1e-5

t0 = time.time()
base = np.array([cs.run_sim(seed=s, periods=PERIODS, condition="baseline")["collusion_index_tail"] for s in SEEDS25])
print("baseline done", time.time() - t0, "mean=", base.mean())
expl = np.array([cs.run_sim(seed=s, periods=PERIODS, condition="audit_explore")["collusion_index_tail"] for s in SEEDS25])
print("audit_explore done", time.time() - t0, "mean=", expl.mean())

t, p = sstats.ttest_rel(expl, base)
p_bonf = min(p * 4, 1.0)
w, pw = sstats.wilcoxon(expl, base)
print(f"beta=1e-5, n=25: baseline={base.mean():.4f} audit_explore={expl.mean():.4f} diff={expl.mean()-base.mean():+.4f} paired-t p={p:.2e} bonf={p_bonf:.2e} wilcoxon p={pw:.2e}")
print("total elapsed", time.time() - t0)
