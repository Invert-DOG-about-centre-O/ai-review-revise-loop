"""
Follow-up requested by round-2 review (question 3): the audit+price-cap result
at high learning rate (alpha=0.30) was only marginal at n=10 (bonf p=0.061).
Re-run baseline and audit_enforce at alpha=0.30 with n=25 (matching the main
comparison) to check whether this is a true attenuation or an n=10 power artifact.
"""
import time
import numpy as np
from scipy import stats as sstats
import collusion_sim as cs

PERIODS = 150000
SEEDS25 = list(range(25))

cs.ALPHA, cs.DELTA, cs.BETA = 0.30, 0.95, 4e-5

t0 = time.time()
base = np.array([cs.run_sim(seed=s, periods=PERIODS, condition="baseline")["collusion_index_tail"] for s in SEEDS25])
print("baseline done", time.time() - t0, "mean=", base.mean())
enf = np.array([cs.run_sim(seed=s, periods=PERIODS, condition="audit_enforce")["collusion_index_tail"] for s in SEEDS25])
print("audit_enforce done", time.time() - t0, "mean=", enf.mean())

t, p = sstats.ttest_rel(enf, base)
p_bonf = min(p * 4, 1.0)
print(f"alpha=0.30, n=25: baseline={base.mean():.4f} audit_enforce={enf.mean():.4f} diff={enf.mean()-base.mean():+.4f} p={p:.2e} bonf={p_bonf:.2e}")
print("total elapsed", time.time() - t0)
