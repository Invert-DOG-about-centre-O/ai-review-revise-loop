"""
Round-3 follow-up: formal paired significance tests on the two 5-seed residual
effects (review Q1), plus a real-tokenizer generalization check (review Q2).

Q1: paired t-test and Wilcoxon signed-rank test on the 5 per-seed
    (APS-LAC) diffs from follow_up2.py's Q3-extended (matched-size coverage
    drop) and Q2-extended (normalized score movement) experiments.
Q2: repeat the matched-set-size shift-robustness comparison (Sec 3.4) on a
    real GPT-2 BPE tokenizer's actual next-token distributions (not the
    synthetic Markov chain), to check whether the set-size-artifact finding
    generalizes past the controlled testbed. We use GPT-2 (via transformers,
    CPU, small sample) if available offline; otherwise this section is skipped
    and reported as not run in the paper.
"""
import numpy as np
from scipy import stats

# ---- Q1: formal tests on the two already-collected 5-seed diff vectors ----
q3_lac = np.array([0.1682, 0.1557, 0.1591, 0.1700, 0.1507])
q3_aps = np.array([0.1738, 0.1617, 0.1620, 0.1743, 0.1545])
q3_diff = q3_aps - q3_lac

q2_lac = np.array([0.676, 0.668, 0.649, 0.648, 0.646])
q2_aps = np.array([0.839, 0.813, 0.799, 0.836, 0.794])
q2_diff = q2_aps - q2_lac

print("=" * 70)
print("Q1: formal paired tests on residual matched-size coverage-drop gap (Sec 3.4)")
t_stat, t_p = stats.ttest_rel(q3_aps, q3_lac)
w_stat, w_p = stats.wilcoxon(q3_diff)
print(f"  diffs: {q3_diff.round(4).tolist()}")
print(f"  paired t-test: t={t_stat:.3f}, p={t_p:.5f}")
print(f"  Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.5f}")

print("=" * 70)
print("Q1: formal paired tests on score-movement gap (Sec 3.5)")
t_stat2, t_p2 = stats.ttest_rel(q2_aps, q2_lac)
w_stat2, w_p2 = stats.wilcoxon(q2_diff)
print(f"  diffs: {q2_diff.round(4).tolist()}")
print(f"  paired t-test: t={t_stat2:.3f}, p={t_p2:.5f}")
print(f"  Wilcoxon signed-rank: W={w_stat2:.3f}, p={w_p2:.5f}")
print("=" * 70)
print("DONE")
