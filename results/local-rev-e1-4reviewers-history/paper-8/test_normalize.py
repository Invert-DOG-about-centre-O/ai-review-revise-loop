import numpy as np
from sim import run_condition, P_USER, P_AI, N_ROUNDS

rng = np.random.default_rng(0)

# Empirically estimate the per-round reward *perturbation* (deviation from
# pure r_approve) that each mitigation injects, to put REG's lambda and
# transparency's discount factor on comparable "signal strength" units
# (reviewer-requested normalization), rather than treating lambda=0.8 and
# transparency=0.8 as commensurable because they share a numeric range.

def perturbation_stats(condition, n=200000, **kw):
    y = rng.random(n) < 0.5
    b_u = np.where(rng.random(n) < P_USER, y, ~y)
    e = np.where(rng.random(n) < P_AI, y, ~y)
    # use a fixed alpha=0.5 "probe" policy so both mitigations are measured
    # under the same action distribution
    use_user = rng.random(n) < 0.5
    o = np.where(use_user, b_u, e)
    r_approve = (o == b_u).astype(float)
    r_correct = (o == y).astype(float)
    if condition == "REG":
        lam = kw["lam"]
        reward = (1 - lam) * r_approve + lam * r_correct
    elif condition == "TRANSPARENCY":
        t = kw["transparency"]
        reward = np.where(o == b_u, (1 - t) * r_approve, r_approve)
    pert = reward - r_approve
    return pert.mean(), pert.std()

reg_mean, reg_std = perturbation_stats("REG", lam=0.8)
print("REG lam=0.8 perturbation: mean=%.4f std=%.4f" % (reg_mean, reg_std))

grid = np.round(np.arange(0.05, 1.001, 0.01), 3)
best_t, best_diff = None, 1e9
rows = []
for t in grid:
    m, s = perturbation_stats("TRANSPARENCY", transparency=float(t))
    rows.append((t, s))
    diff = abs(s - reg_std)
    if diff < best_diff:
        best_diff, best_t = diff, t

print("Best-matching transparency by perturbation std: t=%.3f (std=%.4f vs REG std=%.4f)" % (
    best_t, dict(rows)[best_t], reg_std))

# Rerun the headline comparison at the *matched* transparency value.
regfair = run_condition("regfair0.8", "REG", lam=0.8, q=1.0)
tr_matched = run_condition("tr_matched", "TRANSPARENCY", transparency=float(best_t))
tr_08 = run_condition("tr0.8", "TRANSPARENCY", transparency=0.8)

from scipy import stats as st
t_a, p_a = st.ttest_ind(regfair["final_alpha_vals"], tr_matched["final_alpha_vals"])
t_b, p_b = st.ttest_ind(regfair["accuracy_vals"], tr_matched["accuracy_vals"])
print("\nMatched-strength transparency=%.3f: alpha=%.3f acc=%.3f" % (
    best_t, tr_matched["final_alpha_mean"], tr_matched["accuracy_mean"]))
print("REG-fair lam=0.8,q=1.0: alpha=%.3f acc=%.3f" % (
    regfair["final_alpha_mean"], regfair["accuracy_mean"]))
print("REG-fair vs matched-transparency: alpha t=%.3f p=%.4g | accuracy t=%.3f p=%.4g" % (
    t_a, p_a, t_b, p_b))
print("(for reference, transparency=0.8 unmatched: alpha=%.3f acc=%.3f)" % (
    tr_08["final_alpha_mean"], tr_08["accuracy_mean"]))
