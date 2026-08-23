from scipy import stats
from sim import run_condition

# Reviewers ask whether the REG-fair-beats-transparency reversal is an
# artifact of the paper's custom fixed-scale hill-climb, per Pan et al.
# (2022)'s finding that mitigation rankings can be optimizer-sensitive.
# Swap in a standard REINFORCE score-function update (update_rule="pg")
# for the same two headline conditions and re-check the ranking.

for name, cond, kw in [
    ("Approval-only", "APPROVAL_ONLY", {}),
    ("Transparency 0.8", "TRANSPARENCY", dict(transparency=0.8)),
    ("REG-fair lam=0.8,q=1.0", "REG", dict(lam=0.8, q=1.0)),
]:
    r = run_condition(name, cond, update_rule="pg", **kw)
    print("[pg] %s: alpha=%.3f+-%.3f acc=%.3f+-%.3f" % (
        name, r["final_alpha_mean"], r["final_alpha_std"],
        r["accuracy_mean"], r["accuracy_std"]))

tr = run_condition("tr", "TRANSPARENCY", transparency=0.8, update_rule="pg")
rf = run_condition("rf", "REG", lam=0.8, q=1.0, update_rule="pg")
t_a, p_a = stats.ttest_ind(rf["final_alpha_vals"], tr["final_alpha_vals"])
t_b, p_b = stats.ttest_ind(rf["accuracy_vals"], tr["accuracy_vals"])
print("\n[pg] REG-fair vs transparency: alpha t=%.3f p=%.4g | accuracy t=%.3f p=%.4g" % (
    t_a, p_a, t_b, p_b))
