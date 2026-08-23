import json, numpy as np
from scipy import stats
d = json.load(open("results_robust.json"))
def deltas(regime, split):
    runs = d[regime]["runs"]
    g = np.array([r[split]["glob_ece"] for r in runs])
    a = np.array([r[split]["adap_ece"] for r in runs])
    return a - g  # adap - glob; negative = adap better
for regime, label in [("regime_A_overconfident","A(modest)"), ("regime_B_higher_skill","B(sharp)")]:
    for split in ("in_domain","shift"):
        x = deltas(regime, split)
        t, p = stats.ttest_1samp(x, 0.0)
        npos = int((x > 0).sum()); nneg = int((x < 0).sum())
        sign_p = stats.binomtest(min(npos, nneg), n=len(x), p=0.5).pvalue
        try:
            w, wp = stats.wilcoxon(x)
        except Exception:
            wp = float('nan')
        print(f"{label:10s} {split:9s} mean={x.mean():+.5f} sd={x.std(ddof=1):.5f} t={t:+.2f} p_t={p:.4f} signs(-/+)={nneg}/{npos} p_sign={sign_p:.4f} p_wilcox={wp:.4f}")
