import json
import numpy as np
from scipy import stats as sstats

with open("results.json") as f:
    results = json.load(f)

meta = results["meta"]
print("Nash price:", meta["P_NASH"], "Monopoly price:", meta["P_MONOPOLY"])

def by_condition(records, key="collusion_index_tail"):
    out = {}
    for r in records:
        out.setdefault(r["condition"], {})[r["seed"]] = r[key]
    return out

main = by_condition(results["main"])
conds = ["baseline", "diversity", "transparency", "audit_explore", "audit_enforce"]
seeds = sorted(main["baseline"].keys())

print("\n=== Main comparison (collusion index, last 10% of periods) ===")
summary = {}
for c in conds:
    vals = np.array([main[c][s] for s in seeds])
    summary[c] = vals
    print(f"{c:15s} mean={vals.mean():.4f} sd={vals.std(ddof=1):.4f} n={len(vals)}")

print("\n=== Paired tests vs baseline (Bonferroni-corrected, k=4 comparisons) ===")
k = 4
base = summary["baseline"]
for c in ["diversity", "transparency", "audit_explore", "audit_enforce"]:
    t, p = sstats.ttest_rel(summary[c], base)
    w, pw = sstats.wilcoxon(summary[c], base)
    diff = summary[c].mean() - base.mean()
    p_corr = min(p * k, 1.0)
    print(f"{c:15s} mean_diff={diff:+.4f}  paired-t p={p:.2e} (bonf={p_corr:.2e})  wilcoxon p={pw:.2e}")

print("\n=== Threshold ablation (audit_enforce), matched n=25 seeds ===")
thr = {}
for r in results["threshold_ablation"]:
    thr.setdefault(r["audit_theta"], {})[r["seed"]] = r["collusion_index_tail"]
thr[0.5] = {s: main["audit_enforce"][s] for s in seeds}  # main run used theta=0.5
for theta in sorted(thr.keys()):
    vals = np.array([thr[theta][s] for s in seeds])
    n_audits = [r["n_audits"] for r in results["threshold_ablation"] if r.get("audit_theta") == theta] or \
               [r["n_audits"] for r in results["main"] if r["condition"] == "audit_enforce"]
    print(f"theta={theta}: mean_idx={vals.mean():.4f} sd={vals.std(ddof=1):.4f} mean_n_audits={np.mean(n_audits):.1f}")

print("\n=== Market robustness (MU=0.5, less differentiated), n=15 seeds ===")
mr = by_condition(results["market_robustness"])
rseeds = sorted(mr["baseline"].keys())
rbase = np.array([mr["baseline"][s] for s in rseeds])
print(f"baseline mean={rbase.mean():.4f} sd={rbase.std(ddof=1):.4f}")
kr = 4
for c in ["diversity", "transparency", "audit_explore", "audit_enforce"]:
    vals = np.array([mr[c][s] for s in rseeds])
    t, p = sstats.ttest_rel(vals, rbase)
    p_corr = min(p * kr, 1.0)
    print(f"{c:15s} mean={vals.mean():.4f}  mean_diff={vals.mean()-rbase.mean():+.4f}  paired-t p={p:.2e} (bonf={p_corr:.2e})")

# effect sizes (Cohen's d for paired samples)
print("\n=== Effect sizes (Cohen's dz, paired) vs baseline, main market ===")
for c in ["diversity", "transparency", "audit_explore", "audit_enforce"]:
    diffs = summary[c] - base
    dz = diffs.mean() / diffs.std(ddof=1)
    print(f"{c:15s} dz={dz:+.3f}")
