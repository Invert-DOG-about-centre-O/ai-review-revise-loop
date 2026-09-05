import json
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, spearmanr

with open("round1_review_2_results_raw.json") as f:
    results = json.load(f)

widths = sorted(set(r["width"] for r in results))
summary = {}
print(f"{'width':>6} {'test_acc':>14} {'ece':>16} {'nll':>14} {'T*':>12} {'ece_ts':>16} {'gap':>10}")
for w in widths:
    rows = [r for r in results if r["width"] == w]
    acc = np.array([r["test_acc"] for r in rows])
    ece = np.array([r["test_ece"] for r in rows])
    nll = np.array([r["test_nll"] for r in rows])
    T = np.array([r["temperature"] for r in rows])
    ece_ts = np.array([r["test_ece_ts"] for r in rows])
    gap = np.array([r["train_test_acc_gap"] for r in rows])
    summary[w] = dict(acc=acc, ece=ece, nll=nll, T=T, ece_ts=ece_ts, gap=gap)
    print(f"{w:>6} {acc.mean():.3f}+-{acc.std():.3f} {ece.mean():.3f}+-{ece.std():.3f} "
          f"{nll.mean():.3f}+-{nll.std():.3f} {T.mean():.2f}+-{T.std():.2f} "
          f"{ece_ts.mean():.3f}+-{ece_ts.std():.3f} {gap.mean():+.3f}")

print()
log_w = np.log2(np.array([r["width"] for r in results]))
ece_all = np.array([r["test_ece"] for r in results])
nll_all = np.array([r["test_nll"] for r in results])
rho_ece, p_ece = spearmanr(log_w, ece_all)
rho_nll, p_nll = spearmanr(log_w, nll_all)
print(f"Spearman(log2 width, ECE): rho={rho_ece:.3f}, p={p_ece:.2e}")
print(f"Spearman(log2 width, NLL): rho={rho_nll:.3f}, p={p_nll:.2e}")

print()
w4 = summary[4]["ece"]; w8 = summary[8]["ece"]; w16 = summary[16]["ece"]
plateau = np.concatenate([summary[w]["ece"] for w in [16,32,64,128,256,512]])
t, p = ttest_ind(np.concatenate([w4, w8]), plateau, equal_var=False)
print(f"peak(4+8) vs plateau(16-512): t={t:.3f} p={p:.3e}")

t2, p2 = ttest_ind(w4, w16, equal_var=False)
print(f"width4 vs width16: t={t2:.3f} p={p2:.3e}")

acc128 = summary[128]["acc"]; acc512 = summary[512]["acc"]
t3, p3 = ttest_ind(acc128, acc512, equal_var=False)
print(f"acc width128 vs width512: t={t3:.3f} p={p3:.3f}")

print()
w2rows = [r for r in results if r["width"] == 2]
collapsed = [r for r in w2rows if r["test_acc"] < 0.15]
noncollapsed = [r for r in w2rows if r["test_acc"] >= 0.15]
print("width2 collapsed n=", len(collapsed), "mean_ece=", np.mean([x["test_ece"] for x in collapsed]) if collapsed else None)
print("width2 noncollapsed n=", len(noncollapsed), "mean_ece=", np.mean([x["test_ece"] for x in noncollapsed]) if noncollapsed else None,
      "mean_acc=", np.mean([x["test_acc"] for x in noncollapsed]) if noncollapsed else None)
if noncollapsed:
    nc_ece = [x["test_ece"] for x in noncollapsed]
    t4, p4 = ttest_ind(nc_ece, w16, equal_var=False)
    print(f"noncollapsed width2 vs width16: t={t4:.3f} p={p4:.3e}")

print()
for w in widths:
    rows = [r for r in results if r["width"] == w]
    pre = np.array([r["test_ece"] for r in rows])
    post = np.array([r["test_ece_ts"] for r in rows])
    tp, pp = ttest_rel(pre, post)
    print(f"width={w:>4} ECE pre={pre.mean():.4f} post={post.mean():.4f} paired t={tp:.3f} p={pp:.4f}")
