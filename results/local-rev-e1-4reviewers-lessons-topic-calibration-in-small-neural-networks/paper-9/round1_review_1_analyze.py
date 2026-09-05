import json
import numpy as np
from scipy.stats import ttest_ind, spearmanr

with open("results_raw.json") as f:
    results = json.load(f)

widths = sorted(set(r["width"] for r in results))

print(f"{'width':>6} {'test_acc':>10} {'ece':>18} {'nll':>10} {'T*':>14} {'ece_ts':>18} {'gap':>10}")
summary = {}
for w in widths:
    rows = [r for r in results if r["width"] == w]
    acc = np.array([r["test_acc"] for r in rows])
    ece = np.array([r["test_ece"] for r in rows])
    nll = np.array([r["test_nll"] for r in rows])
    T = np.array([r["temperature"] for r in rows])
    ece_ts = np.array([r["test_ece_ts"] for r in rows])
    gap = np.array([r["train_test_acc_gap"] for r in rows])
    conf = np.array([r["avg_confidence"] for r in rows])
    summary[w] = dict(acc=acc, ece=ece, nll=nll, T=T, ece_ts=ece_ts, gap=gap, conf=conf)
    print(f"{w:>6} {acc.mean():>6.3f}±{acc.std():.3f} {ece.mean():>7.3f}±{ece.std():.3f} "
          f"{nll.mean():>6.3f}±{nll.std():.3f} {T.mean():>6.2f}±{T.std():.2f} "
          f"{ece_ts.mean():>7.3f}±{ece_ts.std():.3f} {gap.mean():>+7.3f}")

print()
log_w = np.log2(np.array([r["width"] for r in results]))
ece_all = np.array([r["test_ece"] for r in results])
nll_all = np.array([r["test_nll"] for r in results])
rho_ece, p_ece = spearmanr(log_w, ece_all)
rho_nll, p_nll = spearmanr(log_w, nll_all)
print(f"Spearman(log2 width, ECE): rho={rho_ece:.3f}, p={p_ece:.2e}")
print(f"Spearman(log2 width, NLL): rho={rho_nll:.3f}, p={p_nll:.2e}")

print()
# pooled 4-8 vs plateau 16-512
peak = np.concatenate([summary[4]["ece"], summary[8]["ece"]])
plateau = np.concatenate([summary[w]["ece"] for w in [16,32,64,128,256,512]])
t,p = ttest_ind(peak, plateau, equal_var=False)
print(f"peak(4,8) n={len(peak)} mean={peak.mean():.4f} vs plateau(16-512) n={len(plateau)} mean={plateau.mean():.4f}: t={t:.3f} p={p:.3e}")

t2,p2 = ttest_ind(summary[4]["ece"], summary[16]["ece"], equal_var=False)
print(f"width4 vs width16: t={t2:.3f} p={p2:.3e}")

t4,p4 = ttest_ind(summary[128]["acc"], summary[512]["acc"], equal_var=False)
print(f"acc width128 vs width512: t={t4:.3f} p={p4:.3f}")

from scipy.stats import ttest_rel
print()
for w in widths:
    pre = summary[w]["ece"]; post = summary[w]["ece_ts"]
    tp, pp = ttest_rel(pre, post)
    print(f"width={w:>4} ECE pre={pre.mean():.4f} post={post.mean():.4f} paired t={tp:.3f} p={pp:.4f}")
