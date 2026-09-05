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
# Spearman correlation of log-width vs ECE across all runs
log_w = np.log2(np.array([r["width"] for r in results]))
ece_all = np.array([r["test_ece"] for r in results])
nll_all = np.array([r["test_nll"] for r in results])
T_all = np.array([r["temperature"] for r in results])
conf_all = np.array([r["avg_confidence"] for r in results])
rho_ece, p_ece = spearmanr(log_w, ece_all)
rho_nll, p_nll = spearmanr(log_w, nll_all)
rho_T, p_T = spearmanr(log_w, T_all)
rho_conf, p_conf = spearmanr(log_w, conf_all)
print(f"Spearman(log2 width, ECE): rho={rho_ece:.3f}, p={p_ece:.2e}")
print(f"Spearman(log2 width, NLL): rho={rho_nll:.3f}, p={p_nll:.2e}")
print(f"Spearman(log2 width, T*):  rho={rho_T:.3f}, p={p_T:.2e}")
print(f"Spearman(log2 width, avg_confidence): rho={rho_conf:.3f}, p={p_conf:.2e}")

print()
# smallest vs largest width group t-test on ECE
small = summary[widths[0]]["ece"]
large = summary[widths[-1]]["ece"]
t, p = ttest_ind(small, large, equal_var=False)
print(f"t-test ECE width={widths[0]} vs width={widths[-1]}: t={t:.3f}, p={p:.4f}")

# best-calibrated width (min mean ECE) vs largest width
mean_ece_by_w = {w: summary[w]["ece"].mean() for w in widths}
best_w = min(mean_ece_by_w, key=mean_ece_by_w.get)
worst_w = max(mean_ece_by_w, key=mean_ece_by_w.get)
print(f"Best-calibrated width (min mean ECE): {best_w} (ECE={mean_ece_by_w[best_w]:.4f})")
print(f"Worst-calibrated width (max mean ECE): {worst_w} (ECE={mean_ece_by_w[worst_w]:.4f})")

t2, p2 = ttest_ind(summary[best_w]["ece"], summary[worst_w]["ece"], equal_var=False)
print(f"t-test ECE width={best_w} vs width={worst_w}: t={t2:.3f}, p={p2:.4f}")

# does temp scaling reduce ECE significantly at each width? paired t-test
print()
for w in widths:
    rows = [r for r in results if r["width"] == w]
    pre = np.array([r["test_ece"] for r in rows])
    post = np.array([r["test_ece_ts"] for r in rows])
    t3, p3 = ttest_ind(pre, post, equal_var=False)  # not paired since different rand draws but same seeds -> use paired
    from scipy.stats import ttest_rel
    t3p, p3p = ttest_rel(pre, post)
    print(f"width={w:>4}: ECE pre={pre.mean():.4f} post={post.mean():.4f} paired t={t3p:.3f} p={p3p:.4f}")

# accuracy plateau check: is width=512 acc significantly different from width=128?
acc128 = summary[128]["acc"]
acc512 = summary[512]["acc"]
t4, p4 = ttest_ind(acc128, acc512, equal_var=False)
print()
print(f"t-test accuracy width=128 vs width=512: t={t4:.3f}, p={p4:.4f}")

with open("analysis_summary.json", "w") as f:
    json.dump({str(w): {k: v.tolist() for k, v in summary[w].items()} for w in widths}, f, indent=2)
