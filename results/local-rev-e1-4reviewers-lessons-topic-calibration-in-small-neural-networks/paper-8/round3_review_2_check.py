import json
import numpy as np
from scipy import stats

with open("results_v2.json") as f:
    v2 = json.load(f)
with open("results_v3.json") as f:
    v3 = json.load(f)

main_res = v2["main"]["results"]
widths_main = [2, 4, 16, 64, 256]
T_by_width = {w: [r["T_star"] for r in main_res if r["width"] == w and r["label_smoothing"] == 0.0] for w in widths_main}

res_dense = v3["dense"]["results"]
all_T = dict(T_by_width)
all_T[8] = [r["T_star"] for r in res_dense if r["width"] == 8]
all_T[32] = [r["T_star"] for r in res_dense if r["width"] == 32]

print("=== Table 1 check: T* mean +/- SD, t-CI half width ===")
ordered = [2, 4, 8, 16, 32, 64, 256]
for w in ordered:
    vals = all_T[w]
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1)
    half = stats.t.ppf(0.975, df=len(vals) - 1) * sd / np.sqrt(len(vals))
    if w in (8, 32):
        rows = [r for r in res_dense if r["width"] == w]
    else:
        rows = [r for r in main_res if r["width"] == w and r["label_smoothing"] == 0.0]
    acc = np.mean([r["acc"] for r in rows])
    ece = np.mean([r["ece"] for r in rows])
    bece = np.mean([r["bayes_ece"] for r in rows])
    ece_ts = np.mean([r["ece_after_ts"] for r in rows])
    print("w=%4d n=%2d acc=%.3f ece=%.3f bayes_ece=%.3f T*=%.2f+/-%.2f half=%.2f ece_ts=%.3f" % (w, len(vals), acc, ece, bece, mean, sd, half, ece_ts))

print("\n=== Welch t-tests adjacent widths ===")
for a, b in zip(ordered[:-1], ordered[1:]):
    t, p = stats.ttest_ind(all_T[a], all_T[b], equal_var=False)
    print("  %d vs %d: t=%.2f p=%.2e" % (a, b, t, p))

print("\n=== Convergence check width2 80 vs 400 epochs ===")
vals80 = T_by_width[2]
w2long = v3["w2_long"]["results"]
vals400 = [r["T_star"] for r in w2long if r["width"] == 2]
print("80ep:", np.mean(vals80), np.std(vals80, ddof=1), len(vals80))
print("400ep:", np.mean(vals400), np.std(vals400, ddof=1), len(vals400))
t, p = stats.ttest_ind(vals80, vals400, equal_var=False)
print("t-test:", t, p)

print("\n=== Label smoothing table (Table 2) ===")
ls_res = [r for r in main_res if r["label_smoothing"] == 0.1]
noLS_res = [r for r in main_res if r["label_smoothing"] == 0.0]
for w in widths_main:
    t_nols = np.mean([r["T_star"] for r in noLS_res if r["width"] == w])
    t_ls = np.mean([r["T_star"] for r in ls_res if r["width"] == w])
    e_nols = np.mean([r["ece"] for r in noLS_res if r["width"] == w])
    e_ls = np.mean([r["ece"] for r in ls_res if r["width"] == w])
    print("w=%4d T*(noLS)=%.2f T*(LS)=%.2f ECE(noLS)=%.4f ECE(LS)=%.4f fold=%.1fx" % (w, t_nols, t_ls, e_nols, e_ls, e_ls / e_nols))

print("\n=== Second instance (robustness) ===")
rob = v2["robustness"]["results"]
for w in widths_main:
    vals = [r["T_star"] for r in rob if r["width"] == w]
    acc = np.mean([r["acc"] for r in rob if r["width"] == w])
    print("w=%4d T*=%.2f acc=%.3f n=%d" % (w, np.mean(vals), acc, len(vals)))
