import json
import numpy as np
from scipy.stats import ttest_ind

with open("results_raw.json") as f:
    r = json.load(f)

w2 = [row for row in r if row["width"] == 2]
collapsed = [row for row in w2 if row["test_acc"] < 0.15]
noncollapsed = [row for row in w2 if row["test_acc"] >= 0.15]
print("collapsed n=", len(collapsed), "mean ece=", np.mean([x["test_ece"] for x in collapsed]))
print("noncollapsed n=", len(noncollapsed),
      "mean ece=", np.mean([x["test_ece"] for x in noncollapsed]),
      "mean acc=", np.mean([x["test_acc"] for x in noncollapsed]))

w4 = [row["test_ece"] for row in r if row["width"] == 4]
w16 = [row["test_ece"] for row in r if row["width"] == 16]
nc_ece = [x["test_ece"] for x in noncollapsed]
t, p = ttest_ind(nc_ece, w16, equal_var=False)
print("noncollapsed width2 vs width16 ECE t-test:", t, p)
t2, p2 = ttest_ind(w4, w16, equal_var=False)
print("width4 vs width16 ECE t-test:", t2, p2)

plateau = []
for w in [16, 32, 64, 128, 256, 512]:
    plateau += [row["test_ece"] for row in r if row["width"] == w]
w8 = [row["test_ece"] for row in r if row["width"] == 8]
t3, p3 = ttest_ind(w4 + w8, plateau, equal_var=False)
print("peak(width4+8) vs plateau(16-512) ECE t-test:", t3, p3,
      "n_peak=", len(w4 + w8), "n_plateau=", len(plateau))
