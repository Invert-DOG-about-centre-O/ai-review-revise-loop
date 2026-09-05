import json
from scipy.stats import spearmanr
import numpy as np

data = json.load(open("raw_results.json"))["results"]
syn = [r for r in data if r["dataset"] == "synthetic"]

logw = [np.log2(r["width"]) for r in syn]
ece_pre = [r["ece_pre"] for r in syn]
rho, p = spearmanr(logw, ece_pre)
print("full synthetic H1 rho", rho, p, "n=", len(syn))

syn_conv = [r for r in syn if r["converged"]]
logw2 = [np.log2(r["width"]) for r in syn_conv]
ece2 = [r["ece_pre"] for r in syn_conv]
rho2, p2 = spearmanr(logw2, ece2)
print("excl non-converged H1 rho", rho2, p2, "n=", len(syn_conv))

w16 = [r for r in syn if r["width"] == 16]
conv16 = [r["ece_pre"] for r in w16 if r["converged"]]
nonconv16 = [r["ece_pre"] for r in w16 if not r["converged"]]
print("width16 converged ece_pre mean", np.mean(conv16), "n=", len(conv16))
print("width16 nonconverged ece_pre mean", np.mean(nonconv16) if nonconv16 else None, "n=", len(nonconv16))
