import json
import numpy as np
d = json.load(open("multiseed_results.json"))
res = [r for r in d["results"] if r["smoothing"]==0.1]
for h in [2,4,8,16]:
    rows = [r for r in res if r["hidden"]==h]
    n_under = sum(1 for r in rows if r["direction"]=="underconfident")
    mean_pre = np.mean([r["test_ece_pre"] for r in rows])
    mean_post = np.mean([r["test_ece_post"] for r in rows])
    print(h, n_under, mean_pre, mean_post)
res0 = [r for r in d["results"] if r["smoothing"]==0.0]
for h in [2,4,8,16]:
    rows = [r for r in res0 if r["hidden"]==h]
    print("unsmoothed", h, np.mean([r["test_ece_pre"] for r in rows]))
