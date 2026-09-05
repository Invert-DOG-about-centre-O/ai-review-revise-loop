import json, numpy as np
from scipy import stats
import collections

data = json.load(open("raw_results_extgrid.json"))
results = data["results"]
by_ds = collections.defaultdict(list)
for r in results:
    by_ds[r["dataset"]].append(r)
for ds, rows in by_ds.items():
    pre = np.array([r["ece_pre"] for r in rows])
    post = np.array([r["ece_post"] for r in rows])
    reduction = pre - post
    r_, p_ = stats.pearsonr(pre, reduction)
    print(ds, "n=", len(rows), "r=", r_, "p=", p_)
