import json
import experiment as E

raw = json.load(open("results_raw.json"))
targets = []
for ds in ["digits", "synthetic"]:
    for w in [4, 256]:
        targets.append((ds, w, 0))
        targets.append((ds, w, 5))

for ds, w, si in targets:
    r = E.run_one(ds, w, si)
    orig = [x for x in raw if x["dataset"] == ds and x["width"] == w and x["seed_idx"] == si][0]
    print(ds, w, si, "T* recompute=%.6f orig=%.6f match=%s" % (
        r["T_star"], orig["T_star"], abs(r["T_star"] - orig["T_star"]) < 1e-6),
        "pre_ece_match=", abs(r["pre_ece"] - orig["pre_ece"]) < 1e-9,
        "test_acc_match=", abs(r["test_acc"] - orig["test_acc"]) < 1e-9)
