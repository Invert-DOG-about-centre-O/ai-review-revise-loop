import json
import numpy as np

d = json.load(open("raw_results.json"))
r = d["results"]
n_increase = sum(1 for x in r if x["ece_ts"] > x["ece_raw"])
print("TS increases ECE:", n_increase, "/", len(r))
low = [x for x in r if x["acc_test"] < 0.4]
n_low_increase = sum(1 for x in low if x["ece_ts"] > x["ece_raw"])
print("low acc n=", len(low), "increase=", n_low_increase)

w2 = [x for x in r if x["dataset"] == "digits" and x["width"] == 2]
acc = np.array([x["acc_test"] for x in w2])
T = np.array([x["temperature"] for x in w2])
ece_ts = np.array([x["ece_ts"] for x in w2])
ece_raw = np.array([x["ece_raw"] for x in w2])
print("digits w2 n=", len(w2), "acc mean", acc.mean(), "T mean", T.mean(), "T sd", T.std(ddof=1))
print("ece_ts mean", ece_ts.mean(), "sd", ece_ts.std(ddof=1), "min", ece_ts.min(), "max", ece_ts.max())
print("ece_raw mean", ece_raw.mean(), "sd", ece_raw.std(ddof=1))

for w in [2, 16, 32, 64, 128, 256]:
    rows = [x for x in r if x["dataset"] == "digits" and x["condition"] == "baseline" and x["width"] == w]
    e = np.array([x["ece_raw"] for x in rows])
    a = np.array([x["acc_test"] for x in rows])
    print(w, "ece_raw mean", e.mean(), "acc mean", a.mean())
