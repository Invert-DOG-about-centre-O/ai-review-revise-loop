import json
import numpy as np

d = json.load(open("raw_results.json"))["results"]
arr = [
    (r["dataset"], r["width"], r["acc_test"], r["ece_raw"], r["ece_ts"], r["temperature"])
    for r in d
]

worse = 0
total = 0
low_acc_worse = 0
low_acc_total = 0
for ds, w, acc, er, et, T in arr:
    total += 1
    if et > er:
        worse += 1
    if acc < 0.4:
        low_acc_total += 1
        if et > er:
            low_acc_worse += 1

print("overall frac ts worsens ece:", worse / total, total)
print("low-acc(<0.4) frac ts worsens ece:", low_acc_worse / max(low_acc_total, 1), low_acc_total)

sub = [T for ds, w, acc, er, et, T in arr if ds == "digits" and w == 2]
print("digits w=2 T mean/std:", np.mean(sub), np.std(sub))
sub2 = [et for ds, w, acc, er, et, T in arr if ds == "digits" and w == 2]
print("digits w=2 ece_ts mean/std:", np.mean(sub2), np.std(sub2), "min/max", min(sub2), max(sub2))
sub3 = [er for ds, w, acc, er, et, T in arr if ds == "digits" and w == 2]
print("digits w=2 ece_raw mean/std:", np.mean(sub3), np.std(sub3))

out = {
    "overall_frac_ts_worsens_ece": worse / total,
    "overall_n": total,
    "low_acc_lt_0.4_frac_ts_worsens_ece": low_acc_worse / max(low_acc_total, 1),
    "low_acc_n": low_acc_total,
    "digits_width2_baseline_and_ls_pooled": {
        "temperature_mean": float(np.mean(sub)),
        "temperature_std": float(np.std(sub)),
        "ece_ts_mean": float(np.mean(sub2)),
        "ece_ts_std": float(np.std(sub2)),
        "ece_ts_min": float(min(sub2)),
        "ece_ts_max": float(max(sub2)),
        "ece_raw_mean": float(np.mean(sub3)),
        "ece_raw_std": float(np.std(sub3)),
    },
}
with open("extra_check_results.json", "w") as f:
    json.dump(out, f, indent=1)
