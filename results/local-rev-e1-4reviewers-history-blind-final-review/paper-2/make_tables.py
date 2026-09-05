import json
import numpy as np

with open("results_v2.json") as f:
    r = json.load(f)

def row(acc, syco=None):
    if syco is None:
        return f"{np.mean(acc):.3f} ± {np.std(acc):.3f}"
    return f"{np.mean(acc):.3f} ± {np.std(acc):.3f} | {np.mean(syco):.3f} ± {np.std(syco):.3f}"

print("Table 3.1 (alpha sweep)")
for a, d in r["alpha_sweep"].items():
    print(f"| {float(a):.2f} | {row(d['acc'], d['syco'])} |")
print(f"Oracle: {row([x[0] for x in r['oracle']])}")

print("\nTable 3.2 (rater aggregation)")
for n, d in r["rater_sweep"].items():
    print(f"| {n} | {row(d['acc'], d['syco'])} |")

print("\nTable 3.3 (correlated rater error)")
for rho, d in r["correlated_sweep"].items():
    print(f"| {float(rho):.2f} | {row(d['acc'], d['syco'])} |")

print("\nSignificance tests")
for k, v in r["significance"].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
