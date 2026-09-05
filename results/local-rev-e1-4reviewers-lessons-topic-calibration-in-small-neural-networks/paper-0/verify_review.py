import json

with open("multiseed_results.json") as f:
    data = json.load(f)

results = data["results"]

WIDTHS = [2,4,8,16,32,64,128,256,512]
crossovers = {}
for seed in range(10):
    rows = [r for r in results if r["seed"]==seed and r["smoothing"]==0.0]
    rows.sort(key=lambda r: WIDTHS.index(r["hidden"]))
    first_over = None
    for r in rows:
        if r["direction"] == "overconfident":
            first_over = r["hidden"]
            break
    crossovers[seed] = first_over

print("crossover widths:", crossovers)
print("mode 8 count:", sum(1 for v in crossovers.values() if v==8))

row0 = [r for r in results if r["seed"]==0 and r["smoothing"]==0.0]
row0.sort(key=lambda r: WIDTHS.index(r["hidden"]))
print("\nTable1 seed0:")
for r in row0:
    print(r["hidden"], round(r["test_acc"],3), round(r["test_conf_pre"],3), round(r["test_ece_pre"],4), round(r["fitted_T"],3), round(r["test_ece_post"],4), r["direction"])

print("\nTable3 (smoothing=0.1):")
for w in [2,4,8,16]:
    rows = [r for r in results if r["hidden"]==w and r["smoothing"]==0.1]
    frac_under = sum(1 for r in rows if r["direction"]=="underconfident")
    mean_ece_pre = sum(r["test_ece_pre"] for r in rows)/len(rows)
    mean_ece_post = sum(r["test_ece_post"] for r in rows)/len(rows)
    rows0 = [r for r in results if r["hidden"]==w and r["smoothing"]==0.0]
    mean_ece_pre0 = sum(r["test_ece_pre"] for r in rows0)/len(rows0)
    ratio = mean_ece_pre/mean_ece_pre0
    print(w, frac_under, "/10", round(mean_ece_pre,3), round(ratio,2), round(mean_ece_post,3))

rows2 = [r for r in results if r["hidden"]==2 and r["smoothing"]==0.1]
floor_count = sum(1 for r in rows2 if abs(r["fitted_T"]-1e-3) < 1e-4)
print("\nhidden=2 smoothed unconstrained T values:", [round(r["fitted_T"],4) for r in rows2])
print("floor count (T==1e-3):", floor_count)
mean_pre2 = sum(r["test_ece_pre"] for r in rows2)/len(rows2)
mean_post2 = sum(r["test_ece_post"] for r in rows2)/len(rows2)
print("mean ece pre/post hidden=2 smoothed:", round(mean_pre2,3), round(mean_post2,3))
