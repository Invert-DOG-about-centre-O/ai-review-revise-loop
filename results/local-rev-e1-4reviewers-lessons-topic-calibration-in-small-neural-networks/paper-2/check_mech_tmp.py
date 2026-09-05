import json
d = json.load(open("results.json"))
results = d["raw_results"]
cw = d["crossover_widths"]["digits"]
sat = {}
for seed_str, w in cw.items():
    seed = int(seed_str)
    rows = sorted([r for r in results if r["dataset"]=="digits" and r["seed"]==seed], key=lambda r:r["width"])
    sat_w = None
    for r in rows:
        if r.get("train_acc",0) >= 0.99:
            sat_w = r["width"]
            break
    sat[seed] = sat_w
    print(seed, "crossover", w, "sat", sat_w)
both = {s:(cw[str(s)],sat[s]) for s in sat if cw[str(s)] is not None and sat[s] is not None}
coincide = sum(1 for s,(c,sa) in both.items() if c==sa)
print("n both defined", len(both), "coincide", coincide)
