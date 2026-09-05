"""
Reviewer question: does the digits/baseline inverse-U (raw ECE rises then
falls as width increases) hold within individual seeds, or is it an
artifact of averaging seeds whose accuracy-ECE trajectories differ in phase?
Checks, per seed, whether ece_raw(width=16) > ece_raw(width=2) and
ece_raw(width=16) > ece_raw(width=256) -- the two inequalities that define
the inverse-U at its reported peak.
"""
import json
import numpy as np

with open("raw_results.json") as f:
    data = json.load(f)
results = data["results"]

rows = [r for r in results if r["dataset"] == "digits" and r["condition"] == "baseline"]
by_seed = {}
for r in rows:
    by_seed.setdefault(r["seed_idx"], {})[r["width"]] = r["ece_raw"]

n_seeds = len(by_seed)
rises_then_falls = 0
for seed_idx, w2e in by_seed.items():
    if w2e[16] > w2e[2] and w2e[16] > w2e[256]:
        rises_then_falls += 1

print(f"seeds where ece_raw(w16) > ece_raw(w2) AND ece_raw(w16) > ece_raw(w256): "
      f"{rises_then_falls}/{n_seeds}")

with open("per_seed_inverse_u_results.json", "w") as f:
    json.dump({"n_seeds": n_seeds, "seeds_matching_inverse_u_pattern": rises_then_falls}, f, indent=1)
