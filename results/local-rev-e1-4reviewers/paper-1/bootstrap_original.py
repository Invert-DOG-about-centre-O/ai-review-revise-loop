import json, random
from multiseed_experiment import auroc, bootstrap_ci

with open("raw_results.json") as f:
    records = json.load(f)

ci = bootstrap_ci(records, n_boot=5000, seed=0)
print(json.dumps(ci, indent=2))
with open("bootstrap_original_results.json", "w") as f:
    json.dump(ci, f, indent=2)
