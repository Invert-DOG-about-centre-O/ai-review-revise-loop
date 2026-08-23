import json
d = json.load(open("results.json"))
rc = d["risk_coverage"]["logistic_combo"]
for c, r in zip(rc["coverage"], rc["risk"]):
    print(c, 1 - r)
