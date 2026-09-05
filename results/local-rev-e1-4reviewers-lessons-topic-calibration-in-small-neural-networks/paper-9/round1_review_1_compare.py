import json
import numpy as np

with open("results_raw.json") as f:
    orig = json.load(f)
with open("round1_review_1_results_raw.json") as f:
    fresh = json.load(f)

widths = sorted(set(r["width"] for r in orig))
for w in widths:
    o = [r for r in orig if r["width"]==w]
    n = [r for r in fresh if r["width"]==w]
    oa = np.array([r["test_acc"] for r in o]); na = np.array([r["test_acc"] for r in n])
    oe = np.array([r["test_ece"] for r in o]); ne = np.array([r["test_ece"] for r in n])
    print(f"width={w:>4} acc orig={oa.mean():.4f} fresh={na.mean():.4f} diff={abs(oa.mean()-na.mean()):.5f} | ece orig={oe.mean():.4f} fresh={ne.mean():.4f} diff={abs(oe.mean()-ne.mean()):.5f}")

# check per-seed exact match at width=16
o16 = sorted([r for r in orig if r["width"]==16], key=lambda r:r["seed"])
n16 = sorted([r for r in fresh if r["width"]==16], key=lambda r:r["seed"])
print()
print("Per-seed exact match check at width=16:")
for a,b in zip(o16,n16):
    print(a["seed"], a["test_acc"], b["test_acc"], a["test_acc"]==b["test_acc"])
