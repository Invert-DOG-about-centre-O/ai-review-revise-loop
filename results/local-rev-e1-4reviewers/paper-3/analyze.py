import json
import math

import numpy as np

with open("raw_results.json") as f:
    data = json.load(f)

records = data["records"]
n = len(records)
correct = np.array([r["correct"] for r in records], dtype=bool)
wrong = ~correct
print(f"N={n}, accuracy={data['accuracy']:.3f}, K_samples={data['k_samples']}")
print(f"n_correct={correct.sum()}, n_wrong={wrong.sum()}")


def auroc(scores, is_positive):
    """AUROC for 'is_positive' (wrong=1) via Mann-Whitney U, scores higher => more likely positive."""
    scores = np.asarray(scores, dtype=float)
    pos = scores[is_positive]
    neg = scores[~is_positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties with average rank
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[is_positive].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def ece(conf, correct_mask, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(conf)
    e = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        acc_bin = correct_mask[mask].mean()
        conf_bin = conf[mask].mean()
        e += (cnt / total) * abs(acc_bin - conf_bin)
        rows.append((lo, hi, cnt, acc_bin, conf_bin))
    return e, rows


def brier(conf, correct_mask):
    conf = np.asarray(conf, dtype=float)
    y = correct_mask.astype(float)
    return float(np.mean((conf - y) ** 2))


ent1 = np.array([r["first_token_entropy"] for r in records])
mean_logp = np.array([r["mean_logp"] for r in records])
sem_ent = np.array([r["semantic_entropy"] for r in records])
sc_conf = np.array([r["self_consistency_conf"] for r in records])

# logp -> pseudo-confidence via exp (it's a mean log-prob per char, so exp is in (0,1])
logp_conf = np.exp(mean_logp)

print("\n=== AUROC for detecting WRONG greedy answers (higher score = more likely wrong) ===")
results_auroc = {}
for name, scores in [
    ("first_token_entropy (1 pass)", ent1),
    ("neg_mean_logp (1 pass, greedy)", -mean_logp),
    ("semantic_entropy (K=%d samples)" % data["k_samples"], sem_ent),
    ("neg_self_consistency_conf (K=%d samples)" % data["k_samples"], -sc_conf),
]:
    a = auroc(scores, wrong)
    results_auroc[name] = a
    print(f"  {name:45s} AUROC={a:.3f}")

print("\n=== Calibration of confidence scores (predict P(correct)) ===")
results_cal = {}
for name, conf in [
    ("logp_conf = exp(mean_logp) (1 pass)", logp_conf),
    ("self_consistency_conf (K=%d samples)" % data["k_samples"], sc_conf),
]:
    e, rows = ece(conf, correct)
    b = brier(conf, correct)
    results_cal[name] = dict(ece=e, brier=b)
    print(f"  {name:45s} ECE={e:.3f}  Brier={b:.3f}")

print("\n=== Reliability table: self_consistency_conf ===")
_, rows = ece(sc_conf, correct)
for lo, hi, cnt, acc_bin, conf_bin in rows:
    print(f"  [{lo:.1f},{hi:.1f}) n={cnt:4d}  acc={acc_bin:.3f}  conf={conf_bin:.3f}")

print("\n=== Reliability table: logp_conf ===")
_, rows2 = ece(logp_conf, correct)
for lo, hi, cnt, acc_bin, conf_bin in rows2:
    print(f"  [{lo:.1f},{hi:.1f}) n={cnt:4d}  acc={acc_bin:.3f}  conf={conf_bin:.3f}")

# Cost-accuracy tradeoff: how does semantic entropy / self-consistency AUROC
# degrade as K shrinks (subsample from the 8 stored samples)?
print("\n=== Cost-accuracy tradeoff: AUROC vs number of samples K ===")
rng = np.random.default_rng(0)
tradeoff = {}
for K in [1, 2, 3, 4, 6, 8]:
    aurocs_sem, aurocs_sc = [], []
    for _ in range(50):  # bootstrap resample which K samples we "use"
        sub_sem, sub_sc = [], []
        for r in records:
            idx = rng.choice(len(r["samples"]), size=K, replace=False)
            subs = [r["samples"][j] for j in idx]
            counts = {}
            for a in subs:
                key = a if a is not None else "PARSE_FAIL"
                counts[key] = counts.get(key, 0) + 1
            e = -sum((c / K) * math.log(c / K) for c in counts.values())
            modal_count = max(counts.values())
            sub_sem.append(e)
            sub_sc.append(modal_count / K)
        aurocs_sem.append(auroc(np.array(sub_sem), wrong))
        aurocs_sc.append(auroc(-np.array(sub_sc), wrong))
    tradeoff[K] = dict(
        sem_ent_auroc_mean=float(np.mean(aurocs_sem)),
        sem_ent_auroc_std=float(np.std(aurocs_sem)),
        sc_auroc_mean=float(np.mean(aurocs_sc)),
        sc_auroc_std=float(np.std(aurocs_sc)),
    )
    print(f"  K={K}: semantic_entropy AUROC={tradeoff[K]['sem_ent_auroc_mean']:.3f}"
          f" (+/-{tradeoff[K]['sem_ent_auroc_std']:.3f})"
          f"   self_consistency AUROC={tradeoff[K]['sc_auroc_mean']:.3f}"
          f" (+/-{tradeoff[K]['sc_auroc_std']:.3f})")

print(f"\ncheap single-pass baseline (K=0 cost): "
      f"first_token_entropy AUROC={results_auroc['first_token_entropy (1 pass)']:.3f}, "
      f"neg_mean_logp AUROC={results_auroc['neg_mean_logp (1 pass, greedy)']:.3f}")

summary = dict(
    n=n, accuracy=data["accuracy"], k_samples=data["k_samples"],
    auroc=results_auroc, calibration=results_cal, tradeoff=tradeoff,
)
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved analysis_summary.json")
