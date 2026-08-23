import json
import math

import numpy as np

with open("raw_results_verify.json") as f:
    data = json.load(f)

records = data["records"]
n = len(records)
correct = np.array([r["correct"] for r in records], dtype=bool)
wrong = ~correct
print(f"N={n}, accuracy={data['accuracy']:.3f}, K_samples={data['k_samples']}")
print(f"n_correct={correct.sum()}, n_wrong={wrong.sum()}")


def auroc(scores, is_positive):
    scores = np.asarray(scores, dtype=float)
    pos = scores[is_positive]
    neg = scores[~is_positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
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
    return e


def brier(conf, correct_mask):
    conf = np.asarray(conf, dtype=float)
    y = correct_mask.astype(float)
    return float(np.mean((conf - y) ** 2))


ent1 = np.array([r["first_token_entropy"] for r in records])
mean_logp = np.array([r["mean_logp"] for r in records])
sem_ent = np.array([r["semantic_entropy"] for r in records])
sc_conf = np.array([r["self_consistency_conf"] for r in records])
logp_conf = np.exp(mean_logp)

print("\n=== AUROC ===")
for name, scores in [
    ("first_token_entropy", ent1),
    ("neg_mean_logp", -mean_logp),
    ("semantic_entropy", sem_ent),
    ("neg_self_consistency_conf", -sc_conf),
]:
    print(f"  {name:30s} AUROC={auroc(scores, wrong):.3f}")

print("\n=== Calibration ===")
for name, conf in [
    ("logp_conf", logp_conf),
    ("self_consistency_conf", sc_conf),
]:
    print(f"  {name:30s} ECE={ece(conf, correct):.3f}  Brier={brier(conf, correct):.3f}")
