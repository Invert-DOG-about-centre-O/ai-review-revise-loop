"""
Bias-corrected semantic entropy for hallucination detection under a sample budget.

Fully synthetic, CPU-only simulation. We model an LLM's per-query output
distribution over *semantic meaning clusters* q (semantic clustering assumed
perfect, following Kuhn et al. 2023 / Farquhar et al. 2024). For each query the
greedy answer is CORRECT iff argmax q equals the true meaning t. We draw k
generations from q, estimate an uncertainty score, and measure how well that
score separates correct from incorrect answers (AUROC), sweeping the sample
budget k. We compare the naive plug-in Shannon-entropy estimator (standard
discrete semantic entropy) against classical bias-corrected entropy estimators.
"""
import numpy as np

rng_master = np.random.default_rng(0)

# ---------------- entropy estimators from a sample of cluster labels ----------
def counts_from_labels(labels, M):
    c = np.bincount(labels, minlength=M)
    return c

def H_plugin(c):
    k = c.sum()
    p = c[c > 0] / k
    return -np.sum(p * np.log(p))

def H_miller_madow(c):
    k = c.sum()
    Kobs = np.count_nonzero(c)
    return H_plugin(c) + (Kobs - 1) / (2.0 * k)

def H_chao_shen(c):
    # Chao & Shen (2003) coverage-adjusted entropy estimator
    k = c.sum()
    c = c[c > 0].astype(float)
    f1 = np.sum(c == 1)
    if f1 == k:            # avoid C=0 degeneracy
        f1 = k - 1
    C = 1.0 - f1 / k       # Good-Turing sample coverage
    pa = C * c / k         # coverage-adjusted probabilities
    pa = pa[pa > 0]
    denom = 1.0 - (1.0 - pa) ** k
    return -np.sum(pa * np.log(pa) / denom)

def H_bayes_kt(c):
    # Krichevsky-Trofimov: posterior-mean probs under symmetric Dirichlet(1/2)
    M = len(c)
    a = 0.5
    p = (c + a) / (c.sum() + a * M)
    return -np.sum(p * np.log(p))

def num_distinct(c):
    return float(np.count_nonzero(c))

def one_minus_max(c):
    k = c.sum()
    return 1.0 - c.max() / k

def H_likelihood_weighted(c, q):
    # Rao-Blackwellised / discrete semantic entropy (Farquhar et al. 2024 style):
    # for each DISTINCT observed meaning cluster, use the model's own likelihood
    # q[cluster] (proxy for length-normalized sequence probability) as its mass,
    # normalize over the observed clusters, then take Shannon entropy.
    obs = np.nonzero(c)[0]
    w = q[obs]
    w = w / w.sum()
    return -np.sum(w * np.log(w))

# ---------------- AUROC (rank-based Mann-Whitney) ----------------------------
def auroc(scores, pos):
    # pos: boolean, True = incorrect answer (the positive class we detect)
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

# ---------------- query population model -------------------------------------
def make_population(n_queries, M, rng):
    """Return q (n,M) output dists, t true meaning, and greedy correctness."""
    q = np.empty((n_queries, M))
    t = rng.integers(0, M, size=n_queries)
    correct = np.empty(n_queries, bool)
    for i in range(n_queries):
        # signal strength of the true meaning: controls peakedness => difficulty
        s = rng.exponential(1.5)
        logits = rng.normal(0, 1.0, size=M)      # background noise over meanings
        logits[t[i]] += s                        # true meaning gets a boost
        z = np.exp(logits - logits.max())
        q[i] = z / z.sum()
        correct[i] = (np.argmax(q[i]) == t[i])
    return q, t, correct

ESTIMATORS = {
    "plugin(SE)": H_plugin,
    "miller-madow": H_miller_madow,
    "chao-shen": H_chao_shen,
    "bayes-KT": H_bayes_kt,
    "num-distinct": num_distinct,
    "1-maxprob": one_minus_max,
}
# "lik-weighted(ours)" is handled specially (needs the per-generation likelihoods)

def run(n_queries=4000, M=12, budgets=(2, 3, 5, 10, 20), n_trials=20):
    all_names = list(ESTIMATORS) + ["lik-weighted(ours)"]
    results = {k: {name: [] for name in all_names} for k in budgets}
    oracle = {k: [] for k in budgets}
    base_acc = []
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        q, t, correct = make_population(n_queries, M, rng)
        base_acc.append(correct.mean())
        incorrect = ~correct
        true_H = np.array([-np.sum(qi[qi > 0] * np.log(qi[qi > 0])) for qi in q])
        for k in budgets:
            # sample k generations per query
            u = rng.random((n_queries, k))
            cdf = np.cumsum(q, axis=1)
            labels = (u[:, :, None] > cdf[:, None, :]).sum(axis=2)  # (n,k)
            labels = np.minimum(labels, M - 1)
            scores = {name: np.empty(n_queries) for name in all_names}
            for i in range(n_queries):
                c = counts_from_labels(labels[i], M)
                for name, fn in ESTIMATORS.items():
                    scores[name][i] = fn(c)
                scores["lik-weighted(ours)"][i] = H_likelihood_weighted(c, q[i])
            for name in all_names:
                results[k][name].append(auroc(scores[name], incorrect))
            oracle[k].append(auroc(true_H, incorrect))
    return results, oracle, np.mean(base_acc), budgets

if __name__ == "__main__":
    results, oracle, base_acc, budgets = run()
    print(f"Synthetic LLM semantic-UQ simulation")
    print(f"greedy base accuracy = {base_acc:.3f}  (task: detect the incorrect answers)\n")
    header = ["k=%d" % k for k in budgets]
    names = list(ESTIMATORS.keys()) + ["lik-weighted(ours)"]
    print(f"{'estimator':<14} " + " ".join(f"{h:>14}" for h in header))
    for name in names:
        row = []
        for k in budgets:
            a = np.array(results[k][name])
            row.append(f"{a.mean():.3f}+-{a.std():.3f}")
        print(f"{name:<14} " + " ".join(f"{r:>14}" for r in row))
    row = []
    for k in budgets:
        a = np.array(oracle[k])
        row.append(f"{a.mean():.3f}+-{a.std():.3f}")
    print(f"{'ORACLE(true H)':<14} " + " ".join(f"{r:>14}" for r in row))

    # paired improvement of Miller-Madow over plugin, per trial, per k
    print("\nMiller-Madow minus plugin AUROC (paired, mean +- std over trials):")
    for k in budgets:
        d = np.array(results[k]["miller-madow"]) - np.array(results[k]["plugin(SE)"])
        # simple paired t stat
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  k={k:<3} delta={d.mean():+.4f} +- {d.std():.4f}  (paired t={tstat:+.2f}, n={len(d)})")
    print("\nChao-Shen minus plugin AUROC (paired):")
    for k in budgets:
        d = np.array(results[k]["chao-shen"]) - np.array(results[k]["plugin(SE)"])
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  k={k:<3} delta={d.mean():+.4f} +- {d.std():.4f}  (paired t={tstat:+.2f}, n={len(d)})")
    print("\nlik-weighted(ours) minus plugin AUROC (paired):")
    for k in budgets:
        d = np.array(results[k]["lik-weighted(ours)"]) - np.array(results[k]["plugin(SE)"])
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  k={k:<3} delta={d.mean():+.4f} +- {d.std():.4f}  (paired t={tstat:+.2f}, n={len(d)})")
