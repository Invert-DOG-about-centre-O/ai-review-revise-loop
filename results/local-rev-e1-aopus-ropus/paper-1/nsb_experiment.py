"""
NSB / Bayesian-posterior-entropy estimator vs plug-in and likelihood-weighted SE.

Reviewer round-3 Q3: the intro frames the problem against Nikitin et al.'s
"Bayesian budget" estimator, which reasons about entropy under a POSTERIOR over
the meaning distribution, yet no such integrating estimator appears in the main
table (bayes-KT is only a fixed-Dirichlet posterior-MEAN of the plug-in probs).
We add the Nemenman-Shafee-Bialek (NSB) estimator: posterior-mean Shannon
entropy under a Dirichlet(beta) prior, with beta itself integrated out against a
prior chosen so the a-priori expected entropy is uniform. This is the canonical
"integrate over the meaning distribution" estimator the round-2/3 reviewer asks
for. Population/sampling seeds match experiment.py exactly (rng=default_rng(1000+
trial)) so the plug-in and lik-weighted rows reproduce the main table.
"""
import numpy as np
from scipy.special import digamma, gammaln, polygamma

M = 12

def H_plugin(c):
    k = c.sum()
    p = c[c > 0] / k
    return -np.sum(p * np.log(p))

def H_lik_weighted(c, q):
    obs = np.nonzero(c)[0]
    w = q[obs]; w = w / w.sum()
    return -np.sum(w * np.log(w))

# ---- NSB machinery, vectorised over queries for a fixed beta -----------------
def make_beta_grid(nb=60):
    # spread beta on a log grid; prior weight p(beta) ~ d(mean entropy)/d(beta)
    beta = np.logspace(-3, 2, nb)
    kappa = M * beta
    prior = M * polygamma(1, kappa + 1) - polygamma(1, beta + 1)  # d xi_bar/d beta
    return beta, prior

BETA, PRIOR = make_beta_grid()

def H_nsb(counts):
    # counts: (n, M) integer counts for one budget/trial. Returns (n,) NSB entropy.
    n = counts.shape[0]
    N = counts.sum(axis=1)                     # = k, constant
    logZ = np.empty((len(BETA), n))
    EH   = np.empty((len(BETA), n))
    for b, beta in enumerate(BETA):
        kappa = M * beta
        # marginal evidence log p(counts | beta)
        logZ[b] = (gammaln(kappa) - gammaln(N + kappa)
                   + gammaln(counts + beta).sum(axis=1) - M * gammaln(beta))
        # posterior-mean entropy under Dirichlet(beta) given counts (Wolpert-Wolf)
        denom = N + kappa
        EH[b] = (digamma(denom + 1)
                 - (( (counts + beta) * digamma(counts + beta + 1) ).sum(axis=1) / denom))
    logw = logZ + np.log(PRIOR)[:, None]       # (nb, n)
    logw -= logw.max(axis=0, keepdims=True)
    w = np.exp(logw); w /= w.sum(axis=0, keepdims=True)
    return (w * EH).sum(axis=0)

def auroc(scores, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def make_population(n_queries, rng):
    q = np.empty((n_queries, M)); t = rng.integers(0, M, size=n_queries)
    correct = np.empty(n_queries, bool)
    for i in range(n_queries):
        s = rng.exponential(1.5)
        logits = rng.normal(0, 1.0, size=M); logits[t[i]] += s
        z = np.exp(logits - logits.max()); q[i] = z / z.sum()
        correct[i] = (np.argmax(q[i]) == t[i])
    return q, t, correct

def run(n_queries=4000, budgets=(2, 3, 5, 10, 20), n_trials=20):
    names = ["plugin(SE)", "bayes-NSB", "lik-weighted(ours)"]
    res = {k: {nm: [] for nm in names} for k in budgets}
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        q, t, correct = make_population(n_queries, rng)
        incorrect = ~correct
        for k in budgets:
            u = rng.random((n_queries, k)); cdf = np.cumsum(q, axis=1)
            labels = (u[:, :, None] > cdf[:, None, :]).sum(axis=2)
            labels = np.minimum(labels, M - 1)
            counts = np.zeros((n_queries, M), int)
            for i in range(n_queries):
                counts[i] = np.bincount(labels[i], minlength=M)
            pl = np.array([H_plugin(counts[i]) for i in range(n_queries)])
            lw = np.array([H_lik_weighted(counts[i], q[i]) for i in range(n_queries)])
            ns = H_nsb(counts)
            res[k]["plugin(SE)"].append(auroc(pl, incorrect))
            res[k]["bayes-NSB"].append(auroc(ns, incorrect))
            res[k]["lik-weighted(ours)"].append(auroc(lw, incorrect))
    return res, names, budgets

if __name__ == "__main__":
    res, names, budgets = run()
    print("NSB (integrating Bayesian) vs plug-in vs lik-weighted, mean AUROC / 20 trials\n")
    print(f"{'estimator':<20}" + "".join(f"{'k=%d'%k:>10}" for k in budgets))
    for nm in names:
        print(f"{nm:<20}" + "".join(f"{np.mean(res[k][nm]):>10.3f}" for k in budgets))
    print("\nNSB minus plug-in (paired, mean +- std, t):")
    for k in budgets:
        d = np.array(res[k]["bayes-NSB"]) - np.array(res[k]["plugin(SE)"])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  k={k:<3} delta={d.mean():+.4f} +- {d.std():.4f} (t={t:+.1f})")
