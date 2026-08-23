"""
Noise-robustness ablation for the likelihood-weighted SE estimator.

Reviewer's central objection: lik-weighted uses the TRUE noise-free q over
observed clusters, while count estimators see only tallies. In practice the
model emits a (noisy) length-normalized sequence log-prob per generation. We
therefore replace the exact log q[m] with a noisy observation
    log(q_hat[m]) = log(q[m]) + N(0, sigma^2)
for each DISTINCT observed cluster m, renormalize, take Shannon entropy, and
ask whether the k=2 AUROC gain survives as sigma grows.
"""
import numpy as np

def H_plugin(c):
    k = c.sum()
    p = c[c > 0] / k
    return -np.sum(p * np.log(p))

def H_lik_noisy(c, q, lognoise, sigma):
    obs = np.nonzero(c)[0]
    logw = np.log(q[obs]) + sigma * lognoise[obs]
    logw -= logw.max()
    w = np.exp(logw)
    w = w / w.sum()
    w = w[w > 0]
    return -np.sum(w * np.log(w))

def auroc(scores, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
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

def make_population(n_queries, M, rng):
    q = np.empty((n_queries, M))
    t = rng.integers(0, M, size=n_queries)
    correct = np.empty(n_queries, bool)
    for i in range(n_queries):
        s = rng.exponential(1.5)
        logits = rng.normal(0, 1.0, size=M)
        logits[t[i]] += s
        z = np.exp(logits - logits.max())
        q[i] = z / z.sum()
        correct[i] = (np.argmax(q[i]) == t[i])
    return q, t, correct

def run(n_queries=4000, M=12, budgets=(2, 5, 20), n_trials=20,
        sigmas=(0.0, 0.25, 0.5, 1.0, 2.0)):
    res = {k: {sig: [] for sig in sigmas} for k in budgets}
    plug = {k: [] for k in budgets}
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        q, t, correct = make_population(n_queries, M, rng)
        incorrect = ~correct
        for k in budgets:
            u = rng.random((n_queries, k))
            cdf = np.cumsum(q, axis=1)
            labels = (u[:, :, None] > cdf[:, None, :]).sum(axis=2)
            labels = np.minimum(labels, M - 1)
            lognoise = rng.normal(0, 1.0, size=(n_queries, M))  # fixed per query
            plug_s = np.empty(n_queries)
            noisy = {sig: np.empty(n_queries) for sig in sigmas}
            for i in range(n_queries):
                c = np.bincount(labels[i], minlength=M)
                plug_s[i] = H_plugin(c)
                for sig in sigmas:
                    noisy[sig][i] = H_lik_noisy(c, q[i], lognoise[i], sig)
            plug[k].append(auroc(plug_s, incorrect))
            for sig in sigmas:
                res[k][sig].append(auroc(noisy[sig], incorrect))
    return res, plug, sigmas, budgets

if __name__ == "__main__":
    res, plug, sigmas, budgets = run()
    print("Noise-robustness of likelihood-weighted SE (mean AUROC over 20 trials)")
    print("sigma = std of Gaussian noise added to log-likelihood of each observed cluster\n")
    print(f"{'k':<5}{'plug-in':>10}" + "".join(f"{'s=%.2f'%s:>10}" for s in sigmas))
    for k in budgets:
        p = np.mean(plug[k])
        row = f"{k:<5}{p:>10.3f}"
        for s in sigmas:
            row += f"{np.mean(res[k][s]):>10.3f}"
        print(row)
    print("\nPaired delta vs plug-in (mean +- std, paired t) at each sigma:")
    for k in budgets:
        print(f" k={k}:")
        for s in sigmas:
            d = np.array(res[k][s]) - np.array(plug[k])
            tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            print(f"   sigma={s:<5} delta={d.mean():+.4f} +- {d.std():.4f} (t={tstat:+.1f})")
    # gap-closed at k=2, sigma=0 sanity check
    p2 = np.mean(plug[2]); l2 = np.mean(res[2][0.0]); orc = 0.790
    print(f"\nk=2 sigma=0 gap-closed = (%.4f-%.4f)/(%.3f-%.4f) = %.1f%%"
          % (l2, p2, orc, p2, 100*(l2-p2)/(orc-p2)))
