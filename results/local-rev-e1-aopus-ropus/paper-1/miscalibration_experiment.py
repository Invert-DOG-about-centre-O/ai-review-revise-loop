"""
Miscalibration-robustness ablation for the likelihood-weighted SE estimator.

Reviewer's remaining objection (round 2, Q1/weakness 2): the noise sweep injects
SYMMETRIC noise centered on the TRUE q[m] --- log(q_hat)=log(q)+sigma*N(0,1) ---
so the corrupted likelihood is still unbiased around ground truth. Real decoder
log-probs are systematically MIScalibrated (typically overconfident), not merely
jittered. We model this the canonical way: a temperature distortion of the
model's own probabilities,
    q_hat[m] proportional to q[m] ** (1/T),
renormalized over observed clusters. T<1 = overconfident (sharpened), T>1 =
underconfident (flattened), T=1 = perfectly calibrated. This is a BIASED
corruption: the mode of q_hat[m] is not q[m]. We also test temperature bias
COMBINED with symmetric noise. Question: does the k=2 gain survive, and is the
robustness a property of the estimator or an artifact of unbiased noise?
"""
import numpy as np

def H_plugin(c):
    k = c.sum()
    p = c[c > 0] / k
    return -np.sum(p * np.log(p))

def H_lik_miscal(c, q, lognoise, T, sigma):
    # miscalibrated + optionally noisy weights over DISTINCT observed clusters
    obs = np.nonzero(c)[0]
    logw = np.log(q[obs]) / T + sigma * lognoise[obs]
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

def run(n_queries=4000, M=12, budgets=(2, 5, 10, 20), n_trials=20,
        temps=(0.5, 0.7, 1.0, 1.5, 2.0), combo_sigma=0.5):
    res = {k: {T: [] for T in temps} for k in budgets}
    combo = {k: {T: [] for T in temps} for k in budgets}  # T-bias + sigma noise
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
            lognoise = rng.normal(0, 1.0, size=(n_queries, M))
            plug_s = np.empty(n_queries)
            mis = {T: np.empty(n_queries) for T in temps}
            cmb = {T: np.empty(n_queries) for T in temps}
            for i in range(n_queries):
                c = np.bincount(labels[i], minlength=M)
                plug_s[i] = H_plugin(c)
                for T in temps:
                    mis[T][i] = H_lik_miscal(c, q[i], lognoise[i], T, 0.0)
                    cmb[T][i] = H_lik_miscal(c, q[i], lognoise[i], T, combo_sigma)
            plug[k].append(auroc(plug_s, incorrect))
            for T in temps:
                res[k][T].append(auroc(mis[T], incorrect))
                combo[k][T].append(auroc(cmb[T], incorrect))
    return res, combo, plug, temps, budgets, combo_sigma

if __name__ == "__main__":
    res, combo, plug, temps, budgets, cs = run()
    print("Miscalibration-robustness of likelihood-weighted SE (mean AUROC, 20 trials)")
    print("T = calibration temperature applied to model log-probs; T=1 perfect,")
    print("T<1 overconfident, T>1 underconfident. Weights q_hat ~ q**(1/T).\n")
    print(f"{'k':<5}{'plug-in':>10}" + "".join(f"{'T=%.1f'%T:>10}" for T in temps))
    for k in budgets:
        p = np.mean(plug[k])
        row = f"{k:<5}{p:>10.3f}"
        for T in temps:
            row += f"{np.mean(res[k][T]):>10.3f}"
        print(row)
    print("\nPaired delta vs plug-in (mean, paired t) under pure T miscalibration:")
    for k in budgets:
        print(f" k={k}:")
        for T in temps:
            d = np.array(res[k][T]) - np.array(plug[k])
            tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            print(f"   T={T:<4} delta={d.mean():+.4f} +- {d.std():.4f} (t={tstat:+.1f})")
    print(f"\nCombined: T miscalibration + symmetric sigma={cs} noise (mean AUROC):")
    print(f"{'k':<5}{'plug-in':>10}" + "".join(f"{'T=%.1f'%T:>10}" for T in temps))
    for k in budgets:
        p = np.mean(plug[k])
        row = f"{k:<5}{p:>10.3f}"
        for T in temps:
            row += f"{np.mean(combo[k][T]):>10.3f}"
        print(row)
