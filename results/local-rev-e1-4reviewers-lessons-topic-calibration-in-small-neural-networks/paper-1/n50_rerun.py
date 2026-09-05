"""
Follow-up: reviewer 1 asked why we didn't just collect n=50 seeds for
digits within this submission, since digits training is fast. We run
seeds 15-49 (35 additional) for digits and combine with the existing
15 to get a properly-powered n=50 paired test on post-TS ECE.
"""
import json
import numpy as np
from scipy.stats import wilcoxon
from experiment import make_digits, single_run, evaluate, \
    fit_temperature, apply_temperature, HIDDEN_LARGE, HIDDEN_SMALL, \
    N_ENSEMBLE, COND_OFFSET, train_mlp, proba_full

N_EXTRA = 35  # seeds 15..49
SEED_MOD = 2**32 - 1  # experiment.py's raw multiply-seed scheme overflows
                       # numpy's 32-bit bound starting at raw seed=18 (noted
                       # in v3 limitations); fixed here via modulo so n>=18
                       # seeds are usable, still deterministic per seed.


def ensemble_run_safe(X_tr, y_tr, hidden, seed, n_members, n_classes):
    members = []
    boot_rng = np.random.RandomState((seed * 7919 + 1) % SEED_MOD)
    n = X_tr.shape[0]
    for m in range(n_members):
        idx = boot_rng.randint(0, n, size=n)
        member_seed = (seed * 1009 + m * 131 + 3) % SEED_MOD
        clf = train_mlp(X_tr[idx], y_tr[idx], hidden, member_seed, n_classes)
        members.append(clf)

    def predict(X):
        probs = [proba_full(c, X, n_classes) for c in members]
        return np.mean(probs, axis=0)
    return predict


def run_seed(seed, n_classes=10):
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = make_digits(seed=seed)
    configs = {'single_large': ('single', HIDDEN_LARGE),
               'single_small': ('single', HIDDEN_SMALL),
               'ensemble4': ('ensemble', HIDDEN_SMALL)}
    out = {}
    for cname, (kind, hidden) in configs.items():
        cond_seed = seed * 31337 + COND_OFFSET[cname]
        if kind == 'single':
            predict = single_run(X_tr, y_tr, hidden, cond_seed % SEED_MOD, n_classes)
        else:
            predict = ensemble_run_safe(X_tr, y_tr, hidden, cond_seed, N_ENSEMBLE, n_classes)
        p_val, p_test = predict(X_va), predict(X_te)
        pre = evaluate(p_test, y_te, n_classes)
        T = fit_temperature(p_val, y_va)
        post = evaluate(apply_temperature(p_test, T), y_te, n_classes)
        out[cname] = dict(seed=seed, pre=pre, post=post)
    return out


def main():
    existing = json.load(open('results.json'))['results']['digits']
    combined = {k: list(v) for k, v in existing.items()}
    for seed in range(15, 15 + N_EXTRA):
        r = run_seed(seed)
        for cname in combined:
            combined[cname].append(r[cname])
        print(f"seed {seed} done")

    def col(cname, phase, key):
        return np.array([r[phase][key] for r in combined[cname]])

    a, b = col('ensemble4', 'post', 'ece'), col('single_large', 'post', 'ece')
    diff = a - b
    stat, p = wilcoxon(a, b, alternative='greater')
    d = diff.mean() / diff.std(ddof=1)
    print(f"\nn={len(a)} post-TS ECE: ensemble4={a.mean():.4f} single_large={b.mean():.4f} "
          f"mean_diff={diff.mean():.5f} wilcoxon p={p:.4f} cohen_d={d:.4f}")

    with open('n50_results.json', 'w') as f:
        json.dump(dict(n=len(a), ensemble4_mean=float(a.mean()),
                        single_large_mean=float(b.mean()),
                        mean_diff=float(diff.mean()), p=float(p), cohen_d=float(d)), f, indent=2)


if __name__ == '__main__':
    main()
