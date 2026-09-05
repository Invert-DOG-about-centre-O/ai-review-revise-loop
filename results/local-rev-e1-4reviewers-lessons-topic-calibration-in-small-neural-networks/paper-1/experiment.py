"""
Parameter-matched ensembles vs single networks: a calibration study.

Compares three model families at a FIXED total hidden-unit budget:
  - Single-Large : one MLP with H=64 hidden units
  - Ensemble-4   : four independent MLPs with H=16 hidden units each
                   (bootstrap-resampled data + independent init),
                   probabilities averaged
  - Single-Small : one MLP with H=16 hidden units (unensembled reference)

on two datasets:
  - Synthetic: K=4-class Gaussian mixture in R^10 with IDENTITY covariance
    and known means, so the TRUE Bayes posterior P(y|x) is analytically
    computable (softmax of -0.5||x-mu_k||^2). This lets us measure
    calibration against ground truth, not just empirical bin accuracy.
  - Real: sklearn `digits` (1797 8x8 images, 10 classes).

Every model is trained from an explicit, condition-specific RandomState
(never relying on carried-over global RNG state), so results for a given
(dataset, config, seed) are trained independently of everything else in
the loop -- see lessons.md item on RNG-state leakage across conditions.

Metrics (test set): accuracy, NLL, multiclass Brier score, ECE (15 bins),
and (synthetic only) mean L1 distance to the true Bayes posterior.
Each metric is reported BEFORE and AFTER post-hoc temperature scaling
(temperature fit on a held-out validation split by minimizing NLL).

15 independent seeds per (dataset, config); results saved as raw JSON.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon

N_SEEDS = 15
HIDDEN_LARGE = 64
HIDDEN_SMALL = 16
N_ENSEMBLE = 4
N_BINS = 15

# Fixed integer offsets per condition name -- NOT Python's built-in hash(),
# which is salted per-process (PYTHONHASHSEED) unless explicitly disabled,
# and was silently making every run of this script non-reproducible.
COND_OFFSET = {'single_large': 401, 'single_small': 733, 'ensemble4': 197}

# ---------------------------------------------------------------- datasets

def make_synthetic(seed=0, n_train=4000, n_val=1000, n_test=2000, d=10, k=4):
    rng = np.random.RandomState(seed)
    means = rng.normal(scale=1.8, size=(k, d))
    n_total = n_train + n_val + n_test
    y = rng.randint(0, k, size=n_total)
    X = means[y] + rng.normal(size=(n_total, d))
    # true Bayes posterior: identity covariance, equal priors ->
    # softmax of (x . mu_k - 0.5||mu_k||^2)
    logits = X @ means.T - 0.5 * (means ** 2).sum(1)[None, :]
    logits -= logits.max(1, keepdims=True)
    p_true = np.exp(logits)
    p_true /= p_true.sum(1, keepdims=True)
    idx_tr = slice(0, n_train)
    idx_va = slice(n_train, n_train + n_val)
    idx_te = slice(n_train + n_val, n_total)
    return (X[idx_tr], y[idx_tr], X[idx_va], y[idx_va],
            X[idx_te], y[idx_te], p_true[idx_te])


def make_digits(seed=0):
    data = load_digits()
    X, y = data.data, data.target
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, None


# ---------------------------------------------------------------- models

def train_mlp(X, y, hidden, seed, n_classes):
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation='relu', solver='adam',
        alpha=1e-4, max_iter=400, random_state=seed, early_stopping=False)
    clf.fit(X, y)
    return clf


def proba_full(clf, X, n_classes):
    """predict_proba padded to n_classes columns (classes_ order)."""
    p = clf.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    out = np.full((X.shape[0], n_classes), 1e-12)
    out[:, clf.classes_] = p
    return out / out.sum(1, keepdims=True)


def single_run(X_tr, y_tr, hidden, seed, n_classes):
    clf = train_mlp(X_tr, y_tr, hidden, seed, n_classes)
    return lambda X: proba_full(clf, X, n_classes)


def ensemble_run(X_tr, y_tr, hidden, seed, n_members, n_classes):
    members = []
    boot_rng = np.random.RandomState(seed * 7919 + 1)
    n = X_tr.shape[0]
    for m in range(n_members):
        idx = boot_rng.randint(0, n, size=n)
        member_seed = seed * 1009 + m * 131 + 3
        clf = train_mlp(X_tr[idx], y_tr[idx], hidden, member_seed, n_classes)
        members.append(clf)

    def predict(X):
        probs = [proba_full(c, X, n_classes) for c in members]
        return np.mean(probs, axis=0)
    return predict, members


# ---------------------------------------------------------------- metrics

def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return -np.log(p).mean()


def brier(probs, y, n_classes):
    onehot = np.eye(n_classes)[y]
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def ece(probs, y, n_bins=N_BINS):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        e += (mask.sum() / total) * abs(acc_bin - conf_bin)
    return e


def fit_temperature(probs_val, y_val):
    logp = np.log(np.clip(probs_val, 1e-12, 1.0))

    def loss(T):
        z = logp / T
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return nll(p, y_val)

    res = minimize_scalar(loss, bounds=(0.05, 10.0), method='bounded')
    return res.x


def apply_temperature(probs, T):
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    z = logp / T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p


# ---------------------------------------------------------------- driver

def evaluate(probs, y, n_classes, p_true=None):
    out = dict(nll=nll(probs, y), brier=brier(probs, y, n_classes),
               ece=ece(probs, y), acc=float((probs.argmax(1) == y).mean()),
               mean_conf=float(probs.max(1).mean()))
    if p_true is not None:
        out['l1_to_bayes'] = float(np.mean(np.sum(np.abs(probs - p_true), axis=1)))
    return out


def run_dataset(name, loader, n_classes):
    results = {'single_large': [], 'ensemble4': [], 'single_small': []}
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_va, y_va, X_te, y_te, p_true = loader(seed=seed)

        configs = {
            'single_large': ('single', HIDDEN_LARGE),
            'single_small': ('single', HIDDEN_SMALL),
            'ensemble4': ('ensemble', HIDDEN_SMALL),
        }
        for cname, (kind, hidden) in configs.items():
            cond_seed = seed * 31337 + COND_OFFSET[cname]
            if kind == 'single':
                predict = single_run(X_tr, y_tr, hidden, cond_seed, n_classes)
            else:
                predict, _ = ensemble_run(X_tr, y_tr, hidden, cond_seed,
                                           N_ENSEMBLE, n_classes)
            p_val = predict(X_va)
            p_test = predict(X_te)

            pre = evaluate(p_test, y_te, n_classes, p_true)
            T = fit_temperature(p_val, y_va)
            p_test_ts = apply_temperature(p_test, T)
            post = evaluate(p_test_ts, y_te, n_classes, p_true)

            n_params = hidden * (X_tr.shape[1] + n_classes) + hidden + n_classes
            if kind == 'ensemble':
                n_params *= N_ENSEMBLE

            results[cname].append(dict(seed=seed, T=float(T), n_params=int(n_params),
                                        pre=pre, post=post))
        print(f"[{name}] seed {seed} done")
    return results


def paired_wilcoxon(results, key_a, key_b, metric, phase='post'):
    a = np.array([r[phase][metric] for r in results[key_a]])
    b = np.array([r[phase][metric] for r in results[key_b]])
    diff = a - b
    if np.allclose(diff, 0):
        return dict(mean_diff=0.0, stat=None, p=1.0, n=len(a))
    stat, p = wilcoxon(a, b, alternative='greater')  # H1: a > b (a worse)
    return dict(mean_diff=float(diff.mean()), stat=float(stat), p=float(p), n=len(a))


def main():
    t0 = time.time()
    all_results = {}

    print("=== Synthetic (Gaussian mixture, known Bayes posterior) ===")
    all_results['synthetic'] = run_dataset('synthetic', make_synthetic, n_classes=4)

    print("=== Digits (sklearn, real data) ===")
    all_results['digits'] = run_dataset('digits', make_digits, n_classes=10)

    # Pre-registered test: H0: ECE(ensemble4, post-TS) >= ECE(single_large, post-TS)
    tests = {}
    for dname in ['synthetic', 'digits']:
        tests[dname] = dict(
            ece_ensemble_vs_large=paired_wilcoxon(
                all_results[dname], 'ensemble4', 'single_large', 'ece', 'post'),
            ece_ensemble_vs_large_pre=paired_wilcoxon(
                all_results[dname], 'ensemble4', 'single_large', 'ece', 'pre'),
            nll_ensemble_vs_large=paired_wilcoxon(
                all_results[dname], 'ensemble4', 'single_large', 'nll', 'post'),
        )

    out = dict(results=all_results, tests=tests, elapsed_sec=time.time() - t0)
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    for dname in ['synthetic', 'digits']:
        print(f"\n--- {dname} ---")
        for cname in ['single_large', 'ensemble4', 'single_small']:
            pre_ece = np.mean([r['pre']['ece'] for r in all_results[dname][cname]])
            post_ece = np.mean([r['post']['ece'] for r in all_results[dname][cname]])
            pre_conf = np.mean([r['pre']['mean_conf'] for r in all_results[dname][cname]])
            post_nll = np.mean([r['post']['nll'] for r in all_results[dname][cname]])
            acc = np.mean([r['post']['acc'] for r in all_results[dname][cname]])
            print(f"{cname:14s} acc={acc:.3f} pre_ece={pre_ece:.4f} post_ece={post_ece:.4f} "
                  f"pre_conf={pre_conf:.3f} post_nll={post_nll:.4f}")
        print("tests:", json.dumps(tests[dname], indent=2))


if __name__ == '__main__':
    main()
