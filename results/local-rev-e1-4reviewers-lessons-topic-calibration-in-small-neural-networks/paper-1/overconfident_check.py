"""
Follow-up requested by two independent reviewers: does the classic
"ensembling corrects overconfidence" story re-emerge if the base networks
are made deliberately overconfident (near-zero weight decay, trained to
convergence, no early stopping) instead of the mildly-regularized
(alpha=1e-4) networks used in the main experiment?

Same architecture/budget comparison as experiment.py (Single-Large H=64 vs
Ensemble-4 x H=16), digits only, 15 seeds, alpha=1e-8 instead of 1e-4.
"""
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
import experiment as E

ALPHA_OVERCONF = 1e-8


def train_mlp_overconf(X, y, hidden, seed, n_classes):
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation='relu', solver='adam',
        alpha=ALPHA_OVERCONF, max_iter=800, random_state=seed, early_stopping=False)
    clf.fit(X, y)
    return clf


def single_run(X_tr, y_tr, hidden, seed, n_classes):
    clf = train_mlp_overconf(X_tr, y_tr, hidden, seed, n_classes)
    return lambda X: E.proba_full(clf, X, n_classes)


def ensemble_run(X_tr, y_tr, hidden, seed, n_members, n_classes):
    members = []
    boot_rng = np.random.RandomState(seed * 7919 + 1)
    n = X_tr.shape[0]
    for m in range(n_members):
        idx = boot_rng.randint(0, n, size=n)
        member_seed = seed * 1009 + m * 131 + 3
        clf = train_mlp_overconf(X_tr[idx], y_tr[idx], hidden, member_seed, n_classes)
        members.append(clf)

    def predict(X):
        probs = [E.proba_full(c, X, n_classes) for c in members]
        return np.mean(probs, axis=0)
    return predict


def run():
    n_classes = 10
    results = {'single_large': [], 'ensemble4': []}
    for seed in range(E.N_SEEDS):
        X_tr, y_tr, X_va, y_va, X_te, y_te, _ = E.make_digits(seed=seed)
        configs = {'single_large': ('single', E.HIDDEN_LARGE),
                   'ensemble4': ('ensemble', E.HIDDEN_SMALL)}
        for cname, (kind, hidden) in configs.items():
            cond_seed = seed * 31337 + E.COND_OFFSET[cname]
            if kind == 'single':
                predict = single_run(X_tr, y_tr, hidden, cond_seed, n_classes)
            else:
                predict = ensemble_run(X_tr, y_tr, hidden, cond_seed, E.N_ENSEMBLE, n_classes)
            p_val = predict(X_va)
            p_test = predict(X_te)
            pre = E.evaluate(p_test, y_te, n_classes)
            T = E.fit_temperature(p_val, y_va)
            p_test_ts = E.apply_temperature(p_test, T)
            post = E.evaluate(p_test_ts, y_te, n_classes)
            results[cname].append(dict(seed=seed, T=float(T), pre=pre, post=post))
        print(f"[overconf digits] seed {seed} done")
    return results


def main():
    results = run()
    tests = dict(
        ece_post=E.paired_wilcoxon(results, 'ensemble4', 'single_large', 'ece', 'post'),
        ece_pre=E.paired_wilcoxon(results, 'ensemble4', 'single_large', 'ece', 'pre'),
    )
    summary = {}
    for cname in ['single_large', 'ensemble4']:
        pre_ece = np.mean([r['pre']['ece'] for r in results[cname]])
        post_ece = np.mean([r['post']['ece'] for r in results[cname]])
        pre_conf = np.mean([r['pre']['mean_conf'] for r in results[cname]])
        acc = np.mean([r['pre']['acc'] for r in results[cname]])
        signed_ece_pre = np.mean([r['pre']['mean_conf'] - r['pre']['acc'] for r in results[cname]])
        summary[cname] = dict(acc=float(acc), pre_ece=float(pre_ece), post_ece=float(post_ece),
                               pre_conf=float(pre_conf), signed_pre=float(signed_ece_pre))
        print(cname, summary[cname])
    print("tests:", json.dumps(tests, indent=2))
    with open('overconfident_results.json', 'w') as f:
        json.dump(dict(results=results, tests=tests, summary=summary), f, indent=2)


if __name__ == '__main__':
    main()
