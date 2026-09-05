"""Quick check: does post-TS ECE ranking/magnitude depend on bin count (10/15/20)?"""
import numpy as np
import experiment as E

n_classes = 10
out = {}
for seed in range(5):
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = E.make_digits(seed=seed)
    configs = {'single_large': ('single', E.HIDDEN_LARGE),
               'ensemble4': ('ensemble', E.HIDDEN_SMALL),
               'single_small': ('single', E.HIDDEN_SMALL)}
    for cname, (kind, hidden) in configs.items():
        cond_seed = seed * 31337 + E.COND_OFFSET[cname]
        if kind == 'single':
            predict = E.single_run(X_tr, y_tr, hidden, cond_seed, n_classes)
        else:
            predict, _ = E.ensemble_run(X_tr, y_tr, hidden, cond_seed, E.N_ENSEMBLE, n_classes)
        p_val = predict(X_va)
        p_test = predict(X_te)
        T = E.fit_temperature(p_val, y_va)
        p_test_ts = E.apply_temperature(p_test, T)
        for nb in [10, 15, 20]:
            out.setdefault((cname, nb), []).append(E.ece(p_test_ts, y_te, n_bins=nb))

for (cname, nb), vals in out.items():
    print(cname, 'bins=', nb, 'mean_post_ece=', round(float(np.mean(vals)), 4))
