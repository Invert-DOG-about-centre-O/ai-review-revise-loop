"""
Round-3 experiment answering review Q3: can a CALIBRATION-CORRECTED confidence
signal restore comparability WITHOUT observing g at decision time (the motivating
legally-excluded-attribute case)?

Idea: instead of the raw max-prob confidence c (which is uninformative on the
minority), learn a "verifier" v(x) = P(AI correct | x) on the validation set --
a logistic-regression correctness predictor over the SAME features X (g never
used). Because the group label rules differ, the region where the AI is
systematically wrong is partly recoverable from X, so a GLOBAL threshold on v
can defer the confident-but-wrong minority cases without ever seeing g.

Compare, over 30 seeds:
  - Global raw-confidence threshold      (standard; needs no g)
  - Group-conditional on raw confidence  (fairness fix; needs g)
  - Global threshold on verifier v(x)    (ours here; needs no g)
CPU-only, deterministic given the seed grid.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from round2_experiments import (make_population, simulate_human, route, err_gap,
                                 fit_ai, tune_global, tune_groupcond, ci95, wB0)

SEEDS = list(range(30))


def one_run_verifier(seed, wBvec, human_acc=0.75):
    rng = np.random.default_rng(seed)
    clf = fit_ai(rng, wBvec)
    Xva, yva, gva = make_population(3000, rng, wBvec)
    Xte, yte, gte = make_population(6000, rng, wBvec)

    def ai(X):
        pr = clf.predict_proba(X)
        return pr.argmax(1), pr.max(1)
    ai_va, c_va = ai(Xva)
    ai_te, c_te = ai(Xte)
    hum_va = simulate_human(yva, rng, human_acc)
    hum_te = simulate_human(yte, rng, human_acc)
    taus = np.linspace(0.5, 0.99, 50)

    # --- baselines ---
    bt = tune_global(c_va, ai_va, hum_va, yva, taus)
    pg = route(ai_te, hum_te, c_te >= bt)
    tg = tune_groupcond(c_va, ai_va, hum_va, yva, gva, taus)
    ugc = np.where(gte == 0, c_te >= tg[0], c_te >= tg[1])
    pgc = route(ai_te, hum_te, ugc)

    # --- verifier: learned P(AI correct | X, conf), NO g used ---
    corr_va = (ai_va == yva).astype(int)
    Fva = np.column_stack([Xva, c_va])
    Fte = np.column_stack([Xte, c_te])
    ver = LogisticRegression(max_iter=1000).fit(Fva, corr_va)
    v_va = ver.predict_proba(Fva)[:, 1]
    v_te = ver.predict_proba(Fte)[:, 1]
    # tune a GLOBAL threshold on the verifier score for max team accuracy
    svs = np.linspace(v_va.min(), v_va.max(), 50)
    sv = tune_global(v_va, ai_va, hum_va, yva, svs)
    pv = route(ai_te, hum_te, v_te >= sv)

    return dict(
        acc_g=np.mean(pg == yte), gap_g=err_gap(yte, pg, gte),
        acc_gc=np.mean(pgc == yte), gap_gc=err_gap(yte, pgc, gte),
        acc_v=np.mean(pv == yte), gap_v=err_gap(yte, pv, gte))


print("=" * 68)
print("Q3: CALIBRATION-CORRECTED (verifier) confidence WITHOUT g, 30 seeds")
print("=" * 68)
runs = [one_run_verifier(s, wB0) for s in SEEDS]
for key, lbl, needg in [("g", "Global raw-conf (no g)", "no"),
                        ("gc", "Group-cond raw-conf (needs g)", "yes"),
                        ("v", "Verifier global (no g)", "no")]:
    a, ah = ci95([r[f"acc_{key}"] for r in runs])
    gp, gh = ci95([r[f"gap_{key}"] for r in runs])
    print(f"  {lbl:<32} needs_g={needg:>3}  "
          f"acc={a:.3f}+/-{ah:.3f}  gap={gp:.3f}+/-{gh:.3f}")
print("\nDONE")
