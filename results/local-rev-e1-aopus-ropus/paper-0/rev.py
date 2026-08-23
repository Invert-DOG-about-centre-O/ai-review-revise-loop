"""Round-3 revision experiments:
(Q1) non-linear estimator (gradient-boosted trees) -- do rankings survive?
(Q2) DEPLOYABLE equal-opportunity (split from predicted probs, no oracle labels).
(Q3) budget-wise paired greedy-vs-parity significance.
"""
import numpy as np, json
import sim
from sim import make_cohort, fit_logreg, predict, sigmoid
from sklearn.ensemble import GradientBoostingClassifier

N = 8

def gbt_fit(X, y):
    if len(np.unique(y)) < 2:
        c = float(y.mean()); return ("const", c)
    m = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.2)
    m.fit(X, y); return ("gbt", m)

def gbt_pred(mdl, X):
    kind, m = mdl
    if kind == "const": return np.full(len(X), m)
    return m.predict_proba(X)[:, 1]

def run(policy, estimator="logreg", rounds=25, cohort=400, budget=0.5, eps=0.15,
        seed=0, offset=1.3):
    sim.OFFSET = offset
    rng = np.random.default_rng(seed)
    Xs, gs, qs, rs = make_cohort(600, rng)
    seed_mask = (gs == 0) | (rng.random(len(gs)) < 0.15)
    Xhist = [Xs[seed_mask]]; yhist = [rs[seed_mask]]
    H = {"appr_gap": [], "tpr_gap": [], "util": [], "oracle_util": []}
    for t in range(rounds):
        Xtr = np.vstack(Xhist); ytr = np.concatenate(yhist)
        if estimator == "gbt":
            mdl = gbt_fit(Xtr, ytr); scoref = lambda X: gbt_pred(mdl, X)
        else:
            th = fit_logreg(Xtr, ytr); scoref = lambda X: predict(th, X)
        X, g, q, repay = make_cohort(cohort, rng)
        score = scoref(X); k = int(budget * cohort)
        approve = np.zeros(cohort, bool)
        if policy == "parity":
            for grp in (0, 1):
                idx = np.where(g == grp)[0]; kk = int(budget * len(idx))
                approve[idx[np.argsort(-score[idx])[:kk]]] = True
        elif policy in ("eqopp", "eqopp_deploy"):
            idxA = np.where(g == 0)[0]; idxB = np.where(g == 1)[0]
            oA = idxA[np.argsort(-score[idxA])]; oB = idxB[np.argsort(-score[idxB])]
            if policy == "eqopp":       # ORACLE: uses realized cohort labels
                qA = repay[oA]; qB = repay[oB]
            else:                        # DEPLOYABLE: uses predicted repay probs
                qA = score[oA]; qB = score[oB]
            totA = max(qA.sum(), 1e-6); totB = max(qB.sum(), 1e-6)
            cumA = np.concatenate([[0], np.cumsum(qA)]); cumB = np.concatenate([[0], np.cumsum(qB)])
            best = None
            for kA in range(0, k + 1):
                kB = k - kA
                if kA > len(idxA) or kB > len(idxB): continue
                d = abs(cumA[kA] / totA - cumB[kB] / totB)
                if best is None or d < best[0]: best = (d, kA, kB)
            _, kA, kB = best
            approve[oA[:kA]] = True; approve[oB[:kB]] = True
        elif policy == "explore_targeted":
            n_rand = int(eps * k); order = np.argsort(-score)
            approve[order[:k - n_rand]] = True; rest = order[k - n_rand:]
            restB = rest[g[rest] == 1]
            if len(restB): approve[rng.choice(restB, size=min(n_rand, len(restB)), replace=False)] = True
        else:
            approve[np.argsort(-score)[:k]] = True
        aA = approve[g == 0].mean(); aB = approve[g == 1].mean()
        def tpr(grp):
            m = (g == grp) & (repay == 1); return approve[m].mean() if m.sum() > 0 else 0.0
        util = repay[approve].sum() - 0.6 * (1 - repay[approve]).sum()
        oracle = np.zeros(cohort, bool); oracle[np.argsort(-sigmoid(1.2 * q))[:k]] = True
        outil = repay[oracle].sum() - 0.6 * (1 - repay[oracle]).sum()
        H["appr_gap"].append(aA - aB); H["tpr_gap"].append(tpr(0) - tpr(1))
        H["util"].append(util); H["oracle_util"].append(outil)
        Xhist.append(X[approve]); yhist.append(repay[approve])
    return {k: np.array(v) for k, v in H.items()}

def agg(policy, **kw):
    G = []; TG = []; R = []
    for s in range(N):
        h = run(policy, seed=s, **kw)
        G.append(h["appr_gap"].mean()); TG.append(h["tpr_gap"][-1])
        R.append(100 * (1 - h["util"].sum() / h["oracle_util"].sum()))
    G = np.array(G); TG = np.array(TG); R = np.array(R)
    se = lambda x: x.std(ddof=1) / np.sqrt(N)
    return dict(gap=(G.mean(), se(G)), tpr=(TG.mean(), se(TG)), regret=(R.mean(), se(R)),
               regret_vec=R.tolist())

def paired_t(a, b):
    a = np.array(a); b = np.array(b); d = a - b
    md = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(len(d))
    t = md / se if se > 0 else 0.0
    rev = int((np.sign(d) != np.sign(md)).sum())
    return round(md, 3), round(se, 3), round(t, 2), rev

out = {}
print("=== Q1: NON-LINEAR estimator (gradient-boosted trees) ===")
for pol in ["greedy", "explore_targeted", "parity", "eqopp"]:
    r = agg(pol, estimator="gbt"); out["gbt_" + pol] = r
    print("gbt %-16s gap=%.3f+/-%.3f tprgap=%.3f regret=%.2f+/-%.2f" %
          (pol, r['gap'][0], r['gap'][1], r['tpr'][0], r['regret'][0], r['regret'][1]))
# paired greedy vs parity under GBT
md, se, t, rev = paired_t(out["gbt_greedy"]["regret_vec"], out["gbt_parity"]["regret_vec"])
out["gbt_paired"] = [md, se, t, rev]
print("gbt greedy-parity regret diff=%.3f+/-%.3f t=%.2f reversals=%d/8" % (md, se, t, rev))

print("=== Q2: DEPLOYABLE eq-opp (predicted-prob split) vs ORACLE eq-opp (logreg) ===")
for pol in ["eqopp", "eqopp_deploy"]:
    r = agg(pol); out["logreg_" + pol] = r
    print("%-14s tprgap=%.3f+/-%.3f gap=%.3f regret=%.2f+/-%.2f" %
          (pol, r['tpr'][0], r['tpr'][1], r['gap'][0], r['regret'][0], r['regret'][1]))

print("=== Q3: budget-wise paired greedy-vs-parity (logreg) ===")
for b in [0.3, 0.5, 0.7]:
    rg = agg("greedy", budget=b); rp = agg("parity", budget=b)
    out["greedy_b%.1f" % b] = rg; out["parity_b%.1f" % b] = rp
    md, se, t, rev = paired_t(rg["regret_vec"], rp["regret_vec"])
    out["paired_b%.1f" % b] = [md, se, t, rev]
    print("budget=%.1f greedy regret=%.2f parity regret=%.2f | diff=%.3f+/-%.3f t=%.2f rev=%d/8 gap_par=%.3f" %
          (b, rg['regret'][0], rp['regret'][0], md, se, t, rev, rp['gap'][0]))

json.dump(out, open("rev.json", "w"), indent=2, default=float)
print("WROTE rev.json")
