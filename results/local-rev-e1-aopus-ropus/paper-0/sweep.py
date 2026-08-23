"""Revision experiments: (Q1) epsilon/budget sweep for exploration;
(Q2) equal-opportunity (equal-TPR) coverage constraint; (Q3) larger offset."""
import numpy as np, json
import sim
from sim import make_cohort, fit_logreg, predict, sigmoid

def run(policy, rounds=25, cohort=400, budget=0.5, eps=0.15, seed=0, offset=1.3):
    sim.OFFSET = offset
    rng = np.random.default_rng(seed)
    Xs, gs, qs, rs = make_cohort(600, rng)
    seed_mask = (gs == 0) | (rng.random(len(gs)) < 0.15)
    Xhist = [Xs[seed_mask]]; yhist = [rs[seed_mask]]
    H = {"appr_gap": [], "tpr_gap": [], "util": [], "oracle_util": []}
    for t in range(rounds):
        theta = fit_logreg(np.vstack(Xhist), np.concatenate(yhist))
        X, g, q, repay = make_cohort(cohort, rng)
        score = predict(theta, X); k = int(budget * cohort)
        approve = np.zeros(cohort, bool)
        if policy == "parity":
            for grp in (0, 1):
                idx = np.where(g == grp)[0]; kk = int(budget * len(idx))
                approve[idx[np.argsort(-score[idx])[:kk]]] = True
        elif policy == "eqopp":
            # equal-opportunity coverage: pick per-group split (kA,kB=k-kA)
            # minimizing |TPR_A-TPR_B|; within group approve top by score.
            idxA = np.where(g == 0)[0]; idxB = np.where(g == 1)[0]
            oA = idxA[np.argsort(-score[idxA])]; oB = idxB[np.argsort(-score[idxB])]
            qA = repay[oA]; qB = repay[oB]
            totA = max(repay[idxA].sum(), 1); totB = max(repay[idxB].sum(), 1)
            cumA = np.concatenate([[0], np.cumsum(qA)]); cumB = np.concatenate([[0], np.cumsum(qB)])
            best = None
            for kA in range(0, k + 1):
                kB = k - kA
                if kA > len(idxA) or kB > len(idxB): continue
                d = abs(cumA[kA] / totA - cumB[kB] / totB)
                if best is None or d < best[0]: best = (d, kA, kB)
            _, kA, kB = best
            approve[oA[:kA]] = True; approve[oB[:kB]] = True
        elif policy == "explore":
            n_rand = int(eps * k); order = np.argsort(-score)
            approve[order[:k - n_rand]] = True; rest = order[k - n_rand:]
            approve[rng.choice(rest, size=min(n_rand, len(rest)), replace=False)] = True
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

N = 8
def agg(policy, **kw):
    G = []; TG = []; R = []
    for s in range(N):
        h = run(policy, seed=s, **kw)
        G.append(h["appr_gap"].mean()); TG.append(h["tpr_gap"][-1])
        R.append(100 * (1 - h["util"].sum() / h["oracle_util"].sum()))
    G = np.array(G); TG = np.array(TG); R = np.array(R)
    se = lambda x: x.std(ddof=1) / np.sqrt(N)
    return dict(gap=(G.mean(), se(G)), tpr=(TG.mean(), se(TG)), regret=(R.mean(), se(R)))

out = {}
print("=== Q1: epsilon sweep (targeted exploration), budget=0.5 ===")
for e in [0.05, 0.10, 0.15, 0.30]:
    r = agg("explore_targeted", eps=e); out["targeted_eps%.2f" % e] = r
    print("eps=%.2f: gap=%.3f+/-%.3f regret=%.2f+/-%.2f tprgap=%.3f" %
          (e, r['gap'][0], r['gap'][1], r['regret'][0], r['regret'][1], r['tpr'][0]))
print("=== Q1b: budget sweep (targeted eps=0.15 vs parity) ===")
for b in [0.3, 0.5, 0.7]:
    rt = agg("explore_targeted", budget=b); rp = agg("parity", budget=b)
    out["targ_b%.1f" % b] = rt; out["parity_b%.1f" % b] = rp
    print("budget=%.1f: targeted regret=%.2f gap=%.3f | parity regret=%.2f gap=%.3f" %
          (b, rt['regret'][0], rt['gap'][0], rp['regret'][0], rp['gap'][0]))
print("=== Q2: equal-opportunity coverage vs demographic parity ===")
for pol in ["parity", "eqopp"]:
    r = agg(pol); out[pol] = r
    print("%s: gap=%.3f tprgap=%.3f+/-%.3f regret=%.2f+/-%.2f" %
          (pol, r['gap'][0], r['tpr'][0], r['tpr'][1], r['regret'][0], r['regret'][1]))
print("=== Q3: larger offset=2.0, rankings ===")
for pol in ["greedy", "explore_targeted", "parity", "eqopp"]:
    r = agg(pol, offset=2.0); out["%s_off2" % pol] = r
    print("%s off2.0: gap=%.3f tprgap=%.3f regret=%.2f" %
          (pol, r['gap'][0], r['tpr'][0], r['regret'][0]))
json.dump({k: {kk: [round(x, 4) for x in vv] for kk, vv in v.items()} for k, v in out.items()},
          open("sweep.json", "w"), indent=2)
print("WROTE sweep.json")
