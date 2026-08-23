"""Heteroskedastic-across-group noise robustness pass (review R3, Q3).

Same world/model/censoring as sim.py, but group B's observed feature is noisier
than group A's: sigma_B = het * sigma_A. Groups remain IDENTICAL in ground truth
p(x); only the *observability* of the individual signal differs by group. Tests
whether the informativeness->trap mechanism survives a group-dependent noise
mechanism (the only het case left open in v3 Limitations).
"""
import numpy as np
from sim import true_p, sample_applicants, features, fit_logistic, predict

def run_het(seed=0, rounds=40, batch=400, tau=0.5, seed_n=200, seed_B_frac=0.15,
            l2=1.0, base_noise=2.0, het=1.0, seed_thresh=-0.3):
    rng = np.random.default_rng(seed)
    def observe(x, g):  # group-dependent noise std
        s = base_noise * np.where(np.asarray(g) == 1, het, 1.0)
        return np.asarray(x, float) + rng.normal(0, 1, size=np.shape(x)) * s
    xs, gs, ps, rep = sample_applicants(rng, 8000)
    xm_all = observe(xs, gs)
    idxA = np.where(gs == 0)[0]
    idxB = np.where((gs == 1) & (xs < seed_thresh))[0]
    nB = int(seed_n * seed_B_frac); nA = seed_n - nB
    sel = np.concatenate([rng.choice(idxA, nA, replace=False),
                          rng.choice(idxB, nB, replace=False)])
    obs_x = list(xm_all[sel]); obs_y = list(rep[sel]); obs_g = list(gs[sel])
    logs = []
    for t in range(rounds):
        X = features(np.array(obs_x), np.array(obs_g))
        w = fit_logistic(X, np.array(obs_y), l2=l2)
        x, g, p, repay = sample_applicants(rng, batch)
        xm = observe(x, g)
        pred = predict(w, xm, g)
        approve = pred >= tau
        for i in np.where(approve)[0]:
            obs_x.append(xm[i]); obs_y.append(repay[i]); obs_g.append(g[i])
        A = g == 0; B = g == 1
        rA, rB = float(approve[A].mean()), float(approve[B].mean())
        qual = p >= tau
        tprA = float(approve[A & qual].mean()) if (A & qual).sum() else 0.0
        tprB = float(approve[B & qual].mean()) if (B & qual).sum() else 0.0
        calB = abs(float(pred[B].mean() - p[B].mean()))
        logs.append(dict(rate_gap=abs(rA - rB), tpr_gap=abs(tprA - tprB), calB=calB))
    return logs

def fw(logs, key): return float(np.mean([lg[key] for lg in logs[-10:]]))

if __name__ == "__main__":
    SEEDS = list(range(8))
    print("Heteroskedastic-across-group noise, greedy, base_noise=2.0, 8 seeds")
    print(f"{'het(sigB/sigA)':>14s} {'rate_gap':>10s} {'tpr_gap':>10s} {'calibErr_B':>11s}")
    for het in [1.0, 1.5, 2.0, 3.0]:
        rg = np.mean([fw(run_het(seed=s, het=het), "rate_gap") for s in SEEDS])
        tg = np.mean([fw(run_het(seed=s, het=het), "tpr_gap") for s in SEEDS])
        cb = np.mean([fw(run_het(seed=s, het=het), "calB") for s in SEEDS])
        print(f"{het:>14.1f} {rg:>10.3f} {tg:>10.3f} {cb:>11.3f}", flush=True)
