"""
Additional experiments answering round-1 review:
  (R1) Multi-seed variance / CIs / paired significance on the headline gaps.
  (R2) Concept-drift sensitivity: how the global-vs-group-cond gap scales with
       the angle between wA and wB (is the effect a knob artifact?).
  (R3) Noisy-proxy group-conditional rule: does the fix survive when g is only
       observed through a noisy proxy (the realistic, legally-constrained case)?
CPU-only, deterministic given the seed grid.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

D = 6
wA = np.array([1.4, -1.1, 0.9, -0.7, 0.5, -0.4])
wB0 = np.array([-1.2, 1.0, 0.9, 0.8, -0.5, 0.4])


def make_population(n, rng, wBvec, minority_frac=0.2):
    g = (rng.random(n) < minority_frac).astype(int)
    X = rng.normal(size=(n, D))
    w = np.where(g[:, None] == 0, wA, wBvec)
    logits = np.einsum("ij,ij->i", X, w)
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y, g


def simulate_human(y, rng, acc=0.75):
    correct = rng.random(len(y)) < acc
    return np.where(correct, y, 1 - y)


def route(ai, hum, use_ai):
    return np.where(use_ai, ai, hum)


def err_gap(y, pred, g):
    ea = np.mean(pred[g == 0] != y[g == 0])
    eb = np.mean(pred[g == 1] != y[g == 1])
    return abs(ea - eb)


def one_run(seed, wBvec, human_acc=0.75, proxy_noise=None):
    rng = np.random.default_rng(seed)
    Xtr, ytr, gtr = make_population(4000, rng, wBvec)
    Xva, yva, gva = make_population(3000, rng, wBvec)
    Xte, yte, gte = make_population(6000, rng, wBvec)
    keep = np.ones(len(ytr), bool)
    minidx = np.where(gtr == 1)[0]
    drop = rng.choice(minidx, size=int(0.6 * len(minidx)), replace=False)
    keep[drop] = False
    clf = LogisticRegression(max_iter=1000).fit(Xtr[keep], ytr[keep])

    def ai(X):
        pr = clf.predict_proba(X)
        return pr.argmax(1), pr.max(1)
    ai_va, c_va = ai(Xva)
    ai_te, c_te = ai(Xte)
    hum_va = simulate_human(yva, rng, human_acc)
    hum_te = simulate_human(yte, rng, human_acc)
    taus = np.linspace(0.5, 0.99, 50)

    def team_acc(mask, a, h, y):
        return np.mean(route(a, h, mask) == y)

    # global
    bt, bb = 0.5, -1
    for t in taus:
        a = team_acc(c_va >= t, ai_va, hum_va, yva)
        if a > bb:
            bb, bt = a, t
    ug = c_te >= bt
    pg = route(ai_te, hum_te, ug)
    acc_g = np.mean(pg == yte)
    gap_g = err_gap(yte, pg, gte)

    # group-conditional (true g at decision time)
    tg = {}
    for grp in (0, 1):
        m = gva == grp
        b2, t2 = -1, 0.5
        for t in taus:
            a = team_acc(c_va[m] >= t, ai_va[m], hum_va[m], yva[m])
            if a > b2:
                b2, t2 = a, t
        tg[grp] = t2
    ugc = np.where(gte == 0, c_te >= tg[0], c_te >= tg[1])
    pgc = route(ai_te, hum_te, ugc)
    acc_gc = np.mean(pgc == yte)
    gap_gc = err_gap(yte, pgc, gte)

    out = dict(acc_g=acc_g, gap_g=gap_g, acc_gc=acc_gc, gap_gc=gap_gc)

    if proxy_noise is not None:
        # decision-time proxy: g flipped w.p. proxy_noise. Thresholds are TUNED
        # on the same noisy proxy on validation (realistic pipeline).
        gva_p = np.where(rng.random(len(gva)) < proxy_noise, 1 - gva, gva)
        gte_p = np.where(rng.random(len(gte)) < proxy_noise, 1 - gte, gte)
        tgp = {}
        for grp in (0, 1):
            m = gva_p == grp
            b2, t2 = -1, 0.5
            for t in taus:
                a = team_acc(c_va[m] >= t, ai_va[m], hum_va[m], yva[m])
                if a > b2:
                    b2, t2 = a, t
            tgp[grp] = t2
        ugp = np.where(gte_p == 0, c_te >= tgp[0], c_te >= tgp[1])
        pgp = route(ai_te, hum_te, ugp)
        out["acc_px"] = np.mean(pgp == yte)
        out["gap_px"] = err_gap(yte, pgp, gte)  # measured on TRUE g
    return out


def ci95(a):
    a = np.asarray(a)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def cohend(x, y):
    x, y = np.asarray(x), np.asarray(y)
    d = x - y
    return d.mean() / d.std(ddof=1)


SEEDS = list(range(30))

print("=" * 64)
print("R1: MULTI-SEED (30 data+human seeds), default drift, h=0.75")
print("=" * 64)
runs = [one_run(s, wB0) for s in SEEDS]
for k, lbl in [("acc_g", "global   acc"), ("gap_g", "global   gap"),
               ("acc_gc", "grp-cond acc"), ("gap_gc", "grp-cond gap")]:
    m, h = ci95([r[k] for r in runs])
    print(f"  {lbl}: {m:.3f} +/- {h:.3f} (95% CI)")
gp = [r["gap_g"] for r in runs]
gc = [r["gap_gc"] for r in runs]
ag = [r["acc_g"] for r in runs]
agc = [r["acc_gc"] for r in runs]
dg, hg = ci95(np.array(gp) - np.array(gc))
da, ha = ci95(np.array(agc) - np.array(ag))
print(f"  paired gap reduction (global-grpcond): {dg:.3f} +/- {hg:.3f}, "
      f"Cohen d={cohend(gp, gc):.2f}")
print(f"  paired acc gain    (grpcond-global):   {da:.3f} +/- {ha:.3f}, "
      f"Cohen d={cohend(agc, ag):.2f}")
print(f"  grp-cond wins on BOTH acc and gap in "
      f"{sum((agc[i] >= ag[i]) and (gc[i] <= gp[i]) for i in range(len(SEEDS)))}"
      f"/{len(SEEDS)} seeds")

print("\n" + "=" * 64)
print("R2: CONCEPT-DRIFT SENSITIVITY (wB = (1-a)*wA + a*wB0), 10 seeds")
print("=" * 64)
print(f"{'alpha':>6} {'cos(wA,wB)':>10} {'AI_accB':>8} "
      f"{'glob_gap':>9} {'gc_gap':>7} {'d_gap':>7}")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    wBv = (1 - alpha) * wA + alpha * wB0
    cos = np.dot(wA, wBv) / (np.linalg.norm(wA) * np.linalg.norm(wBv))
    r = [one_run(s, wBv) for s in range(10)]
    gg = np.mean([x["gap_g"] for x in r])
    gcg = np.mean([x["gap_gc"] for x in r])
    # AI acc on B: quick standalone measure at alpha
    rng = np.random.default_rng(123)
    Xt, yt, gt = make_population(6000, rng, wBv)
    Xtr, ytr, gtr = make_population(4000, rng, wBv)
    keep = np.ones(len(ytr), bool)
    mi = np.where(gtr == 1)[0]
    keep[rng.choice(mi, int(0.6*len(mi)), replace=False)] = False
    clf = LogisticRegression(max_iter=1000).fit(Xtr[keep], ytr[keep])
    accB = np.mean(clf.predict(Xt[gt == 1]) == yt[gt == 1])
    print(f"{alpha:>6.2f} {cos:>10.2f} {accB:>8.3f} "
          f"{gg:>9.3f} {gcg:>7.3f} {gg-gcg:>7.3f}")

print("\n" + "=" * 64)
print("R3: NOISY-PROXY group-conditional rule (30 seeds, default drift)")
print("=" * 64)
print(f"{'noise':>6} {'px_acc':>8} {'px_gap':>8}   "
      f"(true g: acc={np.mean(agc):.3f} gap={np.mean(gc):.3f})")
for eps in [0.0, 0.05, 0.10, 0.20, 0.30]:
    r = [one_run(s, wB0, proxy_noise=eps) for s in SEEDS]
    pa, _ = ci95([x["acc_px"] for x in r])
    pgap, pgh = ci95([x["gap_px"] for x in r])
    print(f"{eps:>6.2f} {pa:>8.3f} {pgap:>8.3f} +/-{pgh:.3f}")
print("\nDONE")
