"""
Round-2 experiments answering the round-2 review:
  (Q1) Paired significance test (Wilcoxon signed-rank + paired t) on the
       headline gap reduction, to complement Cohen's d and the win count.
  (Q2) Automation-bias human: human errors are CORRELATED with the AI's
       confident-but-wrong advice (anchoring), eroding the "reliable human
       fallback" the group-conditional rule relies on. Sweep the bias strength.
  (Q3) GROUP-DEPENDENT proxy noise: the proxy attribute is noisier for the
       minority than the majority (the realistic failure mode), vs. the
       symmetric case already reported.
CPU-only, deterministic given the seed grid.
"""
import numpy as np
from scipy import stats
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


def simulate_human_autobias(y, rng, ai_pred, conf, acc=0.75, beta=0.0,
                            conf_cut=0.8):
    """Automation-bias human: on high-confidence AI cases (conf>=conf_cut),
    with prob beta the human ANCHORS to the AI's prediction (right or wrong);
    otherwise an independent human with accuracy `acc`. beta=0 -> independent."""
    base = simulate_human(y, rng, acc)
    anchor = (conf >= conf_cut) & (rng.random(len(y)) < beta)
    return np.where(anchor, ai_pred, base)


def route(ai, hum, use_ai):
    return np.where(use_ai, ai, hum)


def err_gap(y, pred, g):
    ea = np.mean(pred[g == 0] != y[g == 0])
    eb = np.mean(pred[g == 1] != y[g == 1])
    return abs(ea - eb)


def fit_ai(rng, wBvec):
    Xtr, ytr, gtr = make_population(4000, rng, wBvec)
    keep = np.ones(len(ytr), bool)
    minidx = np.where(gtr == 1)[0]
    drop = rng.choice(minidx, size=int(0.6 * len(minidx)), replace=False)
    keep[drop] = False
    clf = LogisticRegression(max_iter=1000).fit(Xtr[keep], ytr[keep])
    return clf


def tune_global(c_va, ai_va, hum_va, yva, taus):
    bt, bb = 0.5, -1
    for t in taus:
        a = np.mean(route(ai_va, hum_va, c_va >= t) == yva)
        if a > bb:
            bb, bt = a, t
    return bt


def tune_groupcond(c_va, ai_va, hum_va, yva, gva, taus):
    tg = {}
    for grp in (0, 1):
        m = gva == grp
        b2, t2 = -1, 0.5
        for t in taus:
            a = np.mean(route(ai_va[m], hum_va[m], c_va[m] >= t) == yva[m])
            if a > b2:
                b2, t2 = a, t
        tg[grp] = t2
    return tg


def one_run(seed, wBvec, human_acc=0.75, beta=0.0):
    """One full pipeline. If beta>0 the human is automation-biased."""
    rng = np.random.default_rng(seed)
    clf = fit_ai(rng, wBvec)
    Xva, yva, gva = make_population(3000, rng, wBvec)
    Xte, yte, gte = make_population(6000, rng, wBvec)

    def ai(X):
        pr = clf.predict_proba(X)
        return pr.argmax(1), pr.max(1)
    ai_va, c_va = ai(Xva)
    ai_te, c_te = ai(Xte)
    if beta == 0.0:
        hum_va = simulate_human(yva, rng, human_acc)
        hum_te = simulate_human(yte, rng, human_acc)
    else:
        hum_va = simulate_human_autobias(yva, rng, ai_va, c_va, human_acc, beta)
        hum_te = simulate_human_autobias(yte, rng, ai_te, c_te, human_acc, beta)
    taus = np.linspace(0.5, 0.99, 50)

    bt = tune_global(c_va, ai_va, hum_va, yva, taus)
    ug = c_te >= bt
    pg = route(ai_te, hum_te, ug)
    tg = tune_groupcond(c_va, ai_va, hum_va, yva, gva, taus)
    ugc = np.where(gte == 0, c_te >= tg[0], c_te >= tg[1])
    pgc = route(ai_te, hum_te, ugc)
    return dict(acc_g=np.mean(pg == yte), gap_g=err_gap(yte, pg, gte),
                acc_gc=np.mean(pgc == yte), gap_gc=err_gap(yte, pgc, gte))


def one_run_proxy(seed, wBvec, eps_A, eps_B, human_acc=0.75):
    """Group-DEPENDENT proxy noise: majority flipped w.p. eps_A, minority
    w.p. eps_B. Thresholds tuned on the noisy proxy; gap measured on true g."""
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

    def add_noise(g):
        flip = np.where(g == 0, rng.random(len(g)) < eps_A,
                        rng.random(len(g)) < eps_B)
        return np.where(flip, 1 - g, g)
    gva_p = add_noise(gva)
    gte_p = add_noise(gte)
    tgp = tune_groupcond(c_va, ai_va, hum_va, yva, gva_p, taus)
    ugp = np.where(gte_p == 0, c_te >= tgp[0], c_te >= tgp[1])
    pgp = route(ai_te, hum_te, ugp)
    return dict(acc=np.mean(pgp == yte), gap=err_gap(yte, pgp, gte))


def ci95(a):
    a = np.asarray(a)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


SEEDS = list(range(30))

print("=" * 64)
print("Q1: PAIRED SIGNIFICANCE of gap reduction (30 seeds, default drift)")
print("=" * 64)
runs = [one_run(s, wB0) for s in SEEDS]
gp = np.array([r["gap_g"] for r in runs])
gc = np.array([r["gap_gc"] for r in runs])
ag = np.array([r["acc_g"] for r in runs])
agc = np.array([r["acc_gc"] for r in runs])
tt = stats.ttest_rel(gp, gc)
wx = stats.wilcoxon(gp, gc)
tta = stats.ttest_rel(agc, ag)
print(f"  gap: paired t p={tt.pvalue:.2e}  Wilcoxon p={wx.pvalue:.2e}")
print(f"  acc: paired t p={tta.pvalue:.2e}")

print("\n" + "=" * 64)
print("Q2: AUTOMATION-BIAS HUMAN (human anchors to confident AI), 30 seeds")
print("=" * 64)
print(f"{'beta':>6} {'glob_acc':>9} {'glob_gap':>9} "
      f"{'gc_acc':>7} {'gc_gap':>7} {'d_gap':>7}")
for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = [one_run(s, wB0, beta=beta) for s in SEEDS]
    ga, _ = ci95([x["acc_g"] for x in r])
    gg, _ = ci95([x["gap_g"] for x in r])
    ca, _ = ci95([x["acc_gc"] for x in r])
    cg, _ = ci95([x["gap_gc"] for x in r])
    print(f"{beta:>6.2f} {ga:>9.3f} {gg:>9.3f} {ca:>7.3f} {cg:>7.3f} "
          f"{gg-cg:>7.3f}")

print("\n" + "=" * 64)
print("Q3: GROUP-DEPENDENT PROXY NOISE (majority eps=0.05, minority varies)")
print("=" * 64)
print(f"{'epsA':>6} {'epsB':>6} {'px_acc':>8} {'px_gap':>8}   "
      f"(global baseline: acc={np.mean(ag):.3f} gap={np.mean(gp):.3f})")
for epsA, epsB in [(0.05, 0.05), (0.05, 0.10), (0.05, 0.20),
                   (0.05, 0.30), (0.05, 0.40)]:
    r = [one_run_proxy(s, wB0, epsA, epsB) for s in SEEDS]
    pa, _ = ci95([x["acc"] for x in r])
    pg2, ph = ci95([x["gap"] for x in r])
    print(f"{epsA:>6.2f} {epsB:>6.2f} {pa:>8.3f} {pg2:>8.3f} +/-{ph:.3f}")
print("\nDONE")
