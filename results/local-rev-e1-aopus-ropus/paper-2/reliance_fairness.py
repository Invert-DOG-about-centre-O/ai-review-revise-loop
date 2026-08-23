"""
The distributional cost of confidence-based reliance in human-AI teams.

Simulation study. A real logistic-regression AI is trained on a two-group
synthetic population where the minority group B is under-represented, giving
the AI group-dependent accuracy AND miscalibration. A simulated human decision
maker has group-independent accuracy. We route each case to AI or human under
four reliance policies and measure team accuracy and the between-group error gap.

Policies / baselines:
  1. AI-only
  2. Human-only
  3. Global confidence threshold  (rely on AI iff max-prob >= tau*, tau* tuned
     on validation to MAXIMIZE team accuracy)  -- the standard appropriate-
     reliance rule.
  4. Group-conditional thresholds (separate tau per group, our fairness-aware
     variant).

Ablations: (a) sweep tau to trace the accuracy/fairness frontier; (b) sweep
human accuracy.

CPU-only, no downloads, fully reproducible.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

RNG = np.random.default_rng(0)


def make_population(n, rng, minority_frac=0.2):
    """Two groups sharing a feature space but with DIFFERENT label rules
    (concept drift). The AI, trained on a majority-A-dominated sample, learns
    A's rule and applies it to B confidently but often wrongly -- the classic
    representation-harm mechanism. Group membership is NOT a model feature."""
    g = (rng.random(n) < minority_frac).astype(int)  # 1 = minority group B
    d = 6
    wA = np.array([1.4, -1.1, 0.9, -0.7, 0.5, -0.4])
    wB = np.array([-1.2, 1.0, 0.9, 0.8, -0.5, 0.4])  # partial concept drift
    X = rng.normal(size=(n, d))
    w = np.where(g[:, None] == 0, wA, wB)
    logits = np.einsum("ij,ij->i", X, w)
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y, g


def simulate_human(y, rng, acc=0.75):
    """Human predictor with group-independent accuracy `acc`. Errors random so
    that the human is complementary to (uncorrelated with) the AI's errors."""
    correct = rng.random(len(y)) < acc
    yh = np.where(correct, y, 1 - y)
    return yh


def err_gap(y_true, y_pred, g):
    """|error_rate(group A) - error_rate(group B)|."""
    ea = np.mean(y_pred[g == 0] != y_true[g == 0])
    eb = np.mean(y_pred[g == 1] != y_true[g == 1])
    return ea, eb, abs(ea - eb)


def route(ai_pred, human_pred, use_ai):
    return np.where(use_ai, ai_pred, human_pred)


def main():
    rng = np.random.default_rng(42)
    # ---- data ----
    Xtr, ytr, gtr = make_population(4000, rng)
    Xva, yva, gva = make_population(3000, rng)
    Xte, yte, gte = make_population(6000, rng)

    # Under-sample minority in the TRAINING set only (representation harm at the
    # data-collection stage -> a socio-technical bias source).
    keep = np.ones(len(ytr), bool)
    minidx = np.where(gtr == 1)[0]
    drop = rng.choice(minidx, size=int(0.6 * len(minidx)), replace=False)
    keep[drop] = False
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr[keep], ytr[keep])

    def ai_outputs(X):
        proba = clf.predict_proba(X)
        pred = proba.argmax(1)
        conf = proba.max(1)
        return pred, conf

    ai_va, conf_va = ai_outputs(Xva)
    ai_te, conf_te = ai_outputs(Xte)

    HUMAN_ACC = 0.75
    hum_va = simulate_human(yva, rng, HUMAN_ACC)
    hum_te = simulate_human(yte, rng, HUMAN_ACC)

    print("=" * 68)
    print("SETUP")
    print("=" * 68)
    for name, y, ai, g in [("val", yva, ai_va, gva), ("test", yte, ai_te, gte)]:
        acc = np.mean(ai == y)
        ea, eb, gap = err_gap(y, ai, g)
        print(f"[{name}] AI acc={acc:.3f}  errA={ea:.3f} errB={eb:.3f} "
              f"gap={gap:.3f}  n_B={(g==1).sum()}")
    print(f"Human acc (both groups) target={HUMAN_ACC}")

    def report(name, pred, y, g):
        acc = np.mean(pred == y)
        ea, eb, gap = err_gap(y, pred, g)
        print(f"{name:<28} acc={acc:.3f}  errA={ea:.3f} errB={eb:.3f} "
              f"gap={gap:.3f}")
        return acc, gap

    print("\n" + "=" * 68)
    print("TEST-SET RESULTS BY POLICY")
    print("=" * 68)
    results = {}

    # 1. AI-only
    results["AI-only"] = report("1. AI-only", ai_te, yte, gte)
    # 2. Human-only
    results["Human-only"] = report("2. Human-only", hum_te, yte, gte)

    # helper: team accuracy given a use_ai mask on validation
    def team_acc(use_ai, ai, hum, y):
        return np.mean(route(ai, hum, use_ai) == y)

    # 3. Global threshold tuned on validation for max team accuracy
    taus = np.linspace(0.5, 0.99, 50)
    best_tau, best = 0.5, -1
    for t in taus:
        a = team_acc(conf_va >= t, ai_va, hum_va, yva)
        if a > best:
            best, best_tau = a, t
    use_te = conf_te >= best_tau
    results["Global-thresh"] = report(
        f"3. Global tau*={best_tau:.3f}", route(ai_te, hum_te, use_te), yte, gte)
    print(f"   (routed {use_te.mean()*100:.0f}% of cases to AI)")

    # 4. Group-conditional thresholds (fairness-aware): tune tau per group to
    #    maximize each group's team accuracy independently.
    tau_g = {}
    for grp in (0, 1):
        m = gva == grp
        bt, bb = 0.5, -1
        for t in taus:
            a = team_acc(conf_va[m] >= t, ai_va[m], hum_va[m], yva[m])
            if a > bb:
                bb, bt = a, t
        tau_g[grp] = bt
    use_te_gc = np.where(gte == 0, conf_te >= tau_g[0], conf_te >= tau_g[1])
    results["Group-cond"] = report(
        f"4. Group-cond tau={tau_g[0]:.2f}/{tau_g[1]:.2f}",
        route(ai_te, hum_te, use_te_gc), yte, gte)

    # ---- Ablation A: threshold sweep (accuracy/fairness frontier) ----
    print("\n" + "=" * 68)
    print("ABLATION A: global-threshold sweep on TEST (acc vs gap)")
    print("=" * 68)
    print(f"{'tau':>6} {'acc':>7} {'gap':>7} {'%AI':>6}")
    for t in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        u = conf_te >= t
        pred = route(ai_te, hum_te, u)
        acc = np.mean(pred == yte)
        _, _, gap = err_gap(yte, pred, gte)
        print(f"{t:>6.2f} {acc:>7.3f} {gap:>7.3f} {u.mean()*100:>5.0f}%")

    # ---- Ablation B: human accuracy sweep (fixed global tau*) ----
    print("\n" + "=" * 68)
    print("ABLATION B: human-accuracy sweep (global tau*, group-cond)")
    print("=" * 68)
    print(f"{'h_acc':>6} {'glob_acc':>9} {'glob_gap':>9} "
          f"{'gc_acc':>7} {'gc_gap':>7}")
    for hacc in [0.60, 0.70, 0.80, 0.90]:
        h_te = simulate_human(yte, rng, hacc)
        h_va = simulate_human(yva, rng, hacc)
        # retune global tau for this human
        bt, bb = 0.5, -1
        for t in taus:
            a = team_acc(conf_va >= t, ai_va, h_va, yva)
            if a > bb:
                bb, bt = a, t
        u = conf_te >= bt
        pg = route(ai_te, h_te, u)
        _, _, gapg = err_gap(yte, pg, gte)
        # group-cond
        tg = {}
        for grp in (0, 1):
            m = gva == grp
            b2, t2 = -1, 0.5
            for t in taus:
                a = team_acc(conf_va[m] >= t, ai_va[m], h_va[m], yva[m])
                if a > b2:
                    b2, t2 = a, t
            tg[grp] = t2
        ugc = np.where(gte == 0, conf_te >= tg[0], conf_te >= tg[1])
        pgc = route(ai_te, h_te, ugc)
        _, _, gapgc = err_gap(yte, pgc, gte)
        print(f"{hacc:>6.2f} {np.mean(pg==yte):>9.3f} {gapg:>9.3f} "
              f"{np.mean(pgc==yte):>7.3f} {gapgc:>7.3f}")

    # ---- Calibration gap by group (why global tau is unfair) ----
    print("\n" + "=" * 68)
    print("DIAGNOSTIC: AI reliability at high confidence, by group (test)")
    print("=" * 68)
    for grp, lbl in [(0, "A(maj)"), (1, "B(min)")]:
        m = (gte == grp) & (conf_te >= best_tau)
        if m.sum() > 0:
            rel = np.mean(ai_te[m] == yte[m])
            print(f"  group {lbl}: mean_conf={conf_te[m].mean():.3f} "
                  f"actual_acc_when_conf>=tau*={rel:.3f}  (n={m.sum()})")

    print("\nDONE")


if __name__ == "__main__":
    main()
