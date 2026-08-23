"""
Selective-labels feedback loops in algorithmic resource allocation.

Setting: a lender approves loans each round. Applicants have a scalar feature x
predictive of true repayment probability p(x). Two demographic groups A and B are
IDENTICAL in the ground-truth relationship p(x) and in the x distribution -- so any
group disparity that emerges is purely an artifact of the learning loop, not the world.

Censored feedback (selective labels): the repayment label is observed ONLY for
applicants who are approved. The model retrains each round on the accumulated
*observed* data. A small initial imbalance in the seed data (B under-represented)
can therefore be self-reinforcing: the model under-scores B -> approves fewer B ->
observes fewer B labels -> stays wrong about B.

We compare allocators:
  greedy      : approve if predicted p >= tau (myopic).
  eps_explore : greedy but with prob eps approve a random applicant (gather labels).
  ucb         : approve if (pred + beta * uncertainty) >= tau (optimism/exploration).
  dp_parity   : greedy but enforce equal approval RATE across groups each round.

Metrics tracked per round:
  approval-rate gap  |rate_A - rate_B|
  TPR (equal-opp) gap : among truly-creditworthy (p>=tau) applicants, P(approved)
  qualified-approval  : fraction of truly-creditworthy applicants who got approved
  utility             : mean net return (repaid loans - defaults), and cumulative.
  calibration error   : |mean predicted - mean true| by group.

Model: logistic regression fit by gradient descent (numpy only), with an
uncertainty proxy from per-group observed sample counts (for UCB).
"""
import numpy as np

RNG_SEED = 0

# ---- world ----
def true_p(x):
    # identical ground-truth relationship for both groups: logistic in x
    return 1.0 / (1.0 + np.exp(-(1.6 * x)))

def sample_applicants(rng, n):
    # x ~ N(0,1), identical across groups; group assigned 50/50
    x = rng.normal(0, 1, size=n)
    g = rng.integers(0, 2, size=n)  # 0=A, 1=B
    p = true_p(x)
    repay = (rng.random(n) < p).astype(float)  # latent outcome if approved
    return x, g, p, repay

def features(x, g):
    # GROUP-AWARE design: separate intercept & slope per group via interactions.
    # The model CAN represent the (identical) truth, but must estimate each group's
    # parameters from that group's observed labels -> vulnerable to censoring.
    g = np.asarray(g, float)
    x = np.asarray(x, float)
    return np.column_stack([np.ones_like(x), x, g, x * g])

# ---- model: group-aware logistic regression ----
def fit_logistic(X, y, w_prior=None, iters=300, lr=0.3, l2=1.0):
    n, d = X.shape
    w = np.zeros(d) if w_prior is None else w_prior.copy()
    for _ in range(iters):
        z = X @ w
        pr = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (pr - y) / n + l2 * w / max(n, 1)
        w -= lr * grad
    return w

def predict(w, x, g):
    X = features(x, g)
    return 1.0 / (1.0 + np.exp(-(X @ w)))

# ---- simulation ----
def run(allocator, seed=0, rounds=40, batch=400, tau=0.5,
        seed_n=200, seed_B_frac=0.15, eps=0.1, beta=0.6, l2=1.0, feat_noise=1.5,
        seed_thresh=-0.3):
    """seed_B_frac: fraction of the seed labelled set that is group B (imbalanced).
    feat_noise: std of Gaussian noise added to the feature the MODEL observes
    (truth still depends on the clean x). High noise -> the individual signal is
    weak, so the model leans on group identity -> censored feedback can trap B.
    Low noise -> the individual feature lets the model rescue high-x B applicants."""
    rng = np.random.default_rng(seed)

    def observe(x):  # noisy feature the lender actually sees
        return x + rng.normal(0, feat_noise, size=np.shape(x))

    # ---- biased seed data: B under-represented AND region-restricted ----
    # A's seed spans all x; B's few seed labels come only from a narrow low-x slice
    # (a plausible historical artifact: B was only ever approved when very safe).
    xs, gs, ps, rep = sample_applicants(rng, 8000)
    xm_all = observe(xs)
    idxA = np.where(gs == 0)[0]
    idxB = np.where((gs == 1) & (xs < seed_thresh))[0]  # restricted region for B seed
    nB = int(seed_n * seed_B_frac)
    nA = seed_n - nB
    sel = np.concatenate([rng.choice(idxA, nA, replace=False),
                          rng.choice(idxB, nB, replace=False)])
    obs_x = list(xm_all[sel]); obs_y = list(rep[sel]); obs_g = list(gs[sel])
    nseen = {0: nA, 1: nB}  # observed label counts per group (uncertainty proxy)

    logs = []
    cum_util = 0.0
    for t in range(rounds):
        X = features(np.array(obs_x), np.array(obs_g))
        w = fit_logistic(X, np.array(obs_y), l2=l2)

        x, g, p, repay = sample_applicants(rng, batch)
        xm = observe(x)                 # noisy feature the model sees
        pred = predict(w, xm, g)

        # uncertainty proxy: larger for the group with fewer observed labels
        unc = np.array([1.0 / np.sqrt(nseen[gi] + 1) for gi in g])

        if allocator == "greedy":
            approve = pred >= tau
        elif allocator == "eps_explore":
            approve = pred >= tau
            flip = rng.random(len(x)) < eps
            approve = np.where(flip, rng.random(len(x)) < 0.5, approve)
        elif allocator == "ucb":
            approve = (pred + beta * unc) >= tau
        elif allocator == "dp_parity":
            # target approval rate = overall greedy rate; apply same rate per group
            rate = float(np.mean(pred >= tau))
            approve = np.zeros(len(x), bool)
            for grp in (0, 1):
                gi = np.where(g == grp)[0]
                k = int(round(rate * len(gi)))
                if k > 0:
                    top = gi[np.argsort(-pred[gi])[:k]]
                    approve[top] = True
        else:
            raise ValueError(allocator)

        # observe labels only for approved (store the noisy feature seen at decision)
        for i in np.where(approve)[0]:
            obs_x.append(xm[i]); obs_y.append(repay[i]); obs_g.append(g[i])
            nseen[g[i]] += 1

        # ---- metrics on this batch ----
        def rate_of(mask, sub):
            return float(np.mean(approve[sub])) if sub.sum() else 0.0
        A = g == 0; B = g == 1
        rA, rB = float(approve[A].mean()), float(approve[B].mean())
        qual = p >= tau  # truly creditworthy
        tprA = float(approve[A & qual].mean()) if (A & qual).sum() else 0.0
        tprB = float(approve[B & qual].mean()) if (B & qual).sum() else 0.0
        qual_appr = float(approve[qual].mean()) if qual.sum() else 0.0
        # utility: approved loan pays +1 if repaid, -1 if default
        util = float(np.sum(np.where(approve, 2 * repay - 1, 0))) / batch
        cum_util += util * batch
        # calibration: |mean pred - mean true| per group
        calA = abs(float(pred[A].mean() - p[A].mean()))
        calB = abs(float(pred[B].mean() - p[B].mean()))

        logs.append(dict(t=t, rate_gap=abs(rA - rB), rateA=rA, rateB=rB,
                         tpr_gap=abs(tprA - tprB), tprA=tprA, tprB=tprB,
                         qual_appr=qual_appr, util=util, cum_util=cum_util,
                         calA=calA, calB=calB))
    return logs

def summarize(name, all_logs):
    # average final-window (last 10 rounds) across seeds
    import numpy as np
    def fin(key):
        vals = [np.mean([lg[key] for lg in logs[-10:]]) for logs in all_logs]
        return float(np.mean(vals)), float(np.std(vals))
    rg, rgs = fin("rate_gap")
    tg, tgs = fin("tpr_gap")
    qa, qas = fin("qual_appr")
    ut, uts = fin("util")
    cb = np.mean([np.mean([lg["calB"] for lg in logs[-10:]]) for logs in all_logs])
    return dict(name=name, rate_gap=rg, rate_gap_sd=rgs, tpr_gap=tg, tpr_gap_sd=tgs,
                qual_appr=qa, qual_appr_sd=qas, util=ut, util_sd=uts, calB=float(cb))

if __name__ == "__main__":
    import sys, os
    _logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "run_output.txt"), "w")
    class _Tee:
        def write(self, s): sys.__stdout__.write(s); _logf.write(s)
        def flush(self): sys.__stdout__.flush(); _logf.flush()
    sys.stdout = _Tee()

    allocators = ["greedy", "eps_explore", "ucb", "dp_parity"]
    seeds = list(range(8))
    TRAP_NOISE = 2.0   # weak individual signal -> model leans on group -> trap regime
    results = {}
    trajectories = {}
    for a in allocators:
        all_logs = [run(a, seed=s, feat_noise=TRAP_NOISE) for s in seeds]
        results[a] = summarize(a, all_logs)
        # mean trajectory of rate_gap across seeds for plotting/inspection
        traj = np.mean([[lg["rate_gap"] for lg in logs] for logs in all_logs], axis=0)
        trajectories[a] = traj

    print(f"=== MAIN: trap regime (feat_noise={TRAP_NOISE}); final window (last 10 of 40), "
          f"mean +/- sd over 8 seeds ===")
    hdr = f"{'allocator':12s} {'rate_gap':>16s} {'tpr_gap':>16s} {'qual_appr':>16s} {'util':>14s} {'calibErr_B':>11s}"
    print(hdr)
    for a in allocators:
        r = results[a]
        print(f"{a:12s} {r['rate_gap']:.3f}+/-{r['rate_gap_sd']:.3f}   "
              f"{r['tpr_gap']:.3f}+/-{r['tpr_gap_sd']:.3f}   "
              f"{r['qual_appr']:.3f}+/-{r['qual_appr_sd']:.3f}   "
              f"{r['util']:.3f}+/-{r['util_sd']:.3f}   {r['calB']:.3f}")

    print(f"\n=== rate_gap trajectory (rounds 0,5,10,20,39), mean over seeds, "
          f"feat_noise={TRAP_NOISE} ===")
    for a in allocators:
        tr = trajectories[a]
        pts = [tr[i] for i in [0, 5, 10, 20, 39]]
        print(f"{a:12s} " + "  ".join(f"{v:.3f}" for v in pts))

    # KEY ablation: feature informativeness controls whether the trap occurs (greedy)
    print("\n=== Ablation: feature noise vs greedy outcomes (final window) ===")
    print(f"{'feat_noise':>10s} {'rate_gap':>10s} {'tpr_gap':>10s} {'qual_appr':>10s} {'calibErr_B':>11s} {'util':>8s}")
    for fn in [0.5, 1.0, 2.0, 3.0, 4.0]:
        all_logs = [run("greedy", seed=s, feat_noise=fn) for s in seeds]
        r = summarize(f"fn={fn}", all_logs)
        print(f"{fn:>10} {r['rate_gap']:>10.3f} {r['tpr_gap']:>10.3f} {r['qual_appr']:>10.3f} "
              f"{r['calB']:>11.3f} {r['util']:>8.3f}")

    # ablation: exploration rate eps (trap regime)
    print(f"\n=== Ablation: eps_explore vs eps, trap regime feat_noise={TRAP_NOISE} (final window) ===")
    for eps in [0.0, 0.05, 0.1, 0.2, 0.4]:
        all_logs = [run("eps_explore", seed=s, eps=eps, feat_noise=TRAP_NOISE) for s in seeds]
        r = summarize(f"eps={eps}", all_logs)
        print(f"eps={eps:<4} rate_gap={r['rate_gap']:.3f}  tpr_gap={r['tpr_gap']:.3f}  "
              f"qual_appr={r['qual_appr']:.3f}  util={r['util']:.3f}")

    # ablation: UCB exploration bonus beta (trap regime)
    print(f"\n=== Ablation: ucb vs beta, trap regime feat_noise={TRAP_NOISE} (final window) ===")
    for beta in [0.0, 0.3, 0.6, 1.2, 2.0]:
        all_logs = [run("ucb", seed=s, beta=beta, feat_noise=TRAP_NOISE) for s in seeds]
        r = summarize(f"beta={beta}", all_logs)
        print(f"beta={beta:<4} rate_gap={r['rate_gap']:.3f}  tpr_gap={r['tpr_gap']:.3f}  "
              f"qual_appr={r['qual_appr']:.3f}  util={r['util']:.3f}")

    # ablation: seed imbalance severity (greedy, trap regime)
    print(f"\n=== Ablation: greedy vs seed_B_frac, trap regime feat_noise={TRAP_NOISE} (final window) ===")
    for f in [0.5, 0.3, 0.15, 0.05]:
        all_logs = [run("greedy", seed=s, seed_B_frac=f, feat_noise=TRAP_NOISE) for s in seeds]
        r = summarize(f"f={f}", all_logs)
        print(f"seed_B_frac={f:<5} rate_gap={r['rate_gap']:.3f}  tpr_gap={r['tpr_gap']:.3f}  "
              f"calibErr_B={r['calB']:.3f}")
