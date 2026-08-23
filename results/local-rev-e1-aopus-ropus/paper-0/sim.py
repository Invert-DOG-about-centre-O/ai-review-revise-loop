"""
Selective-labels retraining feedback loop simulation.

Socio-technical setup: a lender scores applicants from two groups (A, B) that are
EQUALLY qualified in truth. Outcomes (repay) are observed ONLY for approved
applicants (selective labels). The model is retrained each round on accumulated
observed data. A small initial data bias toward group A is injected. We measure
whether the deployment->data->retraining loop amplifies a spurious disparity, and
compare mitigation policies.

Policies:
  greedy      : approve the top-scoring applicants (by predicted repay prob)
  explore     : eps-greedy -- reserve a fraction of the budget for random approvals
  parity      : enforce equal approval RATE across groups (group-conditional threshold)

CPU-only, numpy-only. Prints a JSON-ish results block and writes results.json.
"""
import numpy as np, json, sys

rng = np.random.default_rng(0)

def sigmoid(z): return 1.0/(1.0+np.exp(-z))

# ---- data generating process ----------------------------------------------
D = 4  # proxy feature dim (before group indicator is appended)
OFFSET = 1.3  # group-correlated nuisance shift in the proxy features
def make_cohort(n, rng):
    # latent qualification q ~ N(0,1), IDENTICAL distribution for both groups;
    # true repay depends ONLY on q, so the groups are equally creditworthy.
    g = rng.integers(0, 2, size=n)              # 0=A, 1=B
    q = rng.normal(0, 1.0, size=n)
    # Proxy features carry a group-correlated nuisance offset: for the SAME q,
    # group B's observable features look worse (proxy discrimination / redlining).
    off = np.where(g==1, -OFFSET, +OFFSET)
    Xp = q[:, None] + off[:, None] + rng.normal(0, 1.0, size=(n, D))
    # The model DOES observe group, so with enough per-group outcome data it can
    # learn to correct the offset -- but only for groups it actually approves.
    X = np.hstack([Xp, g[:, None].astype(float)])
    repay = (rng.random(n) < sigmoid(1.2*q)).astype(float)
    return X, g, q, repay

# ---- logistic regression (numpy, L2) ---------------------------------------
def fit_logreg(X, y, w=None, l2=1.0, iters=200, lr=0.3):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n,1))])
    theta = np.zeros(d+1)
    if w is None: w = np.ones(n)
    w = w/ w.mean()
    for _ in range(iters):
        p = sigmoid(Xb @ theta)
        grad = Xb.T @ (w*(p - y))/n + l2*np.r_[theta[:-1],0]/n
        theta -= lr*grad
    return theta

def predict(theta, X):
    return sigmoid(np.hstack([X, np.ones((len(X),1))]) @ theta)

# ---- simulation ------------------------------------------------------------
def run(policy, rounds=25, cohort=400, budget=0.5, eps=0.15, seed=0):
    rng = np.random.default_rng(seed)
    # biased seed data: mostly group-A approvals observed
    Xs, gs, qs, rs = make_cohort(600, rng)
    seed_mask = (gs==0) | (rng.random(len(gs)) < 0.15)  # A well-represented, B sparse
    Xhist = [Xs[seed_mask]]; yhist=[rs[seed_mask]]
    hist = {"appr_gap":[], "tpr_gap":[], "util":[], "oracle_util":[], "obs_B_frac":[]}
    theta_static = None
    for t in range(rounds):
        if policy=="static":
            # train ONCE on the biased seed, never retrain (isolates the loop)
            if theta_static is None:
                theta_static = fit_logreg(np.vstack(Xhist), np.concatenate(yhist))
            theta = theta_static
        else:
            Xtr = np.vstack(Xhist); ytr = np.concatenate(yhist)
            theta = fit_logreg(Xtr, ytr)
        X, g, q, repay = make_cohort(cohort, rng)
        score = predict(theta, X)
        k = int(budget*cohort)
        if policy=="parity":
            approve = np.zeros(cohort, bool)
            for grp in (0,1):
                idx = np.where(g==grp)[0]
                kk = int(budget*len(idx))
                top = idx[np.argsort(-score[idx])[:kk]]
                approve[top]=True
        elif policy=="explore":
            approve = np.zeros(cohort, bool)
            n_rand = int(eps*k)
            n_greedy = k - n_rand
            order = np.argsort(-score)
            approve[order[:n_greedy]] = True
            rest = order[n_greedy:]
            rand_pick = rng.choice(rest, size=min(n_rand,len(rest)), replace=False)
            approve[rand_pick] = True
        else:  # greedy
            approve = np.zeros(cohort, bool)
            approve[np.argsort(-score)[:k]] = True

        # metrics
        aA = approve[g==0].mean(); aB = approve[g==1].mean()
        # TPR: among truly-qualified (repay==1), fraction approved, per group
        def tpr(grp):
            m = (g==grp)&(repay==1)
            return approve[m].mean() if m.sum()>0 else 0.0
        util = repay[approve].sum() - 0.6*(1-repay[approve]).sum()  # gain-cost of approvals
        # oracle: approve k highest true-repay-prob applicants
        oracle = np.zeros(cohort,bool); oracle[np.argsort(-sigmoid(1.2*q))[:k]]=True
        outil = repay[oracle].sum() - 0.6*(1-repay[oracle]).sum()
        hist["appr_gap"].append(aA-aB)
        hist["tpr_gap"].append(tpr(0)-tpr(1))
        hist["util"].append(util); hist["oracle_util"].append(outil)
        hist["obs_B_frac"].append((g[approve]==1).mean() if approve.sum()>0 else 0)
        # feedback: only approved applicants' outcomes are observed & added
        Xhist.append(X[approve]); yhist.append(repay[approve])
    return {k:np.array(v) for k,v in hist.items()}

def summarize(name, h):
    return {
        "policy": name,
        "final_appr_gap": round(float(h["appr_gap"][-1]),4),
        "mean_appr_gap": round(float(h["appr_gap"].mean()),4),
        "final_tpr_gap": round(float(h["tpr_gap"][-1]),4),
        "mean_tpr_gap": round(float(h["tpr_gap"].mean()),4),
        "regret_pct": round(float(100*(1 - h["util"].sum()/h["oracle_util"].sum())),2),
        "final_obs_B_frac": round(float(h["obs_B_frac"][-1]),4),
    }

if __name__=="__main__":
    results={}; curves={}
    for pol in ["static","greedy","explore","parity"]:
        # average over seeds for stability
        agg=None; N=8
        for s in range(N):
            h=run(pol, seed=s)
            if agg is None: agg={k:v.copy() for k,v in h.items()}
            else:
                for k in agg: agg[k]+=h[k]
        for k in agg: agg[k]/=N
        results[pol]=summarize(pol,agg)
        curves[pol]={k:[round(float(x),4) for x in v] for k,v in agg.items()}
        print(json.dumps(results[pol]))
    json.dump({"summary":results,"curves":curves}, open("results.json","w"), indent=2)
    print("WROTE results.json")
