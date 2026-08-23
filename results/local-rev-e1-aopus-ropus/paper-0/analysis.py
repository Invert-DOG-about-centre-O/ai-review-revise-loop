"""Extended analysis for revision: per-seed stats, paired significance test,
and a GROUP-TARGETED exploration policy (answers reviewer Q3)."""
import numpy as np, json
from sim import make_cohort, fit_logreg, predict

def run(policy, rounds=25, cohort=400, budget=0.5, eps=0.15, seed=0):
    rng = np.random.default_rng(seed)
    Xs, gs, qs, rs = make_cohort(600, rng)
    seed_mask = (gs==0) | (rng.random(len(gs)) < 0.15)
    Xhist=[Xs[seed_mask]]; yhist=[rs[seed_mask]]
    H={"appr_gap":[], "tpr_gap":[], "util":[], "oracle_util":[], "obs_B_frac":[]}
    theta_static=None
    for t in range(rounds):
        if policy=="static":
            if theta_static is None:
                theta_static=fit_logreg(np.vstack(Xhist), np.concatenate(yhist))
            theta=theta_static
        else:
            theta=fit_logreg(np.vstack(Xhist), np.concatenate(yhist))
        X,g,q,repay=make_cohort(cohort, rng)
        score=predict(theta,X)
        k=int(budget*cohort)
        approve=np.zeros(cohort,bool)
        if policy=="parity":
            for grp in (0,1):
                idx=np.where(g==grp)[0]
                kk=int(budget*len(idx))
                approve[idx[np.argsort(-score[idx])[:kk]]]=True
        elif policy=="explore":
            n_rand=int(eps*k); n_greedy=k-n_rand
            order=np.argsort(-score)
            approve[order[:n_greedy]]=True
            rest=order[n_greedy:]
            pick=rng.choice(rest,size=min(n_rand,len(rest)),replace=False)
            approve[pick]=True
        elif policy=="explore_targeted":
            # spend the exploration budget ONLY on the under-covered group B,
            # drawn from B's rejected pool (informed / group-targeted exploration)
            n_rand=int(eps*k); n_greedy=k-n_rand
            order=np.argsort(-score)
            approve[order[:n_greedy]]=True
            rest=order[n_greedy:]
            restB=rest[g[rest]==1]
            pick=rng.choice(restB,size=min(n_rand,len(restB)),replace=False)
            approve[pick]=True
        else:  # greedy
            approve[np.argsort(-score)[:k]]=True
        aA=approve[g==0].mean(); aB=approve[g==1].mean()
        def tpr(grp):
            m=(g==grp)&(repay==1); return approve[m].mean() if m.sum()>0 else 0.0
        util=repay[approve].sum()-0.6*(1-repay[approve]).sum()
        oracle=np.zeros(cohort,bool); oracle[np.argsort(-(1.2*q))[:k]]=True
        # use true repay prob ordering (monotone in q) as oracle
        from sim import sigmoid
        oracle=np.zeros(cohort,bool); oracle[np.argsort(-sigmoid(1.2*q))[:k]]=True
        outil=repay[oracle].sum()-0.6*(1-repay[oracle]).sum()
        H["appr_gap"].append(aA-aB); H["tpr_gap"].append(tpr(0)-tpr(1))
        H["util"].append(util); H["oracle_util"].append(outil)
        H["obs_B_frac"].append((g[approve]==1).mean() if approve.sum()>0 else 0)
        Xhist.append(X[approve]); yhist.append(repay[approve])
    return {k:np.array(v) for k,v in H.items()}

N=8
pols=["static","greedy","explore","explore_targeted","parity"]
# per-seed scalar summaries
perseed={p:{"mean_appr_gap":[], "final_appr_gap":[], "final_tpr_gap":[],
            "regret":[], "final_B":[]} for p in pols}
for p in pols:
    for s in range(N):
        h=run(p,seed=s)
        perseed[p]["mean_appr_gap"].append(float(h["appr_gap"].mean()))
        perseed[p]["final_appr_gap"].append(float(h["appr_gap"][-1]))
        perseed[p]["final_tpr_gap"].append(float(h["tpr_gap"][-1]))
        perseed[p]["regret"].append(float(100*(1-h["util"].sum()/h["oracle_util"].sum())))
        perseed[p]["final_B"].append(float(h["obs_B_frac"][-1]))

def ci(x):
    x=np.array(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)); return m,se
print("POLICY            mean_appr_gap        regret%            final_B")
for p in pols:
    m1,s1=ci(perseed[p]["mean_appr_gap"])
    m2,s2=ci(perseed[p]["regret"])
    m3,s3=ci(perseed[p]["final_B"])
    print(f"{p:17s} {m1:.3f}+/-{s1:.3f}   {m2:5.2f}+/-{s2:.2f}   {m3:.3f}+/-{s3:.3f}")

# paired test greedy vs parity regret
g=np.array(perseed["greedy"]["regret"]); pa=np.array(perseed["parity"]["regret"])
d=g-pa
print("\nPAIRED greedy-parity regret: mean=%.3f sd=%.3f, reversals(parity worse)=%d/%d"
      %(d.mean(), d.std(ddof=1), (d<0).sum(), N))
tstat=d.mean()/(d.std(ddof=1)/np.sqrt(N))
print("paired t=%.3f (df=%d)"%(tstat,N-1))
# targeted vs greedy: does it raise B coverage / close gap?
print("\nTARGETED exploration vs greedy:")
for key in ["final_appr_gap","final_tpr_gap","regret","final_B"]:
    tg=np.array(perseed["explore_targeted"][key]); gr=np.array(perseed["greedy"][key])
    print(f"  {key:16s} targeted={tg.mean():.3f}+/-{tg.std(ddof=1)/np.sqrt(N):.3f}  greedy={gr.mean():.3f}")
json.dump({p:{k:[round(x,4) for x in v] for k,v in perseed[p].items()} for p in pols},
          open("perseed.json","w"), indent=2)
print("WROTE perseed.json")
