import numpy as np, time
from experiment import gen_data, gen_beliefs, rater_score, evaluate, sigmoid, RATER_Q, train_policy
from experiment_v2 import train_policy_corr

d=10; n_train=4000; n_eval=2000; n_rounds=60; lr=0.5

def run_alpha(alpha, n_raters, seeds=8, seed_offset=5000):
    accs=[]; sycos=[]
    for seed in range(seeds):
        rng = np.random.default_rng(seed_offset*seed + int(alpha*1000) + n_raters)
        w_true = rng.normal(size=d); w_true/=np.linalg.norm(w_true)
        X_tr,y_tr = gen_data(n_train,d,w_true,rng)
        b_tr = gen_beliefs(y_tr, RATER_Q, rng)
        X_ev,y_ev = gen_data(n_eval,d,w_true,rng)
        b_ev = gen_beliefs(y_ev, RATER_Q, rng)
        w,_ = train_policy(X_tr,y_tr,b_tr,alpha,n_rounds,4,lr,n_raters=n_raters,rng=rng,X_eval=X_ev,y_eval=y_ev,b_eval=b_ev,eval_every=1000)
        acc,syco = evaluate(w,X_ev,y_ev,b_ev)
        accs.append(acc); sycos.append(syco)
    return np.mean(accs), np.mean(sycos)

t0=time.time()
for alpha in [0.5, 0.75]:
    a,s = run_alpha(alpha, 1)
    print(f'alpha={alpha} acc={a:.3f} syco={s:.3f}  t={time.time()-t0:.1f}s')

def run_corr(rho, nr, seeds=8):
    accs=[]; sycos=[]
    for seed in range(seeds):
        rng = np.random.default_rng(7000*seed + int(rho*100) + nr)
        w_true = rng.normal(size=d); w_true/=np.linalg.norm(w_true)
        X_tr,y_tr = gen_data(n_train,d,w_true,rng)
        X_ev,y_ev = gen_data(n_eval,d,w_true,rng)
        b_ev = gen_beliefs(y_ev, RATER_Q, rng)
        acc,syco = train_policy_corr(X_tr,y_tr,0.75,n_rounds,4,lr,n_raters=nr,rho=rho,rng=rng,X_eval=X_ev,y_eval=y_ev,b_eval_indep=b_ev)
        accs.append(acc); sycos.append(syco)
    return np.mean(accs), np.mean(sycos)

for rho in [0.0, 0.5]:
    for nr in [1,9]:
        a,s = run_corr(rho, nr)
        print(f'rho={rho} nr={nr} acc={a:.3f} syco={s:.3f}  t={time.time()-t0:.1f}s')
print('total', time.time()-t0)
