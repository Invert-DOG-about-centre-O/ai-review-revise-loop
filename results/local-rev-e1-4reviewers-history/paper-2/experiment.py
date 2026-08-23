import numpy as np, json, time, itertools

D = 10
N_TRAIN_PER_ROUND = 256
ROUNDS = 60
K = 4
LR = 0.5
Q = 0.6
EPS_STD = 0.05
N_EVAL = 2000

def sigmoid(z): return 1/(1+np.exp(-z))

def make_problem(rng):
    w_star = rng.normal(size=D); w_star /= np.linalg.norm(w_star)
    return w_star

def gen_data(rng, w_star, n):
    x = rng.normal(size=(n, D))
    y = (rng.random(n) < sigmoid(x @ w_star)).astype(int)
    return x, y

def gen_beliefs(rng, y, n_raters, corr_rho=0.0):
    # returns array shape (n_raters, n) of beliefs
    n = len(y)
    if corr_rho == 0.0:
        correct = rng.random((n_raters, n)) < Q
        b = np.where(correct, y[None, :], 1 - y[None, :])
        return b
    # correlated case: with prob corr_rho, all raters share one common draw (shared misinformation source);
    # otherwise each rater draws independently as usual.
    shared_correct = rng.random(n) < Q
    shared_b = np.where(shared_correct, y, 1 - y)
    use_shared = rng.random((n_raters, n)) < corr_rho
    indiv_correct = rng.random((n_raters, n)) < Q
    indiv_b = np.where(indiv_correct, y[None, :], 1 - y[None, :])
    b = np.where(use_shared, shared_b[None, :], indiv_b)
    return b

def train(rng, w_star, alpha, n_raters, corr_rho=0.0, k=K, eps_std=EPS_STD, rounds=ROUNDS):
    w = np.zeros(D)
    for r in range(rounds):
        x, y = gen_data(rng, w_star, N_TRAIN_PER_ROUND)
        b = gen_beliefs(rng, y, n_raters, corr_rho)  # (n_raters, batch)
        p = sigmoid(x @ w)
        cand = (rng.random((k, N_TRAIN_PER_ROUND)) < p[None, :]).astype(int)  # (K, batch)
        agree = (cand[:, None, :] == b[None, :, :]).sum(axis=1)  # (K, batch) count of raters agreeing
        fact = (cand == y[None, :]).astype(float)  # (K, batch)
        eps = rng.normal(scale=eps_std, size=(k, N_TRAIN_PER_ROUND))
        score = alpha * agree + (1 - alpha) * fact * n_raters + eps
        best = np.argmax(score, axis=0)  # (batch,)
        target = cand[best, np.arange(N_TRAIN_PER_ROUND)]
        grad = x.T @ (p - target) / N_TRAIN_PER_ROUND
        w -= LR * grad
    return w

def train_oracle(rng, w_star, rounds=ROUNDS):
    w = np.zeros(D)
    for r in range(rounds):
        x, y = gen_data(rng, w_star, N_TRAIN_PER_ROUND)
        p = sigmoid(x @ w)
        grad = x.T @ (p - y) / N_TRAIN_PER_ROUND
        w -= LR * grad
    return w

def evaluate(rng, w_star, w):
    x, y = gen_data(rng, w_star, N_EVAL)
    b = gen_beliefs(rng, y, 1, 0.0)[0]
    pred = (sigmoid(x @ w) > 0.5).astype(int)
    acc = np.mean(pred == y)
    wrong_mask = b != y
    syco = np.mean(pred[wrong_mask] == b[wrong_mask]) if wrong_mask.sum() > 0 else np.nan
    return acc, syco

def run_condition(seed, alpha, n_raters, corr_rho=0.0, k=K, eps_std=EPS_STD):
    rng = np.random.default_rng(seed)
    w_star = make_problem(rng)
    w = train(rng, w_star, alpha, n_raters, corr_rho, k, eps_std)
    return evaluate(rng, w_star, w)

def run_oracle(seed):
    rng = np.random.default_rng(seed + 999983)
    w_star = make_problem(rng)
    w = train_oracle(rng, w_star)
    return evaluate(rng, w_star, w)

SEEDS = list(range(30))

def sweep_alpha():
    out = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        accs, sycos = [], []
        for s in SEEDS:
            a, sy = run_condition(seed=1000*s + int(alpha*1000), alpha=alpha, n_raters=1)
            accs.append(a); sycos.append(sy)
        out[alpha] = {"acc": accs, "syco": sycos}
    return out

def sweep_raters():
    out = {}
    for nr in [1, 3, 5, 9]:
        accs, sycos = [], []
        for s in SEEDS:
            a, sy = run_condition(seed=2000*s + nr, alpha=0.75, n_raters=nr)
            accs.append(a); sycos.append(sy)
        out[nr] = {"acc": accs, "syco": sycos}
    return out

def sweep_correlated():
    # fixed alpha=0.75, n_raters=9 (the mitigation setting); sweep corr_rho: prob raters share
    # a common misinformation source instead of drawing beliefs independently.
    out = {}
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        accs, sycos = [], []
        for s in SEEDS:
            a, sy = run_condition(seed=3000*s + int(rho*1000), alpha=0.75, n_raters=9, corr_rho=rho)
            accs.append(a); sycos.append(sy)
        out[rho] = {"acc": accs, "syco": sycos}
    return out

def sweep_hyperparam():
    # sensitivity of the alpha=0.5 vs alpha=0.75 transition to K and eps_std
    out = {}
    for k in [2, 4, 8]:
        for eps_std in [0.02, 0.05, 0.15]:
            for alpha in [0.5, 0.75]:
                accs = []
                for s in SEEDS[:15]:
                    a, sy = run_condition(seed=4000*s + int(alpha*1000)+k*10007, alpha=alpha, n_raters=1, k=k, eps_std=eps_std)
                    accs.append(a)
                out[f"K{k}_eps{eps_std}_a{alpha}"] = accs
    return out

def oracle_runs():
    return [run_oracle(s) for s in SEEDS]

def welch_t(a, b):
    a = np.array(a); b = np.array(b)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va/na + vb/nb)
    t = (a.mean() - b.mean()) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    return float(t), float(df)

def p_from_t(t, df):
    from math import gamma, sqrt, pi
    tt = abs(t)
    xs = np.linspace(tt, tt+60, 200000)
    coef = gamma((df+1)/2) / (sqrt(df*pi)*gamma(df/2))
    pdf = coef * (1 + xs**2/df) ** (-(df+1)/2)
    tail = np.trapezoid(pdf, xs)
    return float(2*tail)

if __name__ == "__main__":
    t0 = time.time()
    results = {}
    results["alpha_sweep"] = sweep_alpha()
    results["rater_sweep"] = sweep_raters()
    results["oracle"] = oracle_runs()
    results["correlated_sweep"] = sweep_correlated()
    results["hyperparam_sweep"] = sweep_hyperparam()

    sig = {}
    a05 = results["alpha_sweep"][0.5]["acc"]
    a075 = results["alpha_sweep"][0.75]["acc"]
    t, df = welch_t(a05, a075)
    sig["alpha_0.5_vs_0.75_acc"] = {"t": t, "df": df, "p": p_from_t(t, df),
                                     "mean_diff": float(np.mean(a05)-np.mean(a075))}

    r1 = results["rater_sweep"][1]["acc"]
    r3 = results["rater_sweep"][3]["acc"]
    r9 = results["rater_sweep"][9]["acc"]
    oracle_acc = [x[0] for x in results["oracle"]]
    t, df = welch_t(r1, r3)
    sig["nraters_1_vs_3_acc"] = {"t": t, "df": df, "p": p_from_t(t, df),
                                  "mean_diff": float(np.mean(r1)-np.mean(r3))}
    t, df = welch_t(r9, oracle_acc)
    sig["nraters_9_vs_oracle_acc"] = {"t": t, "df": df, "p": p_from_t(t, df),
                                       "mean_diff": float(np.mean(r9)-np.mean(oracle_acc))}

    s05 = results["alpha_sweep"][0.5]["syco"]
    s075 = results["alpha_sweep"][0.75]["syco"]
    t, df = welch_t(s075, s05)
    sig["alpha_0.5_vs_0.75_syco"] = {"t": t, "df": df, "p": p_from_t(t, df),
                                      "mean_diff": float(np.mean(s075)-np.mean(s05))}

    for rho in [0.0, 0.5, 1.0]:
        rho_acc = results["correlated_sweep"][rho]["acc"]
        t, df = welch_t(rho_acc, r1)
        sig[f"rho_{rho}_vs_single_rater_acc"] = {"t": t, "df": df, "p": p_from_t(t, df),
                                                  "mean_diff": float(np.mean(rho_acc)-np.mean(r1))}

    results["significance"] = sig

    print("=== Correlated-rater sweep (alpha=0.75, n_raters=9) ===")
    for rho, d in results["correlated_sweep"].items():
        print(f"rho={rho:.2f}  acc={np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f}  syco={np.mean(d['syco']):.3f}±{np.std(d['syco']):.3f}")

    print("\n=== Significance tests ===")
    for k, v in sig.items():
        print(k, v)

    print("\n=== Hyperparam sensitivity (15 seeds each) ===")
    for k, v in results["hyperparam_sweep"].items():
        print(k, f"{np.mean(v):.3f}±{np.std(v):.3f}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    with open("results_v2.json", "w") as f:
        json.dump(results, f, indent=1)
