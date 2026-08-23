import numpy as np, json, sys, time

def gen_claims(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, size=n)
    v = rng.integers(0, 2, size=n)
    flip = rng.random(n) < 0.5
    u = np.where(flip, 1 - t, t)
    sigma = np.where(v == 1, 1.0, 4.0)
    c = (2 * t - 1) * 3 + rng.normal(0, 1, size=n) * sigma
    return t, v, u, c

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(seed, iters=400, batch=256, lr=0.08, n_train=6000,
          l2_lambda=0.0, precommit_q=0.0, outliers=False, robust_median=False,
          acc_reweight=None):
    rng = np.random.default_rng(seed + 10000)
    t, v, u, c = gen_claims(n_train, seed)
    s = 2 * u - 1
    w_c, g, h = 1.0, 0.0, 0.0
    baseline = 0.0
    beta = 0.05
    K = 5
    for it in range(iters):
        idx = rng.integers(0, n_train, size=batch)
        ti, vi, ui, ci, si = t[idx], v[idx], u[idx], c[idx], s[idx]
        mask = np.ones(batch)
        if precommit_q > 0:
            hide = rng.random(batch) < precommit_q
            mask = np.where(hide, 0.0, 1.0)
        z = w_c * ci + g * (si * mask) + h
        p = sigmoid(z)
        a = (rng.random(batch) < p).astype(float)

        acc_weight = np.where(vi == 1, 1.0, 0.0)
        if acc_reweight is not None:
            acc_w, agree_w_scale = acc_reweight
            acc_weight = np.where(vi == 1, acc_w, 1.0 - acc_w)
        agree_base = np.where(vi == 1, 0.2, 1.0)
        if acc_reweight is not None:
            agree_base = agree_base * acc_reweight[1]

        rewards = np.zeros(batch)
        for k in range(K):
            bias = rng.uniform(0.5, 1.5, size=batch)
            if outliers and k == 0:
                bias = rng.uniform(2.5, 4.0, size=batch)
            noise = rng.normal(0, 0.1, size=batch)
            r_k = acc_weight * (a == ti).astype(float) + agree_base * bias * (a == ui).astype(float) + noise
            rewards += r_k
        if robust_median:
            all_r = np.zeros((K, batch))
            for k in range(K):
                bias = rng.uniform(0.5, 1.5, size=batch)
                if outliers and k == 0:
                    bias = rng.uniform(2.5, 4.0, size=batch)
                noise = rng.normal(0, 0.1, size=batch)
                all_r[k] = acc_weight * (a == ti).astype(float) + agree_base * bias * (a == ui).astype(float) + noise
            reward = np.median(all_r, axis=0)
        else:
            reward = rewards / K

        baseline = (1 - beta) * baseline + beta * reward.mean()
        adv = reward - baseline

        dz = (a - p)
        grad_w_c = np.mean(dz * adv * ci)
        grad_g = np.mean(dz * adv * (si * mask))
        grad_h = np.mean(dz * adv)

        w_c += lr * grad_w_c
        g += lr * grad_g
        h += lr * grad_h

        if l2_lambda > 0:
            g -= lr * l2_lambda * g

    return w_c, g, h

def evaluate(w_c, g, h, seed, n_test=3000):
    t, v, u, c = gen_claims(n_test, seed + 999999)
    s = 2 * u - 1
    z = w_c * c + g * s + h
    p = sigmoid(z)
    rng = np.random.default_rng(seed + 555)
    a = (rng.random(n_test) < p).astype(float)

    acc_overall = (a == t).mean()
    acc_v = (a[v == 1] == t[v == 1]).mean()
    acc_a = (a[v == 0] == t[v == 0]).mean()

    wrong_mask = (u != t)
    syco_overall = (a[wrong_mask] == u[wrong_mask]).mean()
    wv = wrong_mask & (v == 1)
    wa = wrong_mask & (v == 0)
    syco_v = (a[wv] == u[wv]).mean()
    syco_a = (a[wa] == u[wa]).mean()

    acc_weight = np.where(v == 1, 1.0, 0.0)
    agree_base = np.where(v == 1, 0.2, 1.0)
    approval = (acc_weight * (a == t) + agree_base * (a == u)).mean()

    return dict(acc_overall=acc_overall, acc_v=acc_v, acc_a=acc_a,
                syco_v=syco_v, syco_a=syco_a, approval=approval, g=g, w_c=w_c, h=h)

VARIANTS = {
    "naive": dict(),
    "l2_penalty": dict(l2_lambda=0.15),
    "precommit": dict(precommit_q=0.5),
    "l2_plus_precommit": dict(l2_lambda=0.15, precommit_q=0.5),
    "rubric_reweight": dict(acc_reweight=(1.0, 0.2)),
    "naive_outliers": dict(outliers=True),
    "robust_median_outliers": dict(outliers=True, robust_median=True),
    "l2_penalty_outliers": dict(l2_lambda=0.15, outliers=True),
    "precommit_outliers": dict(precommit_q=0.5, outliers=True),
    "robust_median_no_outliers": dict(robust_median=True),
}

def run_all(seeds):
    results = {}
    for name, kwargs in VARIANTS.items():
        rows = []
        for seed in seeds:
            w_c, g, h = train(seed, **kwargs)
            rows.append(evaluate(w_c, g, h, seed))
        agg = {}
        for key in rows[0]:
            vals = np.array([r[key] for r in rows])
            agg[key + "_mean"] = float(vals.mean())
            agg[key + "_std"] = float(vals.std())
        results[name] = agg
    return results

if __name__ == "__main__":
    t0 = time.time()
    seeds = list(range(10))
    results = run_all(seeds)
    print(json.dumps(results, indent=2))
    print("elapsed:", time.time() - t0, file=sys.stderr)
    with open(sys.argv[1] if len(sys.argv) > 1 else "results_multiseed.json", "w") as f:
        json.dump(results, f, indent=2)
