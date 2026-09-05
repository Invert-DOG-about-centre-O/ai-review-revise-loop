import numpy as np, json, time, sys
from scipy import stats
from sycophancy_sim import train, evaluate, gen_claims, sigmoid

t0 = time.time()

# 1. Agree-weight ratio sweep: hold ambiguous agree_weight=1.0, vary verifiable
#    agree_weight (the "0.2" constant), since all four reviewers flagged this
#    as the one unswept constant that most directly encodes the mechanism.
# The acc_reweight hook in sycophancy_sim.py scales BOTH claim types' agree_base
# by the same factor (used for rubric reweighting). To vary the verifiable/
# ambiguous RATIO itself we need a small local trainer that sets agree_base
# directly per claim type, holding ambiguous fixed at 1.0.
def train_ratio(seed, verif_agree, iters=400, batch=256, lr=0.08, n_train=6000):
    rng = np.random.default_rng(seed + 10000)
    t, v, u, c = gen_claims(n_train, seed)
    s = 2 * u - 1
    w_c, g, h = 1.0, 0.0, 0.0
    baseline, beta, K = 0.0, 0.05, 5
    for it in range(iters):
        idx = rng.integers(0, n_train, size=batch)
        ti, vi, ui, ci, si = t[idx], v[idx], u[idx], c[idx], s[idx]
        z = w_c * ci + g * si + h
        p = sigmoid(z)
        a = (rng.random(batch) < p).astype(float)
        acc_weight = np.where(vi == 1, 1.0, 0.0)
        agree_base = np.where(vi == 1, verif_agree, 1.0)
        rewards = np.zeros(batch)
        for k in range(K):
            bias = rng.uniform(0.5, 1.5, size=batch)
            noise = rng.normal(0, 0.1, size=batch)
            rewards += acc_weight * (a == ti) + agree_base * bias * (a == ui) + noise
        reward = rewards / K
        baseline = (1 - beta) * baseline + beta * reward.mean()
        adv = reward - baseline
        dz = a - p
        w_c += lr * np.mean(dz * adv * ci)
        g += lr * np.mean(dz * adv * si)
        h += lr * np.mean(dz * adv)
    return w_c, g, h

def agree_ratio_sweep_real(ratios, seeds=range(8)):
    out = {}
    for r in ratios:
        rows_v, rows_a = [], []
        for seed in seeds:
            w_c, g, h = train_ratio(seed, r)
            ev = evaluate(w_c, g, h, seed)
            rows_v.append(ev["syco_v"]); rows_a.append(ev["syco_a"])
        out[str(r)] = dict(
            syco_v_mean=float(np.mean(rows_v)), syco_v_std=float(np.std(rows_v)),
            syco_a_mean=float(np.mean(rows_a)), syco_a_std=float(np.std(rows_a)),
            gap_ratio=float(np.mean(rows_a) / max(np.mean(rows_v), 1e-6)),
        )
    return out

ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
agree_sweep = agree_ratio_sweep_real(ratios)

# 2. Intermediate pre-commitment q sweep: is there a smooth transition into
#    the q=1 degeneracy, or a sharp discontinuity? (raised by 3/4 reviewers)
def q_sweep(qs, seeds=range(8)):
    out = {}
    for q in qs:
        g_vals, syco_vals = [], []
        for seed in seeds:
            w_c, g, h = train(seed, precommit_q=q)
            ev = evaluate(w_c, g, h, seed)
            g_vals.append(g); syco_vals.append(ev["syco_a"])
        out[str(q)] = dict(g_mean=float(np.mean(g_vals)), g_std=float(np.std(g_vals)),
                            syco_a_mean=float(np.mean(syco_vals)), syco_a_std=float(np.std(syco_vals)))
    return out

qs = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
q_results = q_sweep(qs)

# 3. Paired t-tests for the sigma sweep endpoints and the agree-ratio sweep
#    endpoints, so the "not a knife-edge" claims carry the same significance
#    rigor as the mitigation comparisons (raised by reviewer 1).
def paired_gap(ratio, seeds=range(8)):
    rows = []
    for seed in seeds:
        w_c, g, h = train_ratio(seed, ratio)
        rows.append(evaluate(w_c, g, h, seed)["syco_a"])
    return np.array(rows)

low = paired_gap(0.1)
high = paired_gap(0.8)
tstat, pval = stats.ttest_rel(high, low)
ratio_sig = dict(mean_diff=float(high.mean() - low.mean()), t=float(tstat), p=float(pval))

result = dict(agree_ratio_sweep=agree_sweep, q_sweep=q_results)
result["ratio_sig_0.8_vs_0.1"] = ratio_sig

print(json.dumps(result, indent=2))
print("elapsed:", time.time() - t0, file=sys.stderr)
with open("revision_extra2_results.json", "w") as f:
    json.dump(result, f, indent=2)
