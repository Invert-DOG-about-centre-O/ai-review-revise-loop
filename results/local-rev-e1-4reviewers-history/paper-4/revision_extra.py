import numpy as np, json, sys
from scipy import stats
from sycophancy_sim import train, evaluate, gen_claims, sigmoid

# 1. Paired significance tests across the same 10 seeds
def paired_rows(variant_kwargs, seeds=range(10)):
    out = []
    for seed in seeds:
        w_c, g, h = train(seed, **variant_kwargs)
        out.append(evaluate(w_c, g, h, seed)["syco_a"])
    return np.array(out)

naive = paired_rows(dict())
l2 = paired_rows(dict(l2_lambda=0.15))
precommit = paired_rows(dict(precommit_q=0.5))
rubric = paired_rows(dict(acc_reweight=(1.0, 0.2)))
combo = paired_rows(dict(l2_lambda=0.15, precommit_q=0.5))

sig = {}
def report(name_a, a, name_b, b):
    t, p = stats.ttest_rel(a, b)
    sig[f"{name_a}_vs_{name_b}"] = dict(mean_diff=float(a.mean()-b.mean()), t=float(t), p=float(p))

report("precommit", precommit, "l2", l2)
report("l2", l2, "rubric", rubric)
report("combo", combo, "l2", l2)
report("combo", combo, "precommit", precommit)
report("l2", l2, "naive", naive)

# 2. q=1 degeneracy + combined L2 + full precommit
q1 = paired_rows(dict(precommit_q=1.0))
l2_plus_q1 = paired_rows(dict(l2_lambda=0.15, precommit_q=1.0))
sig["q1_mean"] = float(q1.mean())
sig["l2_plus_q1_mean"] = float(l2_plus_q1.mean())
report("l2_plus_q1", l2_plus_q1, "combo_q0.5", combo)

# 3. Environment-parameter sweep (sigma ratio, agree-weight ratio), naive training, 5 seeds
def env_sweep_sigma():
    out = {}
    for sigma_a in [1.5, 2.0, 4.0, 6.0]:
        rows_v, rows_a = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed + 10000)
            t, v, u, c = gen_claims(6000, seed)
            sigma = np.where(v == 1, 1.0, sigma_a)
            c = (2 * t - 1) * 3 + np.random.default_rng(seed).normal(0, 1, size=6000) * sigma
            s = 2 * u - 1
            w_c, g, h = 1.0, 0.0, 0.0
            baseline, beta, K = 0.0, 0.05, 5
            for it in range(400):
                idx = rng.integers(0, 6000, size=256)
                ti, vi, ui, ci, si = t[idx], v[idx], u[idx], c[idx], s[idx]
                z = w_c * ci + g * si + h
                p = sigmoid(z)
                a = (rng.random(256) < p).astype(float)
                acc_weight = np.where(vi == 1, 1.0, 0.0)
                agree_base = np.where(vi == 1, 0.2, 1.0)
                rewards = np.zeros(256)
                for k in range(K):
                    bias = rng.uniform(0.5, 1.5, size=256)
                    noise = rng.normal(0, 0.1, size=256)
                    rewards += acc_weight*(a==ti) + agree_base*bias*(a==ui) + noise
                reward = rewards/K
                baseline = (1-beta)*baseline + beta*reward.mean()
                adv = reward - baseline
                dz = a - p
                w_c += 0.08*np.mean(dz*adv*ci)
                g += 0.08*np.mean(dz*adv*si)
                h += 0.08*np.mean(dz*adv)
            ev = evaluate(w_c, g, h, seed)
            rows_v.append(ev["syco_v"]); rows_a.append(ev["syco_a"])
        out[str(sigma_a)] = dict(syco_v_mean=float(np.mean(rows_v)), syco_a_mean=float(np.mean(rows_a)),
                                   gap_ratio=float(np.mean(rows_a)/max(np.mean(rows_v),1e-6)))
    return out

def rubric_factor_sweep():
    out = {}
    for factor in [0.1, 0.2, 0.4, 0.6, 1.0]:
        rows = []
        for seed in range(8):
            w_c, g, h = train(seed, acc_reweight=(1.0, factor))
            rows.append(evaluate(w_c, g, h, seed)["syco_a"])
        out[str(factor)] = dict(syco_a_mean=float(np.mean(rows)), syco_a_std=float(np.std(rows)))
    return out

env_sigma = env_sweep_sigma()
rubric_sweep = rubric_factor_sweep()

result = dict(significance=sig, env_sigma_sweep=env_sigma, rubric_factor_sweep=rubric_sweep)
print(json.dumps(result, indent=2))
with open("revision_extra_results.json", "w") as f:
    json.dump(result, f, indent=2)
