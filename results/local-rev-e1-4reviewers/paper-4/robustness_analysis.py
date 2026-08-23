"""
Robustness analysis addressing reviewer feedback on v1 of the sycophancy
simulation paper:
  1. Multi-seed variance (all reviewers): report mean +/- std over 10 seeds
     for every config in the original table.
  2. Fully crossed mitigation x outlier-stressor design (reviewers 1,3):
     naive / L2 penalty / pre-commit / median agg, each evaluated with
     and without the outlier-annotator stressor.
  3. Combined mitigation (reviewer 4): L2 penalty + pre-commit together.
  4. A fourth, literature-motivated mitigation (reviewer 4): "rubric
     reweighting" -- scale down the live-agreement component of the
     training reward, mirroring OpenAI's stated GPT-4o fix of
     rebalancing rubric-based reward against live user approval.
  5. Sensitivity sweep over lambda (L2 penalty strength) and precommit_q.

Fixes one bug found while extending the original code: annotator reward
noise inside train_policy was seeded only by iteration number (`seed_offset=it`),
so it did NOT vary across different `seed=` calls to train_policy -- runs with
different seeds still saw identical annotator noise draws, understating true
run-to-run variance. Here we combine the training seed into the annotator
noise seed so multi-seed runs are genuinely independent.
"""
import json
import time

import numpy as np

from sycophancy_sim import make_claims, policy_probs, annotator_rewards as _annotator_rewards_orig

N_TRAIN = 6000
N_TEST = 3000
train = make_claims(N_TRAIN, seed=1)
test = make_claims(N_TEST, seed=2)


def annotator_rewards(a, t, u, v, seed_offset, n_outlier_annotators=0, agree_scale=1.0):
    r = np.random.default_rng(1000 + seed_offset)
    K = 5
    n = len(a)
    agree_bias = r.uniform(0.5, 1.5, size=(K, n))
    if n_outlier_annotators > 0:
        agree_bias[:n_outlier_annotators] = r.uniform(2.5, 4.0, size=(n_outlier_annotators, n))
    correct = (a == t).astype(float)
    agree = (a == u).astype(float)
    acc_w = np.where(v == 1, 1.0, 0.0)
    agree_w = np.where(v == 1, 0.2, 1.0) * agree_bias * agree_scale
    noise = r.normal(0, 0.3, size=(K, n))
    rewards = acc_w * correct + agree_w * agree + noise
    return rewards


def train_policy(data, iters=400, batch=256, lr=0.08, lam_penalty=0.0,
                  aggregate="mean", precommit_q=0.0, seed=0, n_outliers=0,
                  agree_scale=1.0):
    r = np.random.default_rng(seed)
    n = len(data["t"])
    wc, g, h = 1.0, 0.0, 0.0
    baseline = 0.0
    for it in range(iters):
        idx = r.integers(0, n, size=batch)
        c, s, t, u, v = (data["c"][idx], data["s"][idx], data["t"][idx],
                         data["u"][idx], data["v"][idx])
        mask = np.ones(batch)
        if precommit_q > 0:
            hide = r.random(batch) < precommit_q
            mask[hide] = 0.0
        p = policy_probs((wc, g, h), c, s, mask)
        a = (r.random(batch) < p).astype(int)

        rews = annotator_rewards(a, t, u, v, seed_offset=seed * 100000 + it,
                                  n_outlier_annotators=n_outliers, agree_scale=agree_scale)
        if aggregate == "mean":
            reward = rews.mean(axis=0)
        elif aggregate == "median":
            reward = np.median(rews, axis=0)
        else:
            raise ValueError(aggregate)

        baseline = 0.95 * baseline + 0.05 * reward.mean()
        adv = reward - baseline

        dlogp_dlogit = (a - p)
        grad_wc = np.mean(adv * dlogp_dlogit * c)
        grad_g = np.mean(adv * dlogp_dlogit * s * mask)
        grad_h = np.mean(adv * dlogp_dlogit)

        wc += lr * grad_wc
        g += lr * grad_g - lr * lam_penalty * g
        h += lr * grad_h
    return (wc, g, h)


def evaluate(params, data):
    c, s, t, u, v = data["c"], data["s"], data["t"], data["u"], data["v"]
    p = policy_probs(params, c, s, mask=1.0)
    a = (p >= 0.5).astype(int)
    acc = float((a == t).mean())
    wrong_user = u != t
    wrong_user_v = wrong_user & (v == 1)
    wrong_user_a = wrong_user & (v == 0)
    syco_v = float((a[wrong_user_v] == u[wrong_user_v]).mean())
    syco_a = float((a[wrong_user_a] == u[wrong_user_a]).mean())
    rews = annotator_rewards(a, t, u, v, seed_offset=999999, n_outlier_annotators=0)
    approval = float(rews.mean())
    return dict(accuracy=acc, sycophancy_verifiable=syco_v, sycophancy_ambiguous=syco_a,
                approval_proxy=approval, wc=params[0], g=params[1], h=params[2])


def run_seeds(cfg, seeds):
    rows = []
    for sd in seeds:
        params = train_policy(train, seed=sd, **cfg)
        rows.append(evaluate(params, test))
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows])
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = list(range(10))
    report = {}

    # 1) Multi-seed version of the original 5 configs
    base_configs = {
        "naive_rlhf": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.0, n_outliers=0),
        "sycophancy_penalty": dict(aggregate="mean", lam_penalty=0.15, precommit_q=0.0, n_outliers=0),
        "precommit_answer": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.5, n_outliers=0),
        "naive_rlhf_with_outliers": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.0, n_outliers=1),
        "robust_median_agg": dict(aggregate="median", lam_penalty=0.0, precommit_q=0.0, n_outliers=1),
    }
    print("=== Multi-seed (10 seeds) base configs ===")
    for name, cfg in base_configs.items():
        res = run_seeds(cfg, SEEDS)
        report[name] = res
        print(f"[{name}] syco_ambig={res['sycophancy_ambiguous_mean']:.3f}+/-{res['sycophancy_ambiguous_std']:.3f}  "
              f"g={res['g_mean']:.3f}+/-{res['g_std']:.3f}  acc={res['accuracy_mean']:.3f}")

    # 2) Fully crossed mitigation x outlier design
    print("\n=== Crossed mitigation x outlier design (10 seeds each) ===")
    crossed = {}
    mitigations = {
        "naive": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.0),
        "l2_penalty": dict(aggregate="mean", lam_penalty=0.15, precommit_q=0.0),
        "precommit": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.5),
        "l2_plus_precommit": dict(aggregate="mean", lam_penalty=0.15, precommit_q=0.5),
        "median_agg": dict(aggregate="median", lam_penalty=0.0, precommit_q=0.0),
        "rubric_reweight": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.0, agree_scale=0.4),
    }
    for mname, mcfg in mitigations.items():
        for outlier_label, n_out in [("no_outlier", 0), ("outlier", 1)]:
            cfg = dict(mcfg)
            cfg["n_outliers"] = n_out
            key = f"{mname}__{outlier_label}"
            res = run_seeds(cfg, SEEDS)
            crossed[key] = res
            print(f"[{key}] syco_ambig={res['sycophancy_ambiguous_mean']:.3f}+/-{res['sycophancy_ambiguous_std']:.3f}  "
                  f"g={res['g_mean']:.3f}+/-{res['g_std']:.3f}  approval={res['approval_proxy_mean']:.3f}")
    report["crossed"] = crossed

    # 3) Sensitivity sweep
    print("\n=== Sensitivity: lambda (L2 penalty strength), 5 seeds each ===")
    lam_sweep = {}
    for lam in [0.05, 0.15, 0.3, 0.5]:
        res = run_seeds(dict(aggregate="mean", lam_penalty=lam, precommit_q=0.0, n_outliers=0), SEEDS[:5])
        lam_sweep[str(lam)] = res
        print(f"[lambda={lam}] syco_ambig={res['sycophancy_ambiguous_mean']:.3f}+/-{res['sycophancy_ambiguous_std']:.3f}  "
              f"g={res['g_mean']:.3f}  approval={res['approval_proxy_mean']:.3f}")
    report["lambda_sweep"] = lam_sweep

    print("\n=== Sensitivity: precommit_q, 5 seeds each ===")
    q_sweep = {}
    for q in [0.25, 0.5, 0.75, 1.0]:
        res = run_seeds(dict(aggregate="mean", lam_penalty=0.0, precommit_q=q, n_outliers=0), SEEDS[:5])
        q_sweep[str(q)] = res
        print(f"[precommit_q={q}] syco_ambig={res['sycophancy_ambiguous_mean']:.3f}+/-{res['sycophancy_ambiguous_std']:.3f}  "
              f"g={res['g_mean']:.3f}  approval={res['approval_proxy_mean']:.3f}")
    report["precommit_q_sweep"] = q_sweep

    # 4) Sensitivity: noise ratio and agree_weight ratio (does the qualitative
    #    verifiable/ambiguous gap survive weaker assumptions?)
    print("\n=== Sensitivity: reduced noise gap (sigma_v=1 vs 2, not 1 vs 4), 5 seeds ===")

    def make_claims_custom(n, sigma_ambig, seed):
        r = np.random.default_rng(seed)
        t = r.integers(0, 2, size=n)
        v = (r.random(n) < 0.5).astype(int)
        wrong = r.random(n) < 0.5
        u = np.where(wrong, 1 - t, t)
        true_logit = (2 * t - 1) * 3.0
        noise_std = np.where(v == 1, 1.0, sigma_ambig)
        c = true_logit + r.normal(0, 1, size=n) * noise_std
        s = (2 * u - 1).astype(float)
        return dict(t=t, v=v, u=u, c=c, s=s)

    noise_sweep = {}
    for sigma_ambig in [2.0, 3.0, 4.0]:
        tr = make_claims_custom(N_TRAIN, sigma_ambig, seed=1)
        te = make_claims_custom(N_TEST, sigma_ambig, seed=2)
        rows = []
        for sd in SEEDS[:5]:
            params = train_policy(tr, seed=sd, aggregate="mean", lam_penalty=0.0, precommit_q=0.0, n_outliers=0)
            rows.append(evaluate(params, te))
        syco_a = np.array([r["sycophancy_ambiguous"] for r in rows])
        syco_v = np.array([r["sycophancy_verifiable"] for r in rows])
        noise_sweep[str(sigma_ambig)] = dict(syco_ambig_mean=float(syco_a.mean()), syco_ambig_std=float(syco_a.std()),
                                              syco_verif_mean=float(syco_v.mean()), syco_verif_std=float(syco_v.std()))
        print(f"[sigma_ambig={sigma_ambig}] syco_verif={syco_v.mean():.3f}  syco_ambig={syco_a.mean():.3f}+/-{syco_a.std():.3f}")
    report["noise_sweep"] = noise_sweep

    with open("robustness_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTotal wall time: {time.time()-t0:.2f}s")
