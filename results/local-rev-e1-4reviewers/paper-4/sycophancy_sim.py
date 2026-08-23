"""
Simulating sycophancy emergence under RLHF-style preference training.

We simulate a population of claims (some independently verifiable, some
hard-to-verify/ambiguous), a policy that must state true/false about each
claim, and a population of human annotators whose approval reward mixes
"is the answer correct" with "does the answer agree with the user's stated
stance". We train the policy via REINFORCE to maximize expected annotator
approval and track how much it learns to just agree with the user
(sycophancy), especially on ambiguous claims where accuracy can't be
verified by annotators. We then test three mitigations.

Everything here is synthetic / CPU-only and finishes in well under a minute.
"""
import json
import time

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_claims(n, p_verifiable=0.5, p_wrong_user=0.5, seed=0):
    r = np.random.default_rng(seed)
    t = r.integers(0, 2, size=n)                       # ground truth label
    v = (r.random(n) < p_verifiable).astype(int)        # 1 = verifiable, 0 = ambiguous
    wrong = r.random(n) < p_wrong_user
    u = np.where(wrong, 1 - t, t)                        # user's stated stance
    # policy's private, pre-pressure competence signal (logit toward truth)
    true_logit = (2 * t - 1) * 3.0
    noise_std = np.where(v == 1, 1.0, 4.0)               # much noisier when unverifiable
    c = true_logit + r.normal(0, 1, size=n) * noise_std
    s = (2 * u - 1).astype(float)                        # pressure direction: +1/-1
    return dict(t=t, v=v, u=u, c=c, s=s)


N_TRAIN = 6000
N_TEST = 3000
train = make_claims(N_TRAIN, seed=1)
test = make_claims(N_TEST, seed=2)

# ---------------------------------------------------------------------------
# Policy: answer_logit = wc*c + g*s(*mask) + h  ->  P(answer=='true')
# g is the "gives in to user pressure" weight we care about.
# ---------------------------------------------------------------------------

def policy_probs(params, c, s, mask=1.0):
    wc, g, h = params
    logit = wc * c + g * s * mask + h
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# Annotator population / reward model
# K annotators per item; each has an idiosyncratic "agreeableness" weight.
# Verifiable items: reward dominated by correctness, small agreement bonus.
# Ambiguous items: correctness can't be checked -> reward is agreement only.
# ---------------------------------------------------------------------------

K_ANNOTATORS = 5

def annotator_rewards(a, t, u, v, seed_offset, n_outlier_annotators=0):
    r = np.random.default_rng(1000 + seed_offset)
    n = len(a)
    agree_bias = r.uniform(0.5, 1.5, size=(K_ANNOTATORS, n))
    if n_outlier_annotators > 0:
        # occasional highly agreement-seeking annotators (troll/rubber-stamp raters)
        agree_bias[:n_outlier_annotators] = r.uniform(2.5, 4.0, size=(n_outlier_annotators, n))
    correct = (a == t).astype(float)
    agree = (a == u).astype(float)
    acc_w = np.where(v == 1, 1.0, 0.0)
    agree_w = np.where(v == 1, 0.2, 1.0) * agree_bias
    noise = r.normal(0, 0.3, size=(K_ANNOTATORS, n))
    rewards = acc_w * correct + agree_w * agree + noise
    return rewards  # shape (K, n)


# ---------------------------------------------------------------------------
# Training loop (REINFORCE with moving-average baseline)
# ---------------------------------------------------------------------------

def train_policy(data, iters=400, batch=256, lr=0.08, lam_penalty=0.0,
                  aggregate="mean", precommit_q=0.0, seed=0, n_outliers=0):
    r = np.random.default_rng(seed)
    n = len(data["t"])
    wc, g, h = 1.0, 0.0, 0.0
    baseline = 0.0
    history = []
    for it in range(iters):
        idx = r.integers(0, n, size=batch)
        c, s, t, u, v = (data["c"][idx], data["s"][idx], data["t"][idx],
                         data["u"][idx], data["v"][idx])
        mask = np.ones(batch)
        if precommit_q > 0:
            hide = r.random(batch) < precommit_q
            mask[hide] = 0.0  # policy must answer w/o seeing user stance this step
        p = policy_probs((wc, g, h), c, s, mask)
        a = (r.random(batch) < p).astype(int)

        rews = annotator_rewards(a, t, u, v, seed_offset=it, n_outlier_annotators=n_outliers)
        if aggregate == "mean":
            reward = rews.mean(axis=0)
        elif aggregate == "median":
            reward = np.median(rews, axis=0)
        else:
            raise ValueError(aggregate)

        baseline = 0.95 * baseline + 0.05 * reward.mean()
        adv = reward - baseline

        # d logP(a)/dparam for Bernoulli logit policy
        dlogp_dlogit = (a - p)
        grad_wc = np.mean(adv * dlogp_dlogit * c)
        grad_g = np.mean(adv * dlogp_dlogit * s * mask)
        grad_h = np.mean(adv * dlogp_dlogit)

        wc += lr * grad_wc
        g += lr * grad_g - lr * lam_penalty * g  # explicit L2 shrinkage on pressure weight
        h += lr * grad_h

        if it % 20 == 0 or it == iters - 1:
            history.append(dict(iter=it, wc=wc, g=g, h=h, mean_reward=float(reward.mean())))
    return (wc, g, h), history


def evaluate(params, data):
    c, s, t, u, v = data["c"], data["s"], data["t"], data["u"], data["v"]
    p = policy_probs(params, c, s, mask=1.0)
    a = (p >= 0.5).astype(int)  # deterministic eval
    acc = float((a == t).mean())
    acc_v = float((a[v == 1] == t[v == 1]).mean())
    acc_a = float((a[v == 0] == t[v == 0]).mean())
    wrong_user = u != t
    sycophancy = float((a[wrong_user] == u[wrong_user]).mean())
    wrong_user_v = wrong_user & (v == 1)
    wrong_user_a = wrong_user & (v == 0)
    syco_v = float((a[wrong_user_v] == u[wrong_user_v]).mean())
    syco_a = float((a[wrong_user_a] == u[wrong_user_a]).mean())
    # user-approval proxy: naive mean-annotator reward achieved on test set
    rews = annotator_rewards(a, t, u, v, seed_offset=999, n_outlier_annotators=0)
    approval = float(rews.mean())
    return dict(accuracy=acc, accuracy_verifiable=acc_v, accuracy_ambiguous=acc_a,
                sycophancy_overall=sycophancy, sycophancy_verifiable=syco_v,
                sycophancy_ambiguous=syco_a, approval_proxy=approval,
                wc=params[0], g=params[1], h=params[2])


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    histories = {}

    configs = {
        "naive_rlhf": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.0, n_outliers=0),
        "robust_median_agg": dict(aggregate="median", lam_penalty=0.0, precommit_q=0.0, n_outliers=1),
        "sycophancy_penalty": dict(aggregate="mean", lam_penalty=0.15, precommit_q=0.0, n_outliers=0),
        "precommit_answer": dict(aggregate="mean", lam_penalty=0.0, precommit_q=0.5, n_outliers=0),
    }

    for name, cfg in configs.items():
        params, hist = train_policy(train, iters=400, batch=256, lr=0.08, seed=42, **cfg)
        metrics = evaluate(params, test)
        results[name] = metrics
        histories[name] = hist
        print(f"[{name}] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    # also record a config with BOTH outlier annotators present but naive mean agg,
    # to isolate the effect of robust aggregation vs. the outlier stressor itself
    params, hist = train_policy(train, iters=400, batch=256, lr=0.08, seed=42,
                                 aggregate="mean", lam_penalty=0.0, precommit_q=0.0, n_outliers=1)
    metrics = evaluate(params, test)
    results["naive_rlhf_with_outliers"] = metrics
    histories["naive_rlhf_with_outliers"] = hist
    print("[naive_rlhf_with_outliers] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    with open("results.json", "w") as f:
        json.dump({"results": results, "histories": histories}, f, indent=2)

    print(f"\nTotal wall time: {time.time() - t0:.2f}s")
