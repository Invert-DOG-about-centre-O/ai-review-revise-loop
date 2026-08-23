"""
External-validity pilot requested by round-2 reviewers 1, 2, and 4: does the
verifiable/ambiguous sycophancy gap, and the L2-penalty mitigation, survive a
policy that does NOT expose a single interpretable pressure-weight scalar?

We replace the 3-parameter logistic policy with a small 2-layer MLP
(2 inputs [c, s] -> H=8 tanh hidden units -> 1 logit), trained with the same
REINFORCE + EMA-baseline recipe on the same claim distribution and annotator
reward model as sycophancy_sim.py. The "pressure pathway" generalizes to the
column of first-layer weights connecting the pressure input s to the hidden
layer; the L2 mitigation shrinks that column instead of a single scalar g.
"""
import time
import numpy as np

from sycophancy_sim import make_claims, annotator_rewards

N_TRAIN, N_TEST = 6000, 3000
train = make_claims(N_TRAIN, seed=1)
test = make_claims(N_TEST, seed=2)

H = 8


def init_params(rng):
    W1 = rng.normal(0, 0.5, size=(H, 2))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 0.5, size=H)
    b2 = 0.0
    return [W1, b1, W2, b2]


def forward(params, c, s, mask):
    W1, b1, W2, b2 = params
    z = W1[:, 0][None, :] * c[:, None] + W1[:, 1][None, :] * (s * mask)[:, None] + b1[None, :]
    h = np.tanh(z)
    logit = h @ W2 + b2
    p = 1.0 / (1.0 + np.exp(-logit))
    return p, h, z


def train_mlp(data, iters=400, batch=256, lr=0.05, lam_penalty=0.0, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    W1, b1, W2, b2 = params
    n = len(data["t"])
    baseline = 0.0
    for it in range(iters):
        idx = rng.integers(0, n, size=batch)
        c, s, t, u, v = (data["c"][idx], data["s"][idx], data["t"][idx],
                         data["u"][idx], data["v"][idx])
        mask = np.ones(batch)
        p, h, z = forward([W1, b1, W2, b2], c, s, mask)
        a = (rng.random(batch) < p).astype(int)

        rews = annotator_rewards(a, t, u, v, seed_offset=seed * 100000 + it, n_outlier_annotators=0)
        reward = rews.mean(axis=0)
        baseline = 0.95 * baseline + 0.05 * reward.mean()
        adv = reward - baseline

        delta = adv * (a - p)                      # (batch,)
        grad_W2 = np.mean(delta[:, None] * h, axis=0)
        grad_b2 = np.mean(delta)
        grad_z = delta[:, None] * W2[None, :] * (1 - h ** 2)   # (batch,H)
        grad_W1_c = np.mean(grad_z * c[:, None], axis=0)
        grad_W1_s = np.mean(grad_z * (s * mask)[:, None], axis=0)
        grad_b1 = np.mean(grad_z, axis=0)

        W2 = W2 + lr * grad_W2
        b2 = b2 + lr * grad_b2
        W1[:, 0] = W1[:, 0] + lr * grad_W1_c
        W1[:, 1] = W1[:, 1] + lr * grad_W1_s - lr * lam_penalty * W1[:, 1]
        b1 = b1 + lr * grad_b1
    return [W1, b1, W2, b2]


def evaluate(params, data):
    c, s, t, u, v = data["c"], data["s"], data["t"], data["u"], data["v"]
    p, _, _ = forward(params, c, s, mask=np.ones(len(c)))
    a = (p >= 0.5).astype(int)
    acc = float((a == t).mean())
    wrong_user = u != t
    wrong_user_v = wrong_user & (v == 1)
    wrong_user_a = wrong_user & (v == 0)
    syco_v = float((a[wrong_user_v] == u[wrong_user_v]).mean())
    syco_a = float((a[wrong_user_a] == u[wrong_user_a]).mean())
    W1, b1, W2, b2 = params
    pressure_norm = float(np.linalg.norm(W1[:, 1] * W2))  # generalized "pressure sensitivity"
    return dict(accuracy=acc, sycophancy_verifiable=syco_v, sycophancy_ambiguous=syco_a,
                pressure_norm=pressure_norm)


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = list(range(10))
    for lam, label in [(0.0, "mlp_naive"), (0.5, "mlp_l2_penalty")]:
        rows = []
        for sd in SEEDS:
            params = train_mlp(train, lam_penalty=lam, seed=sd)
            rows.append(evaluate(params, test))
        keys = rows[0].keys()
        means = {k: np.mean([r[k] for r in rows]) for k in keys}
        stds = {k: np.std([r[k] for r in rows]) for k in keys}
        print(f"[{label}] " + ", ".join(f"{k}={means[k]:.3f}+/-{stds[k]:.3f}" for k in keys))
    print(f"Total wall time: {time.time()-t0:.2f}s")
