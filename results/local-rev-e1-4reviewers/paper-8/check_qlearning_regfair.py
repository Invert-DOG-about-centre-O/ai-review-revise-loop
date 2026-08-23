import numpy as np

N_ARMS = 11
ALPHA_GRID = np.linspace(0.0, 1.0, N_ARMS)
RNG_SEED0 = 12345


def run_qlearning_regfair(seed, p_user, p_ai, lam=0.8, n_rounds=20000, q_lr=0.05, epsilon=0.1):
    rng = np.random.default_rng(seed)
    Q = np.zeros(N_ARMS)
    accs = np.empty(n_rounds)
    alpha_choice = np.empty(n_rounds)

    for t in range(n_rounds):
        y = rng.integers(0, 2)
        b_u = y if rng.random() < p_user else 1 - y
        e = y if rng.random() < p_ai else 1 - y

        if rng.random() < epsilon:
            arm = rng.integers(0, N_ARMS)
        else:
            arm = int(np.argmax(Q))
        alpha_t = ALPHA_GRID[arm]

        use_user = rng.random() < alpha_t
        o = b_u if use_user else e
        agree = 1.0 if o == b_u else 0.0
        correct = 1.0 if o == y else 0.0

        reward = (1 - lam) * agree + lam * correct

        Q[arm] += q_lr * (reward - Q[arm])

        accs[t] = correct
        alpha_choice[t] = alpha_t

    tail = max(1, int(0.2 * n_rounds))
    return dict(mean_alpha_tail=float(np.mean(alpha_choice[-tail:])),
                accuracy=float(np.mean(accs[-tail:])))


rows = []
for s in range(10):
    r = run_qlearning_regfair(RNG_SEED0 + s, 0.60, 0.80, lam=0.8, n_rounds=20000)
    rows.append(r)

alphas = [r['mean_alpha_tail'] for r in rows]
accs = [r['accuracy'] for r in rows]
print("mean alpha_tail:", np.mean(alphas), "mean accuracy:", np.mean(accs))
