"""
Conformal prediction sets for next-token prediction: coverage/efficiency
trade-offs under temperature miscalibration and distribution shift, in a
synthetic Markov-chain setting with a known ground-truth generating process.

Fully deterministic (fixed seeds). Runs on CPU in a few minutes.
"""
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"
VOCAB = 24          # vocabulary size
ORDER = 2           # true Markov order of the generating process
SEQ_LEN = 30
N_TRAIN_SEQ = 1500
N_CALIB_SEQ = 800    # will be subsampled for the calib-size ablation
N_TEST_SEQ = 800
ALPHA = 0.10         # target miscoverage -> 90% nominal coverage

# ---------------------------------------------------------------------------
# 1. Synthetic data-generating process: a random, peaked order-2 Markov chain
# ---------------------------------------------------------------------------
def make_markov_kernel(vocab, order, peak=6.0, rng=None):
    """Dirichlet-peaked transition distribution for every (context) -> next-token."""
    rng = rng or np.random.default_rng(SEED)
    n_ctx = vocab ** order
    alpha_vec = np.full(vocab, 0.3)
    kernel = np.zeros((n_ctx, vocab))
    for c in range(n_ctx):
        # give each context a random "preferred" token cluster to keep entropy low-ish
        a = alpha_vec.copy()
        favored = rng.choice(vocab, size=3, replace=False)
        a[favored] += peak
        kernel[c] = rng.dirichlet(a)
    return kernel


def ctx_index(history, vocab, order):
    idx = 0
    for h in history[-order:]:
        idx = idx * vocab + h
    return idx


def sample_sequences(kernel, n_seq, seq_len, vocab, order, rng):
    seqs = np.zeros((n_seq, seq_len), dtype=np.int64)
    for i in range(n_seq):
        history = list(rng.integers(0, vocab, size=order))
        out = []
        for t in range(seq_len):
            c = ctx_index(history, vocab, order)
            tok = rng.choice(vocab, p=kernel[c])
            out.append(tok)
            history.append(tok)
        seqs[i] = out
    return seqs


def shift_kernel(kernel, shift_frac, vocab, rng):
    """Interpolate the transition kernel toward uniform to simulate covariate/label shift."""
    uniform = np.full(vocab, 1.0 / vocab)
    return (1 - shift_frac) * kernel + shift_frac * uniform


rng = np.random.default_rng(SEED)
true_kernel = make_markov_kernel(VOCAB, ORDER, peak=6.0, rng=rng)

train_seqs = sample_sequences(true_kernel, N_TRAIN_SEQ, SEQ_LEN, VOCAB, ORDER, rng)
calib_seqs = sample_sequences(true_kernel, N_CALIB_SEQ, SEQ_LEN, VOCAB, ORDER, rng)
test_seqs = sample_sequences(true_kernel, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng)

shifted_kernel = shift_kernel(true_kernel, shift_frac=0.5, vocab=VOCAB, rng=rng)
test_seqs_shift = sample_sequences(shifted_kernel, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng)

# ---------------------------------------------------------------------------
# 2. Small LSTM language model (deliberately undertrained relative to the
#    true generator, so its softmax output is a realistically imperfect,
#    miscalibrated estimate of the ground-truth next-token distribution).
# ---------------------------------------------------------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab, emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        return self.out(h)


def to_xy(seqs):
    x = torch.tensor(seqs[:, :-1], dtype=torch.long)
    y = torch.tensor(seqs[:, 1:], dtype=torch.long)
    return x, y


model = LSTMLM(VOCAB).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
lossf = nn.CrossEntropyLoss()

x_train, y_train = to_xy(train_seqs)
BATCH = 64
N_EPOCHS = 8

t0 = time.time()
for epoch in range(N_EPOCHS):
    perm = torch.randperm(x_train.size(0))
    total_loss, nb = 0.0, 0
    for i in range(0, x_train.size(0), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = x_train[idx], y_train[idx]
        logits = model(xb)
        loss = lossf(logits.reshape(-1, VOCAB), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        nb += 1
    print(f"epoch {epoch} loss {total_loss / nb:.4f}")
print(f"training took {time.time() - t0:.1f}s")

model.eval()


def get_probs(seqs, temperature=1.0):
    """Return model softmax probs (N*T', V) and the true next tokens, flattened
    over all positions/sequences (excluding the first ORDER-1 warmup steps)."""
    x, y = to_xy(seqs)
    with torch.no_grad():
        logits = model(x) / temperature
        probs = torch.softmax(logits, dim=-1)
    probs = probs.reshape(-1, VOCAB).numpy()
    y = y.reshape(-1).numpy()
    return probs, y


# ---------------------------------------------------------------------------
# 3. Prediction-set constructors
# ---------------------------------------------------------------------------
def naive_topp_sets(probs, alpha):
    """Uncalibrated cumulative-probability ('nucleus') sets targeting 1-alpha
    directly from the model's own softmax, with no conformal correction."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    # smallest k such that cum >= 1 - alpha
    k = (cum < (1 - alpha)).sum(axis=1) + 1
    k = np.clip(k, 1, probs.shape[1])
    return order, k  # set = order[i, :k[i]]


def lac_calibrate(calib_probs, calib_y, alpha):
    """Least Ambiguous Classifier (LAC): nonconformity = 1 - p(true class)."""
    scores = 1 - calib_probs[np.arange(len(calib_y)), calib_y]
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, q_level, method="higher")
    return qhat


def lac_sets(probs, qhat):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    include = sorted_p >= (1 - qhat)
    k = include.sum(axis=1)
    k = np.clip(k, 1, probs.shape[1])
    return order, k


def aps_calibrate(calib_probs, calib_y, alpha, rng):
    """Adaptive Prediction Sets (Romano et al. 2020), with randomization."""
    order = np.argsort(-calib_probs, axis=1)
    sorted_p = np.take_along_axis(calib_probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = len(calib_y)
    rank_of_true = np.array([np.where(order[i] == calib_y[i])[0][0] for i in range(n)])
    cum_before = np.where(rank_of_true > 0,
                           cum[np.arange(n), np.maximum(rank_of_true - 1, 0)], 0.0)
    p_true = sorted_p[np.arange(n), rank_of_true]
    u = rng.uniform(size=n)
    scores = cum_before + u * p_true
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, q_level, method="higher")
    return qhat


def aps_sets(probs, qhat, rng):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = probs.shape[0]
    u = rng.uniform(size=n)
    thresh = qhat
    # include tokens while cumulative (with randomization on the boundary token) <= qhat
    cum_before = np.concatenate([np.zeros((n, 1)), cum[:, :-1]], axis=1)
    include = cum_before < thresh
    # ensure boundary token included per randomized rule approx (deterministic version: include while cum_before < qhat)
    k = include.sum(axis=1) + 1
    k = np.clip(k, 1, probs.shape[1])
    return order, k


def evaluate(order, k, y):
    n = len(y)
    covered = np.zeros(n, dtype=bool)
    sizes = k.copy()
    for i in range(n):
        covered[i] = y[i] in order[i, :k[i]]
    return covered.mean(), sizes.mean()


# ---------------------------------------------------------------------------
# 4. Main experiments
# ---------------------------------------------------------------------------
results = {"main": [], "calib_size_ablation": [], "temperature_ablation": []}
eval_rng = np.random.default_rng(SEED + 1)

# --- Experiment A: in-distribution vs shifted, at fixed temperature=1.0 ---
calib_probs, calib_y = get_probs(calib_seqs, temperature=1.0)
qhat_lac = lac_calibrate(calib_probs, calib_y, ALPHA)
qhat_aps = aps_calibrate(calib_probs, calib_y, ALPHA, np.random.default_rng(SEED + 2))

for name, seqs in [("in_distribution", test_seqs), ("shifted_0.5", test_seqs_shift)]:
    probs, y = get_probs(seqs, temperature=1.0)
    order_n, k_n = naive_topp_sets(probs, ALPHA)
    cov_n, size_n = evaluate(order_n, k_n, y)
    order_l, k_l = lac_sets(probs, qhat_lac)
    cov_l, size_l = evaluate(order_l, k_l, y)
    order_a, k_a = aps_sets(probs, qhat_aps, eval_rng)
    cov_a, size_a = evaluate(order_a, k_a, y)
    row = dict(condition=name,
               naive_coverage=cov_n, naive_size=size_n,
               lac_coverage=cov_l, lac_size=size_l, lac_qhat=qhat_lac,
               aps_coverage=cov_a, aps_size=size_a, aps_qhat=qhat_aps,
               n_test_points=len(y))
    results["main"].append(row)
    print(name, row)

# --- Experiment B: calibration-set-size ablation (in-distribution) ---
for n_calib_seq in [10, 25, 50, 100, 200, 800]:
    sub = calib_seqs[:n_calib_seq]
    cprobs, cy = get_probs(sub, temperature=1.0)
    if len(cy) < 5:
        continue
    q_l = lac_calibrate(cprobs, cy, ALPHA)
    q_a = aps_calibrate(cprobs, cy, ALPHA, np.random.default_rng(SEED + 3))
    probs, y = get_probs(test_seqs, temperature=1.0)
    order_l, k_l = lac_sets(probs, q_l)
    cov_l, size_l = evaluate(order_l, k_l, y)
    order_a, k_a = aps_sets(probs, q_a, eval_rng)
    cov_a, size_a = evaluate(order_a, k_a, y)
    row = dict(n_calib_points=len(cy), n_calib_seqs=n_calib_seq,
               lac_coverage=cov_l, lac_size=size_l,
               aps_coverage=cov_a, aps_size=size_a)
    results["calib_size_ablation"].append(row)
    print("calib_ablation", row)

# --- Experiment C: temperature ablation ---
for T in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    cprobs, cy = get_probs(calib_seqs, temperature=T)
    q_n_alpha = ALPHA  # naive always targets nominal directly, no calibration
    q_l = lac_calibrate(cprobs, cy, ALPHA)
    q_a = aps_calibrate(cprobs, cy, ALPHA, np.random.default_rng(SEED + 4))
    probs, y = get_probs(test_seqs, temperature=T)
    order_n, k_n = naive_topp_sets(probs, ALPHA)
    cov_n, size_n = evaluate(order_n, k_n, y)
    order_l, k_l = lac_sets(probs, q_l)
    cov_l, size_l = evaluate(order_l, k_l, y)
    order_a, k_a = aps_sets(probs, q_a, eval_rng)
    cov_a, size_a = evaluate(order_a, k_a, y)
    row = dict(temperature=T,
               naive_coverage=cov_n, naive_size=size_n,
               lac_coverage=cov_l, lac_size=size_l,
               aps_coverage=cov_a, aps_size=size_a)
    results["temperature_ablation"].append(row)
    print("temp_ablation", row)

# --- Experiment D: shift severity sweep (finer resolution, per skill note on thresholds) ---
results["shift_sweep"] = []
for shift_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    sk = shift_kernel(true_kernel, shift_frac, VOCAB, rng)
    seqs_s = sample_sequences(sk, 400, SEQ_LEN, VOCAB, ORDER, rng)
    probs, y = get_probs(seqs_s, temperature=1.0)
    order_l, k_l = lac_sets(probs, qhat_lac)
    cov_l, size_l = evaluate(order_l, k_l, y)
    order_a, k_a = aps_sets(probs, qhat_aps, eval_rng)
    cov_a, size_a = evaluate(order_a, k_a, y)
    order_n, k_n = naive_topp_sets(probs, ALPHA)
    cov_n, size_n = evaluate(order_n, k_n, y)
    row = dict(shift_frac=shift_frac,
               naive_coverage=cov_n, lac_coverage=cov_l, aps_coverage=cov_a,
               naive_size=size_n, lac_size=size_l, aps_size=size_a)
    results["shift_sweep"].append(row)
    print("shift_sweep", row)

def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=_default)

print("DONE total time", time.time() - t0)
