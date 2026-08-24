"""
Follow-up experiments answering round-1 review questions:
 Q1: does the deterministic APS test-time rule (vs. fully randomized) bias
     coverage/size, and in which direction?
 Q2: does the LAC nonconformity-score distribution move more under shift than
     the APS score distribution (the hypothesized mechanism)?
 Q3: does APS's shift-robustness advantage survive if LAC is recalibrated to
     match APS's average set size (rather than matched nominal alpha)?
Reuses the exact same synthetic generator / LSTM / calibration code as
experiment.py and multiseed_check.py. Deterministic (fixed seeds).
"""
import math
import numpy as np
import torch
import torch.nn as nn

VOCAB = 24
ORDER = 2
SEQ_LEN = 30
N_TRAIN_SEQ = 1500
N_CALIB_SEQ = 800
N_TEST_SEQ = 800
ALPHA = 0.10


def make_markov_kernel(vocab, order, peak, rng):
    n_ctx = vocab ** order
    kernel = np.zeros((n_ctx, vocab))
    for c in range(n_ctx):
        a = np.full(vocab, 0.3)
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


def shift_kernel(kernel, shift_frac, vocab):
    uniform = np.full(vocab, 1.0 / vocab)
    return (1 - shift_frac) * kernel + shift_frac * uniform


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


def get_probs(model, seqs, vocab):
    x, y = to_xy(seqs)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)
    return probs.reshape(-1, vocab).numpy(), y.reshape(-1).numpy()


def lac_calibrate(probs, y, alpha):
    scores = 1 - probs[np.arange(len(y)), y]
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, q_level, method="higher")


def lac_sets(probs, qhat):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    include = sorted_p >= (1 - qhat)
    return order, np.clip(include.sum(axis=1), 1, probs.shape[1])


def aps_calibrate(probs, y, alpha, rng):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = len(y)
    rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(n)])
    cum_before = np.where(rank > 0, cum[np.arange(n), np.maximum(rank - 1, 0)], 0.0)
    p_true = sorted_p[np.arange(n), rank]
    u = rng.uniform(size=n)
    scores = cum_before + u * p_true
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, q_level, method="higher")


def aps_scores(probs, y):
    """Return (cum_before, p_true) for the true token at every position -- the two
    pieces of the APS nonconformity score, without the randomization draw."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = len(y)
    rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(n)])
    cum_before = np.where(rank > 0, cum[np.arange(n), np.maximum(rank - 1, 0)], 0.0)
    p_true = sorted_p[np.arange(n), rank]
    return cum_before, p_true


def aps_sets_deterministic(probs, qhat):
    """Paper's original test-time rule: include while cum_before < qhat (no per-token randomization)."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = probs.shape[0]
    cum_before = np.concatenate([np.zeros((n, 1)), cum[:, :-1]], axis=1)
    include = cum_before < qhat
    return order, np.clip(include.sum(axis=1) + 1, 1, probs.shape[1])


def aps_sets_randomized(probs, qhat, rng):
    """Fully randomized Romano et al. (2020) rule: include the boundary token
    with probability (qhat - cum_before_boundary) / p_boundary."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = probs.shape[0]
    cum_before = np.concatenate([np.zeros((n, 1)), cum[:, :-1]], axis=1)
    k_det = (cum_before < qhat).sum(axis=1)  # tokens strictly before boundary
    k_det = np.clip(k_det, 0, probs.shape[1] - 1)
    boundary_cum_before = cum_before[np.arange(n), k_det]
    boundary_p = sorted_p[np.arange(n), k_det]
    boundary_p_safe = np.where(boundary_p > 0, boundary_p, 1.0)
    incl_prob = np.clip((qhat - boundary_cum_before) / boundary_p_safe, 0.0, 1.0)
    u = rng.uniform(size=n)
    include_boundary = u < incl_prob
    k = k_det + include_boundary.astype(int)
    k = np.clip(k, 1, probs.shape[1])
    return order, k


def lac_sets_alpha(probs, calib_probs, calib_y, alpha):
    qhat = lac_calibrate(calib_probs, calib_y, alpha)
    return lac_sets(probs, qhat), qhat


def evaluate(order, k, y):
    n = len(y)
    covered = np.array([y[i] in order[i, :k[i]] for i in range(n)])
    return covered.mean(), k.mean()


def train_model(seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    true_kernel = make_markov_kernel(VOCAB, ORDER, 6.0, rng)
    train_seqs = sample_sequences(true_kernel, N_TRAIN_SEQ, SEQ_LEN, VOCAB, ORDER, rng)
    calib_seqs = sample_sequences(true_kernel, N_CALIB_SEQ, SEQ_LEN, VOCAB, ORDER, rng)
    test_seqs = sample_sequences(true_kernel, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng)

    model = LSTMLM(VOCAB)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    x_train, y_train = to_xy(train_seqs)
    for epoch in range(8):
        perm = torch.randperm(x_train.size(0))
        for i in range(0, x_train.size(0), 64):
            idx = perm[i:i + 64]
            logits = model(x_train[idx])
            loss = lossf(logits.reshape(-1, VOCAB), y_train[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return rng, true_kernel, calib_seqs, test_seqs, model


# =========================================================================
# Q1: deterministic vs. fully randomized APS test-time rule
# =========================================================================
print("=" * 70)
print("Q1: deterministic vs randomized APS")
rng, true_kernel, calib_seqs, test_seqs, model = train_model(seed=0)
calib_probs, calib_y = get_probs(model, calib_seqs, VOCAB)
qhat_aps = aps_calibrate(calib_probs, calib_y, ALPHA, np.random.default_rng(102))
shifted_kernel = shift_kernel(true_kernel, 0.5, VOCAB)
test_seqs_shift = sample_sequences(shifted_kernel, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng)

q1_results = {}
for name, seqs in [("in_distribution", test_seqs), ("shifted_0.5", test_seqs_shift)]:
    probs, y = get_probs(model, seqs, VOCAB)
    order_d, k_d = aps_sets_deterministic(probs, qhat_aps)
    cov_d, size_d = evaluate(order_d, k_d, y)
    order_r, k_r = aps_sets_randomized(probs, qhat_aps, np.random.default_rng(200))
    cov_r, size_r = evaluate(order_r, k_r, y)
    q1_results[name] = dict(det_coverage=cov_d, det_size=size_d,
                             rand_coverage=cov_r, rand_size=size_r)
    print(f"  {name}: deterministic cov={cov_d:.4f} size={size_d:.3f} | "
          f"randomized cov={cov_r:.4f} size={size_r:.3f}")

# =========================================================================
# Q2: does the LAC score distribution move more under shift than APS's?
# =========================================================================
print("=" * 70)
print("Q2: nonconformity-score distributional shift, LAC vs APS")
qhat_lac = lac_calibrate(calib_probs, calib_y, ALPHA)
lac_score_calib = 1 - calib_probs[np.arange(len(calib_y)), calib_y]
cum_before_c, p_true_c = aps_scores(calib_probs, calib_y)
aps_score_calib_mean = cum_before_c + 0.5 * p_true_c  # expected score under u~U(0,1)

q2_results = []
for shift_frac in [0.0, 0.2, 0.5, 0.7, 1.0]:
    sk = shift_kernel(true_kernel, shift_frac, VOCAB)
    seqs_s = sample_sequences(sk, 500, SEQ_LEN, VOCAB, ORDER, rng)
    probs, y = get_probs(model, seqs_s, VOCAB)
    lac_score = 1 - probs[np.arange(len(y)), y]
    cum_before, p_true = aps_scores(probs, y)
    aps_score_mean = cum_before + 0.5 * p_true
    row = dict(shift_frac=shift_frac,
               lac_score_mean=lac_score.mean(), lac_score_std=lac_score.std(),
               aps_score_mean=aps_score_mean.mean(), aps_score_std=aps_score_mean.std())
    q2_results.append(row)
    print(f"  s={shift_frac}: LAC score mean={row['lac_score_mean']:.4f} "
          f"(calib mean={lac_score_calib.mean():.4f}) | "
          f"APS score mean={row['aps_score_mean']:.4f} "
          f"(calib mean={aps_score_calib_mean.mean():.4f})")

lac_move = (q2_results[-1]['lac_score_mean'] - lac_score_calib.mean()) / lac_score_calib.std()
aps_move = (q2_results[-1]['aps_score_mean'] - aps_score_calib_mean.mean()) / aps_score_calib_mean.std()
print(f"  Normalized score-distribution shift (s=0->1, in calib-std units): "
      f"LAC={lac_move:.3f}  APS={aps_move:.3f}")

# =========================================================================
# Q3: LAC vs APS shift-robustness at matched average set size
# =========================================================================
print("=" * 70)
print("Q3: matched-set-size shift-robustness comparison (3 seeds)")


def find_alpha_for_target_size(calib_probs, calib_y, target_size, probs_ref):
    lo, hi = 0.001, 0.6
    for _ in range(25):
        mid = (lo + hi) / 2
        qhat = lac_calibrate(calib_probs, calib_y, mid)
        _, k = lac_sets(probs_ref, qhat)
        size = k.mean()
        if size > target_size:
            lo = mid
        else:
            hi = mid
    return mid, lac_calibrate(calib_probs, calib_y, mid)


q3_rows = []
for seed in [0, 1, 2]:
    rng_s, kernel_s, calib_s, test_s, model_s = train_model(seed)
    cprobs, cy = get_probs(model_s, calib_s, VOCAB)
    qhat_aps_s = aps_calibrate(cprobs, cy, ALPHA, np.random.default_rng(seed + 100))
    probs0, y0 = get_probs(model_s, test_s, VOCAB)
    order_a0, k_a0 = aps_sets_deterministic(probs0, qhat_aps_s)
    target_size = k_a0.mean()
    alpha_matched, qhat_lac_matched = find_alpha_for_target_size(cprobs, cy, target_size, probs0)

    eval_rng = np.random.default_rng(seed + 300)
    out = {}
    for shift_frac in [0.0, 1.0]:
        sk = shift_kernel(kernel_s, shift_frac, VOCAB)
        seqs_shift = sample_sequences(sk, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng_s)
        probs, y = get_probs(model_s, seqs_shift, VOCAB)
        o_l, k_l = lac_sets(probs, qhat_lac_matched)
        cov_l, size_l = evaluate(o_l, k_l, y)
        o_a, k_a = aps_sets_deterministic(probs, qhat_aps_s)
        cov_a, size_a = evaluate(o_a, k_a, y)
        out[shift_frac] = dict(lac_cov=cov_l, lac_size=size_l, aps_cov=cov_a, aps_size=size_a)
    drop_l = out[0.0]['lac_cov'] - out[1.0]['lac_cov']
    drop_a = out[0.0]['aps_cov'] - out[1.0]['aps_cov']
    q3_rows.append(dict(seed=seed, alpha_matched=alpha_matched,
                         lac_size_s0=out[0.0]['lac_size'], aps_size_s0=out[0.0]['aps_size'],
                         lac_drop=drop_l, aps_drop=drop_a))
    print(f"  seed {seed}: matched alpha_lac={alpha_matched:.4f} | "
          f"sizes s=0: LAC={out[0.0]['lac_size']:.2f} APS={out[0.0]['aps_size']:.2f} | "
          f"coverage drop: LAC={drop_l:.4f} APS={drop_a:.4f}")

lac_drops = np.array([r['lac_drop'] for r in q3_rows])
aps_drops = np.array([r['aps_drop'] for r in q3_rows])
print(f"  Matched-size mean drop over 3 seeds: LAC={lac_drops.mean():.4f} +/- {lac_drops.std():.4f}  "
      f"APS={aps_drops.mean():.4f} +/- {aps_drops.std():.4f}")

print("=" * 70)
print("DONE")
