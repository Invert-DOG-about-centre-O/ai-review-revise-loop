"""
Robustness check (multi-seed) for the shift-sweep finding in experiment.py:
that under strong distribution shift, LAC-conformal degrades MORE than the
naive uncalibrated baseline while APS-conformal degrades LESS. Re-run the
full pipeline (fresh data generator, fresh LSTM, fresh calibration) for 5
independent seeds and report mean +/- std of the coverage drop
(shift_frac=0 -> shift_frac=1) for each method.
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
N_TEST_SEQ = 500
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


def naive_topp_sets(probs, alpha):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    k = (cum < (1 - alpha)).sum(axis=1) + 1
    return order, np.clip(k, 1, probs.shape[1])


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


def aps_sets(probs, qhat, rng):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    n = probs.shape[0]
    cum_before = np.concatenate([np.zeros((n, 1)), cum[:, :-1]], axis=1)
    include = cum_before < qhat
    return order, np.clip(include.sum(axis=1) + 1, 1, probs.shape[1])


def evaluate(order, k, y):
    n = len(y)
    covered = np.array([y[i] in order[i, :k[i]] for i in range(n)])
    return covered.mean()


def run_seed(seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    true_kernel = make_markov_kernel(VOCAB, ORDER, 6.0, rng)
    train_seqs = sample_sequences(true_kernel, N_TRAIN_SEQ, SEQ_LEN, VOCAB, ORDER, rng)
    calib_seqs = sample_sequences(true_kernel, N_CALIB_SEQ, SEQ_LEN, VOCAB, ORDER, rng)

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

    calib_probs, calib_y = get_probs(model, calib_seqs, VOCAB)
    qhat_lac = lac_calibrate(calib_probs, calib_y, ALPHA)
    qhat_aps = aps_calibrate(calib_probs, calib_y, ALPHA, np.random.default_rng(seed + 100))
    eval_rng = np.random.default_rng(seed + 200)

    out = {}
    for shift_frac in [0.0, 1.0]:
        sk = shift_kernel(true_kernel, shift_frac, VOCAB)
        seqs_s = sample_sequences(sk, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng)
        probs, y = get_probs(model, seqs_s, VOCAB)
        o_n, k_n = naive_topp_sets(probs, ALPHA)
        o_l, k_l = lac_sets(probs, qhat_lac)
        o_a, k_a = aps_sets(probs, qhat_aps, eval_rng)
        out[shift_frac] = dict(naive=evaluate(o_n, k_n, y),
                                lac=evaluate(o_l, k_l, y),
                                aps=evaluate(o_a, k_a, y))
    return out


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    drops = {"naive": [], "lac": [], "aps": []}
    for s in seeds:
        r = run_seed(s)
        for m in drops:
            d = r[0.0][m] - r[1.0][m]
            drops[m].append(d)
        print(f"seed {s}: shift0={r[0.0]} shift1={r[1.0]}")

    print("\nCoverage drop (shift 0 -> shift 1), mean +/- std over", len(seeds), "seeds:")
    for m in drops:
        arr = np.array(drops[m])
        print(f"  {m}: {arr.mean():.4f} +/- {arr.std():.4f}  (values={list(np.round(arr,4))})")
