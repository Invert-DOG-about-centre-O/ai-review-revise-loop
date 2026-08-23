"""
Toy study: cheap single-pass uncertainty signals vs expensive sampling-based
signals for predicting correctness of a small transformer LM on a synthetic
2-digit addition task, plus a linear probe that tries to recover the
expensive signal from cheap features (a toy analogue of "semantic entropy
probes").

Fully self-contained: trains a small causal transformer from scratch on
synthetic data (no internet / pretrained weights needed), CPU only.
"""
import time
import math
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression

t0 = time.time()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Data: "aa+bb=sss" with a,b in [0,99], sum zero-padded to 3 digits.
# Fixed-length sequence of 9 characters -> no padding needed.
# Vocab: digits 0-9, '+', '='
# ---------------------------------------------------------------------------
CHARS = list("0123456789+=")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
IN_LEN = 6   # "aa+bb="
OUT_LEN = 3  # "sss"
SEQ_LEN = IN_LEN + OUT_LEN


def make_example():
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    s = a + b
    text = f"{a:02d}+{b:02d}={s:03d}"
    assert len(text) == SEQ_LEN
    return text, a, b, s


def encode(text):
    return [STOI[c] for c in text]


def batch(n):
    xs = []
    for _ in range(n):
        text, a, b, s = make_example()
        xs.append(encode(text))
    return torch.tensor(xs, dtype=torch.long)


# ---------------------------------------------------------------------------
# Tiny causal transformer LM
# ---------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, seq_len=SEQ_LEN, d=64, nhead=4, layers=2, ff=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                                batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.seq_len = seq_len
        self.d = d

    def forward(self, x, return_hidden=False):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.ln(h)
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits


device = "cpu"
model = TinyGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

TRAIN_STEPS = 260
BATCH = 128
print(f"[{time.time()-t0:.1f}s] starting training, {TRAIN_STEPS} steps")

for step in range(TRAIN_STEPS):
    x = batch(BATCH).to(device)
    inp = x[:, :-1]
    tgt = x[:, 1:]
    logits = model(inp)
    # only compute loss on positions that predict the answer digits
    # answer tokens occupy target indices IN_LEN-1 .. IN_LEN+OUT_LEN-2 (0-indexed into tgt)
    loss = F.cross_entropy(logits[:, IN_LEN - 1:].reshape(-1, VOCAB),
                            tgt[:, IN_LEN - 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"[{time.time()-t0:.1f}s] step {step} loss {loss.item():.4f}")

print(f"[{time.time()-t0:.1f}s] training done")

# ---------------------------------------------------------------------------
# Evaluation: for each test example, greedy-decode the 3 answer digits,
# record cheap single-pass features, then draw K stochastic samples for
# expensive self-consistency / sampling-entropy features.
# ---------------------------------------------------------------------------
model.eval()

N_TEST = 700
K_SAMPLES = 20
TEMP = 1.0

test_examples = [make_example() for _ in range(N_TEST)]


@torch.no_grad()
def greedy_decode_with_features(prefix_ids):
    """prefix_ids: list of IN_LEN ints. Returns (answer_str, maxprobs, entropies, hidden_mean)."""
    seq = list(prefix_ids)
    maxprobs, entropies, hiddens = [], [], []
    for _ in range(OUT_LEN):
        x = torch.tensor([seq], dtype=torch.long)
        logits, h = model(x, return_hidden=True)
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        mp = probs.max().item()
        nxt = int(probs.argmax().item())
        maxprobs.append(mp)
        entropies.append(ent)
        hiddens.append(h[0, -1].numpy())
        seq.append(nxt)
    answer = "".join(ITOS[i] for i in seq[IN_LEN:])
    return answer, float(np.mean(maxprobs)), float(np.mean(entropies)), np.mean(hiddens, axis=0)


@torch.no_grad()
def sample_decode(prefix_ids, temp=TEMP):
    seq = list(prefix_ids)
    for _ in range(OUT_LEN):
        x = torch.tensor([seq], dtype=torch.long)
        logits = model(x)
        last_logits = logits[0, -1] / temp
        probs = F.softmax(last_logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        seq.append(nxt)
    return "".join(ITOS[i] for i in seq[IN_LEN:])


rows = []
print(f"[{time.time()-t0:.1f}s] starting eval on {N_TEST} examples, K={K_SAMPLES} samples each")

for text, a, b, s in test_examples:
    true_ans = f"{s:03d}"
    prefix = encode(text)[:IN_LEN]

    greedy_ans, maxprob, pred_entropy, hidden_mean = greedy_decode_with_features(prefix)
    correct = int(greedy_ans == true_ans)

    samples = [sample_decode(prefix) for _ in range(K_SAMPLES)]
    agree = sum(1 for smp in samples if smp == greedy_ans) / K_SAMPLES

    # Shannon entropy over the empirical distribution of sampled answer strings
    from collections import Counter
    counts = Counter(samples)
    probs_emp = np.array(list(counts.values())) / K_SAMPLES
    sample_entropy = float(-(probs_emp * np.log(probs_emp)).sum())

    rows.append({
        "correct": correct,
        "maxprob": maxprob,
        "pred_entropy": pred_entropy,
        "sample_agree": agree,
        "sample_entropy": sample_entropy,
        "hidden": hidden_mean,
    })

print(f"[{time.time()-t0:.1f}s] eval decoding done")

acc = np.mean([r["correct"] for r in rows])
print(f"Greedy accuracy on test set: {acc:.3f}")

y = np.array([r["correct"] for r in rows])
maxprob = np.array([r["maxprob"] for r in rows])
pred_entropy = np.array([r["pred_entropy"] for r in rows])
sample_agree = np.array([r["sample_agree"] for r in rows])
sample_entropy = np.array([r["sample_entropy"] for r in rows])
hidden = np.stack([r["hidden"] for r in rows])

results = {"accuracy": float(acc), "n_test": N_TEST, "k_samples": K_SAMPLES}

# AUROC: does the score separate correct vs incorrect?
# use "confidence-like" direction: maxprob & sample_agree (higher=more confident, positive-correlated w/ correct)
# entropies are negatively correlated with correctness, so use -entropy for AUROC in same direction
def safe_auroc(score, label):
    if len(set(label.tolist())) < 2:
        return None
    return roc_auc_score(label, score)

results["auroc"] = {
    "maxprob (cheap, single-pass)": safe_auroc(maxprob, y),
    "neg_pred_entropy (cheap, single-pass)": safe_auroc(-pred_entropy, y),
    "sample_agree (expensive, K samples)": safe_auroc(sample_agree, y),
    "neg_sample_entropy (expensive, K samples)": safe_auroc(-sample_entropy, y),
}

# ECE for maxprob-as-confidence and sample_agree-as-confidence
def ece(conf, label, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = len(conf)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = label[mask].mean()
        e += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(e)

results["ece"] = {
    "maxprob": ece(maxprob, y),
    "sample_agree": ece(sample_agree, y),
}

# ---------------------------------------------------------------------------
# Cheap linear probe trying to recover the EXPENSIVE sample_entropy signal
# from CHEAP features (maxprob, pred_entropy, hidden state) -- single forward
# pass only, no extra sampling. Held-out split for honesty.
# ---------------------------------------------------------------------------
idx = np.arange(N_TEST)
rng = np.random.RandomState(0)
rng.shuffle(idx)
split = int(0.5 * N_TEST)
train_idx, test_idx = idx[:split], idx[split:]

X_cheap = np.concatenate([
    maxprob.reshape(-1, 1),
    pred_entropy.reshape(-1, 1),
    hidden,
], axis=1)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_cheap[train_idx])
Xs = scaler.transform(X_cheap)
probe = Ridge(alpha=1.0)
probe.fit(Xs[train_idx], sample_entropy[train_idx])
pred_sample_entropy = probe.predict(Xs[test_idx])

# R^2 of probe predicting the expensive signal
ss_res = np.sum((sample_entropy[test_idx] - pred_sample_entropy) ** 2)
ss_tot = np.sum((sample_entropy[test_idx] - sample_entropy[test_idx].mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
results["probe_r2_recovering_sample_entropy"] = float(r2)

# Does the PROBE'S predicted expensive-entropy separate correct/incorrect as well
# as the true expensive signal, on the held-out half?
results["auroc_holdout"] = {
    "true_neg_sample_entropy (expensive)": safe_auroc(-sample_entropy[test_idx], y[test_idx]),
    "probe_neg_predicted_sample_entropy (cheap, probes expensive signal)": safe_auroc(-pred_sample_entropy, y[test_idx]),
    "maxprob (cheap, single-pass)": safe_auroc(maxprob[test_idx], y[test_idx]),
    "neg_pred_entropy (cheap, single-pass)": safe_auroc(-pred_entropy[test_idx], y[test_idx]),
}

print(json.dumps(results, indent=2))

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"[{time.time()-t0:.1f}s] DONE, total elapsed {time.time()-t0:.1f}s")
