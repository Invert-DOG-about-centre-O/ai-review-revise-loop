"""
Entropy-Conditioned Adaptive Temperature Scaling for Language-Model Calibration.

We train a small char-level LSTM language model on synthetic text drawn from an
order-2 Markov source. We then study the calibration of its next-token
predictions and compare post-hoc calibration methods:

  (0) Raw softmax (no calibration)
  (1) Global temperature scaling (single scalar T, standard baseline)
  (2) [ours] Entropy-conditioned adaptive temperature scaling: fit a separate
      temperature per predictive-entropy bin, i.e. T = T(H(p)).

Key question: under distribution shift (a Markov source with different
parameters), does a single global temperature transfer, and does making the
temperature depend on the model's own predictive entropy help?

All parameters are fit on a validation split and evaluated on held-out test
splits (in-domain and shifted). CPU-only, finishes in a couple of minutes.
"""
import time, math, json
import numpy as np
import torch
import torch.nn as nn

t0 = time.time()
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Synthetic language: order-2 Markov chain over a small alphabet.
# A "temperature" tau on the source transition logits controls source entropy;
# distribution shift = changing tau (and re-drawing transition logits).
# ---------------------------------------------------------------------------
V = 20                      # alphabet size
ORDER = 2                   # markov order

def make_source(seed, tau):
    r = np.random.default_rng(seed)
    # transition logits for each (prev1, prev2) context
    logits = r.normal(0, 1.0, size=(V, V, V)) / tau
    P = np.exp(logits - logits.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    return P

def sample_text(P, n, seed):
    r = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.int64)
    x[0] = r.integers(V); x[1] = r.integers(V)
    for i in range(2, n):
        p = P[x[i-1], x[i-2]]
        x[i] = r.choice(V, p=p)
    return x

# In-domain source (train/val/test) and a shifted source (harder test).
P_in  = make_source(seed=1, tau=1.0)
P_out = make_source(seed=2, tau=1.6)   # different params + higher entropy

# Small training set + many epochs -> the model overfits and becomes
# OVERCONFIDENT, the realistic regime where calibration matters.
N_TRAIN, N_VAL, N_TEST = 15000, 12000, 12000
train = sample_text(P_in, N_TRAIN, seed=10)
val   = sample_text(P_in, N_VAL,   seed=11)
test  = sample_text(P_in, N_TEST,  seed=12)
test_shift = sample_text(P_out, N_TEST, seed=13)

# ---------------------------------------------------------------------------
# Small char-level LSTM language model.
# ---------------------------------------------------------------------------
CTX = 8       # context length fed to the model
EMB = 48
HID = 128

class CharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.lstm = nn.LSTM(EMB, HID, batch_first=True)
        self.head = nn.Linear(HID, V)
    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        return self.head(o[:, -1, :])   # predict next token after context

def make_batches(seq, ctx, bs):
    # build (context, target) pairs
    X, Y = [], []
    for i in range(ctx, len(seq)):
        X.append(seq[i-ctx:i]); Y.append(seq[i])
    X = torch.tensor(np.array(X)); Y = torch.tensor(np.array(Y))
    idx = torch.randperm(len(X))
    X, Y = X[idx], Y[idx]
    for j in range(0, len(X), bs):
        yield X[j:j+bs], Y[j:j+bs]

model = CharLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
lossf = nn.CrossEntropyLoss()

EPOCHS = 18
BS = 128
model.train()
for ep in range(EPOCHS):
    tot = 0.0; nb = 0
    for xb, yb in make_batches(train, CTX, BS):
        opt.zero_grad()
        logits = model(xb)
        loss = lossf(logits, yb)
        loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    print(f"epoch {ep} train_nll={tot/nb:.4f}  ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Collect held-out logits + targets for calibration.
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect(seq):
    model.eval()
    Xs, Ys = [], []
    for i in range(CTX, len(seq)):
        Xs.append(seq[i-CTX:i]); Ys.append(seq[i])
    X = torch.tensor(np.array(Xs)); Y = torch.tensor(np.array(Ys))
    logits = []
    for j in range(0, len(X), 1024):
        logits.append(model(X[j:j+1024]))
    return torch.cat(logits), Y

val_logits, val_y   = collect(val)
test_logits, test_y = collect(test)
shift_logits, shift_y = collect(test_shift)

# ---------------------------------------------------------------------------
# Calibration metrics.
# ---------------------------------------------------------------------------
def nll(logits, y, T=1.0):
    return nn.functional.cross_entropy(logits / T, y).item()

def probs(logits, T=1.0):
    return torch.softmax(logits / T, dim=-1)

def ece(logits, y, T=1.0, n_bins=15):
    p = probs(logits, T)
    conf, pred = p.max(dim=-1)
    correct = (pred == y).float()
    conf = conf.numpy(); correct = correct.numpy()
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b+1])
        if m.sum() == 0: continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e

def entropy(logits, T=1.0):
    p = probs(logits, T)
    return (-(p * torch.log(p + 1e-12)).sum(-1))

# ---------------------------------------------------------------------------
# (1) Global temperature scaling: fit single T on val NLL (grid + refine).
# ---------------------------------------------------------------------------
def fit_global_T(logits, y):
    Ts = np.linspace(0.5, 14.0, 136)
    best = min(Ts, key=lambda T: nll(logits, y, T))
    # local refine
    fine = np.linspace(max(0.05, best-0.1), best+0.1, 21)
    return float(min(fine, key=lambda T: nll(logits, y, T)))

T_global = fit_global_T(val_logits, val_y)

# ---------------------------------------------------------------------------
# (2) Entropy-conditioned adaptive temperature: bin val examples by raw
# predictive entropy, fit a separate temperature per bin. At eval time each
# example uses the temperature of the bin its entropy falls into.
# ---------------------------------------------------------------------------
N_EBINS = 8
val_H = entropy(val_logits).numpy()
edges = np.quantile(val_H, np.linspace(0, 1, N_EBINS + 1))
edges[0] -= 1e-6; edges[-1] += 1e-6

def bin_of(H):
    return np.clip(np.digitize(H, edges) - 1, 0, N_EBINS - 1)

bin_T = np.ones(N_EBINS)
val_bins = bin_of(val_H)
for b in range(N_EBINS):
    m = val_bins == b
    if m.sum() < 20:
        bin_T[b] = T_global; continue
    bin_T[b] = fit_global_T(val_logits[m], val_y[m])

def adaptive_apply(logits):
    H = entropy(logits).numpy()
    b = bin_of(H)
    T = torch.tensor(bin_T[b], dtype=torch.float32).unsqueeze(-1)
    return logits / T

def nll_from_probs(scaled_logits, y):
    logp = torch.log_softmax(scaled_logits, dim=-1)
    return nn.functional.nll_loss(logp, y).item()

def ece_from_probs(scaled_logits, y, n_bins=15):
    p = torch.softmax(scaled_logits, dim=-1)
    conf, pred = p.max(dim=-1)
    correct = (pred == y).float().numpy(); conf = conf.numpy()
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for bb in range(n_bins):
        mm = (conf > bins[bb]) & (conf <= bins[bb+1])
        if mm.sum() == 0: continue
        e += mm.mean() * abs(correct[mm].mean() - conf[mm].mean())
    return e

# ---------------------------------------------------------------------------
# Evaluate all three methods on in-domain and shifted test sets.
# ---------------------------------------------------------------------------
def eval_all(logits, y, name):
    acc = (logits.argmax(-1) == y).float().mean().item()
    out = {"split": name, "acc": acc}
    out["raw_nll"]  = nll(logits, y, 1.0);      out["raw_ece"]  = ece(logits, y, 1.0)
    out["glob_nll"] = nll(logits, y, T_global); out["glob_ece"] = ece(logits, y, T_global)
    sl = adaptive_apply(logits)
    out["adap_nll"] = nll_from_probs(sl, y);    out["adap_ece"] = ece_from_probs(sl, y)
    return out

results = {
    "T_global": T_global,
    "bin_T": bin_T.tolist(),
    "entropy_edges": edges.tolist(),
    "in_domain":  eval_all(test_logits, test_y, "in_domain"),
    "shift":      eval_all(shift_logits, shift_y, "shift"),
    "val":        eval_all(val_logits, val_y, "val"),
    "runtime_s":  time.time() - t0,
}

print(json.dumps(results, indent=2))
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"TOTAL {time.time()-t0:.1f}s")
