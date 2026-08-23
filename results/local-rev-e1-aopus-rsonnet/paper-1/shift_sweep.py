"""Q2: characterize how TS shift-robustness scales with shift magnitude.
Fix tau_in=1.0 (regime-A base source), sweep tau_out, measure global-TS
shift-ECE reduction and residual adaptivity gain. 3 seeds per point."""
import time, json
import numpy as np, torch, torch.nn as nn

t0 = time.time()
V = 20
CTX, EMB, HID = 8, 48, 128
N_TRAIN, N_VAL, N_TEST = 15000, 12000, 12000
EPOCHS, BS = 18, 128
N_EBINS = 8

def make_source(seed, tau):
    r = np.random.default_rng(seed)
    logits = r.normal(0, 1.0, size=(V, V, V)) / tau
    P = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return P / P.sum(axis=-1, keepdims=True)

def sample_text(P, n, seed):
    r = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.int64)
    x[0] = r.integers(V); x[1] = r.integers(V)
    for i in range(2, n):
        x[i] = r.choice(V, p=P[x[i-1], x[i-2]])
    return x

class CharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.lstm = nn.LSTM(EMB, HID, batch_first=True)
        self.head = nn.Linear(HID, V)
    def forward(self, x):
        e = self.emb(x); o, _ = self.lstm(e); return self.head(o[:, -1, :])

def make_batches(seq, ctx, bs, gen):
    X, Y = [], []
    for i in range(ctx, len(seq)):
        X.append(seq[i-ctx:i]); Y.append(seq[i])
    X = torch.tensor(np.array(X)); Y = torch.tensor(np.array(Y))
    idx = torch.randperm(len(X), generator=gen); X, Y = X[idx], Y[idx]
    for j in range(0, len(X), bs):
        yield X[j:j+bs], Y[j:j+bs]

def nll(logits, y, T=1.0):
    return nn.functional.cross_entropy(logits / T, y).item()
def probs(logits, T=1.0):
    return torch.softmax(logits / T, dim=-1)
def entropy(logits, T=1.0):
    p = probs(logits, T); return (-(p * torch.log(p + 1e-12)).sum(-1))
def ece_arr(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b+1])
        if m.sum() == 0: continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e
def fit_global_T(logits, y):
    Ts = np.linspace(0.5, 14.0, 136)
    best = min(Ts, key=lambda T: nll(logits, y, T))
    fine = np.linspace(max(0.05, best-0.1), best+0.1, 21)
    return float(min(fine, key=lambda T: nll(logits, y, T)))
def collect(model, seq):
    model.eval(); Xs, Ys = [], []
    for i in range(CTX, len(seq)):
        Xs.append(seq[i-CTX:i]); Ys.append(seq[i])
    X = torch.tensor(np.array(Xs)); Y = torch.tensor(np.array(Ys))
    with torch.no_grad():
        logits = [model(X[j:j+1024]) for j in range(0, len(X), 1024)]
    return torch.cat(logits), Y

def run_one(seed, tau_in, tau_out):
    np.random.seed(seed); torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    P_in  = make_source(seed=1 + 100*seed, tau=tau_in)
    P_out = make_source(seed=2 + 100*seed, tau=tau_out)
    train = sample_text(P_in, N_TRAIN, seed=10 + 100*seed)
    val   = sample_text(P_in, N_VAL,   seed=11 + 100*seed)
    shift = sample_text(P_out, N_TEST, seed=13 + 100*seed)
    model = CharLM(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss(); model.train()
    for ep in range(EPOCHS):
        for xb, yb in make_batches(train, CTX, BS, gen):
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    val_logits, val_y = collect(model, val)
    shift_logits, shift_y = collect(model, shift)
    T_global = fit_global_T(val_logits, val_y)
    val_H = entropy(val_logits).numpy()
    edges = np.quantile(val_H, np.linspace(0, 1, N_EBINS + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    def bin_of(H): return np.clip(np.digitize(H, edges) - 1, 0, N_EBINS - 1)
    bin_T = np.ones(N_EBINS); vb = bin_of(val_H)
    for b in range(N_EBINS):
        m = vb == b
        bin_T[b] = T_global if m.sum() < 20 else fit_global_T(val_logits[m], val_y[m])
    def adap(logits):
        b = bin_of(entropy(logits).numpy())
        return logits / torch.tensor(bin_T[b], dtype=torch.float32).unsqueeze(-1)
    def ece_of(sl, y):
        p = torch.softmax(sl, dim=-1); conf, pred = p.max(dim=-1)
        return ece_arr(conf.numpy(), (pred == y).float().numpy())
    raw = ece_of(shift_logits, shift_y)
    glob = ece_of(shift_logits / T_global, shift_y)
    adp = ece_of(adap(shift_logits), shift_y)
    acc = (shift_logits.argmax(-1) == shift_y).float().mean().item()
    return raw, glob, adp, acc

TAUS = [1.0, 1.15, 1.3, 1.6, 2.2, 3.2]
SEEDS = [0, 1, 2]
rows = []
for to in TAUS:
    raws, globs, adps, accs = [], [], [], []
    for s in SEEDS:
        raw, glob, adp, acc = run_one(s, 1.0, to)
        raws.append(raw); globs.append(glob); adps.append(adp); accs.append(acc)
    raw_m, glob_m, adp_m = np.mean(raws), np.mean(globs), np.mean(adps)
    red = 100 * (raw_m - glob_m) / raw_m
    rows.append({"tau_out": to, "shift_acc": float(np.mean(accs)),
                 "raw_ece": float(raw_m), "glob_ece": float(glob_m),
                 "glob_ece_sd": float(np.std(globs)),
                 "adap_ece": float(adp_m), "pct_reduction": float(red),
                 "adap_minus_glob": float(adp_m - glob_m)})
    print(f"tau_out={to:4.2f} acc={np.mean(accs):.3f} raw={raw_m:.3f} glob={glob_m:.4f} "
          f"reduction={red:5.1f}% adap-glob={adp_m-glob_m:+.5f}  [t={time.time()-t0:.0f}s]")
json.dump(rows, open("shift_sweep.json", "w"), indent=2)
print(f"TOTAL {time.time()-t0:.1f}s")
