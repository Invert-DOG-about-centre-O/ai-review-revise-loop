"""
Robustness experiments for the revision:
  (A) multi-seed variance / bootstrap CIs for the ECE comparisons
  (B) a HIGHER-SKILL source regime (sharper Markov transitions) so the model
      has real predictive structure, well above the 5% chance floor, to test
      whether the shift-robustness and ECATS-null findings survive.

Reuses the exact model/calibration logic of experiment.py.
"""
import time, json
import numpy as np
import torch
import torch.nn as nn

t0 = time.time()
V, ORDER = 20, 2
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
        e = self.emb(x)
        o, _ = self.lstm(e)
        return self.head(o[:, -1, :])

def make_batches(seq, ctx, bs, gen):
    X, Y = [], []
    for i in range(ctx, len(seq)):
        X.append(seq[i-ctx:i]); Y.append(seq[i])
    X = torch.tensor(np.array(X)); Y = torch.tensor(np.array(Y))
    idx = torch.randperm(len(X), generator=gen)
    X, Y = X[idx], Y[idx]
    for j in range(0, len(X), bs):
        yield X[j:j+bs], Y[j:j+bs]

def nll(logits, y, T=1.0):
    return nn.functional.cross_entropy(logits / T, y).item()
def probs(logits, T=1.0):
    return torch.softmax(logits / T, dim=-1)
def entropy(logits, T=1.0):
    p = probs(logits, T)
    return (-(p * torch.log(p + 1e-12)).sum(-1))
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
    model.eval()
    Xs, Ys = [], []
    for i in range(CTX, len(seq)):
        Xs.append(seq[i-CTX:i]); Ys.append(seq[i])
    X = torch.tensor(np.array(Xs)); Y = torch.tensor(np.array(Ys))
    with torch.no_grad():
        logits = [model(X[j:j+1024]) for j in range(0, len(X), 1024)]
    return torch.cat(logits), Y

def bootstrap_ece_ci(conf, correct, n_boot=300, seed=0):
    r = np.random.default_rng(seed); n = len(conf); vals = []
    for _ in range(n_boot):
        idx = r.integers(0, n, n)
        vals.append(ece_arr(conf[idx], correct[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def run_one(seed, tau_in, tau_out):
    np.random.seed(seed); torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    P_in  = make_source(seed=1 + 100*seed, tau=tau_in)
    P_out = make_source(seed=2 + 100*seed, tau=tau_out)
    train = sample_text(P_in, N_TRAIN, seed=10 + 100*seed)
    val   = sample_text(P_in, N_VAL,   seed=11 + 100*seed)
    test  = sample_text(P_in, N_TEST,  seed=12 + 100*seed)
    test_shift = sample_text(P_out, N_TEST, seed=13 + 100*seed)

    model = CharLM()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in make_batches(train, CTX, BS, gen):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()

    val_logits, val_y   = collect(model, val)
    test_logits, test_y = collect(model, test)
    shift_logits, shift_y = collect(model, test_shift)

    T_global = fit_global_T(val_logits, val_y)
    val_H = entropy(val_logits).numpy()
    edges = np.quantile(val_H, np.linspace(0, 1, N_EBINS + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    def bin_of(H): return np.clip(np.digitize(H, edges) - 1, 0, N_EBINS - 1)
    bin_T = np.ones(N_EBINS)
    val_bins = bin_of(val_H)
    for b in range(N_EBINS):
        m = val_bins == b
        bin_T[b] = T_global if m.sum() < 20 else fit_global_T(val_logits[m], val_y[m])
    def adap(logits):
        b = bin_of(entropy(logits).numpy())
        return logits / torch.tensor(bin_T[b], dtype=torch.float32).unsqueeze(-1)

    def split_stats(logits, y):
        acc = (logits.argmax(-1) == y).float().mean().item()
        out = {"acc": acc}
        for name, sl in [("raw", logits), ("glob", logits / T_global), ("adap", adap(logits))]:
            p = torch.softmax(sl, dim=-1)
            conf, pred = p.max(dim=-1)
            conf = conf.numpy(); correct = (pred == y).float().numpy()
            out[name + "_ece"] = ece_arr(conf, correct)
            if name in ("glob", "adap"):
                lo, hi = bootstrap_ece_ci(conf, correct, seed=seed)
                out[name + "_ci"] = [lo, hi]
        return out

    return {
        "seed": seed, "T_global": T_global,
        "bin_T_min": float(bin_T.min()), "bin_T_max": float(bin_T.max()),
        "in_domain": split_stats(test_logits, test_y),
        "shift": split_stats(shift_logits, shift_y),
    }

def summarize(runs, key):
    def col(split, m): return np.array([r[split][m] for r in runs])
    s = {}
    for split in ("in_domain", "shift"):
        s[split] = {"acc_mean": float(col(split, "acc").mean())}
        for m in ("raw_ece", "glob_ece", "adap_ece"):
            a = col(split, m)
            s[split][m] = [float(a.mean()), float(a.std())]
        # paired glob vs adap difference across seeds
        d = col(split, "adap_ece") - col(split, "glob_ece")
        s[split]["adap_minus_glob_mean"] = float(d.mean())
        s[split]["adap_minus_glob_std"] = float(d.std())
    return s

SEEDS = [0, 1, 2, 3, 4]
print("=== REGIME A: original overconfident source (tau_in=1.0, tau_out=1.6) ===")
runs_A = [run_one(s, 1.0, 1.6) for s in SEEDS]
sumA = summarize(runs_A, "A")
print(json.dumps(sumA, indent=2))
print(f"[t={time.time()-t0:.0f}s]")

print("=== REGIME B: higher-skill sharper source (tau_in=0.45, tau_out=0.7) ===")
runs_B = [run_one(s, 0.45, 0.7) for s in SEEDS]
sumB = summarize(runs_B, "B")
print(json.dumps(sumB, indent=2))

out = {
    "regime_A_overconfident": {"summary": sumA, "runs": runs_A},
    "regime_B_higher_skill": {"summary": sumB, "runs": runs_B},
    "runtime_s": time.time() - t0,
}
with open("results_robust.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"TOTAL {time.time()-t0:.1f}s")
