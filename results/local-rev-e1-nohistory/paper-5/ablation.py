"""
Ablation: how does the raw-vs-temperature-scaled gap (KL-to-truth, oracle-ECE)
evolve with training budget? Same generative process/seed as experiment.py,
trained for varying numbers of optimizer steps.
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

V, K, T_LEN = 10, 3, 40
N_TRAIN, N_VAL, N_CAL, N_TEST = 4000, 800, 800, 1200

def sample_transition_matrix(v, alpha, rng):
    return rng.dirichlet(alpha * np.ones(v), size=v)

alphas = [0.3, 0.3, 0.3]
mode_T = [sample_transition_matrix(V, a, rng) for a in alphas]
mode_prior = np.ones(K) / K

def gen_sequence(rng):
    m = rng.integers(K)
    x = [rng.integers(V)]
    for _ in range(T_LEN - 1):
        p = mode_T[m][x[-1]]
        x.append(rng.choice(V, p=p))
    return np.array(x, dtype=np.int64)

def gen_dataset(n, rng):
    return np.stack([gen_sequence(rng) for _ in range(n)])

train_x = gen_dataset(N_TRAIN, rng)
val_x = gen_dataset(N_VAL, rng)
test_x = gen_dataset(N_TEST, rng)

def true_predictive_probs(x):
    T = len(x)
    post = mode_prior.copy()
    out = np.zeros((T - 1, V))
    for t in range(T - 1):
        pred = np.zeros(V)
        for m in range(K):
            pred += post[m] * mode_T[m][x[t]]
        out[t] = pred
        lik = np.array([mode_T[m][x[t], x[t + 1]] for m in range(K)])
        post = post * lik
        post = post / post.sum()
    return out

def batch_true_predictive(X):
    return np.stack([true_predictive_probs(x) for x in X])

test_true = batch_true_predictive(test_x)
print(f"[{time.time()-t_start:.1f}s] data ready")

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=48, nhead=4, nlayers=2, max_len=T_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                            dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)

def kl_div(p, q, eps=1e-9):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)

def true_ece(model_probs, true_probs, n_bins=10):
    conf = model_probs.max(axis=-1)
    top_tok = model_probs.argmax(axis=-1)
    true_prob_of_top = np.take_along_axis(true_probs, top_tok[..., None], axis=-1).squeeze(-1)
    conf = conf.flatten(); true_prob_of_top = true_prob_of_top.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / N) * abs(conf[m].mean() - true_prob_of_top[m].mean())
    return float(ece)

@torch.no_grad()
def model_predictive_probs(model, X, T_scale=1.0):
    model.eval()
    xt = torch.from_numpy(X)
    inp = xt[:, :-1]
    logits = model(inp) / T_scale
    return F.softmax(logits, dim=-1).numpy()

def nll_for_T(model, Tval, X):
    xt = torch.from_numpy(X)
    inp = xt[:, :-1]; tgt = xt[:, 1:]
    with torch.no_grad():
        logits = model(inp) / Tval
        logp = F.log_softmax(logits, dim=-1)
        return F.nll_loss(logp.reshape(-1, V), tgt.reshape(-1)).item()

STEP_BUDGETS = [50, 150, 300, 600, 1200]
BATCH = 64
results = []
for total_steps in STEP_BUDGETS:
    torch.manual_seed(SEED)
    model = TinyTransformerLM(V)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    train_t = torch.from_numpy(train_x)
    for step in range(total_steps):
        idx = rng.choice(N_TRAIN, BATCH, replace=False)
        b = train_t[idx]
        model.train()
        inp = b[:, :-1]; tgt = b[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    Ts = np.linspace(0.5, 2.5, 21)
    nlls = [nll_for_T(model, float(Tv), val_x) for Tv in Ts]
    best_T = float(Ts[int(np.argmin(nlls))])

    raw = model_predictive_probs(model, test_x, 1.0)
    scaled = model_predictive_probs(model, test_x, best_T)
    kl_raw = float(kl_div(raw, test_true).mean())
    kl_scaled = float(kl_div(scaled, test_true).mean())
    ece_raw = true_ece(raw, test_true)
    ece_scaled = true_ece(scaled, test_true)
    train_nll_final = loss.item()

    row = dict(steps=total_steps, best_T=best_T, train_loss=train_nll_final,
               kl_raw=kl_raw, kl_scaled=kl_scaled, ece_raw=ece_raw, ece_scaled=ece_scaled)
    results.append(row)
    print(f"[{time.time()-t_start:.1f}s] steps={total_steps:5d} T*={best_T:.2f} "
          f"train_loss={train_nll_final:.3f} KL raw/scaled={kl_raw:.4f}/{kl_scaled:.4f} "
          f"ECE raw/scaled={ece_raw:.4f}/{ece_scaled:.4f}")

with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[{time.time()-t_start:.1f}s] DONE")
