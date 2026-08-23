"""
Synthetic testbed for evaluating LLM-style probabilistic calibration methods
against an EXACT, analytically-known ground-truth predictive distribution.

Generative process: a latent-mode Markov mixture. Each sequence is generated
by first drawing a hidden mode m ~ Uniform{1..K}, then emitting tokens via a
first-order Markov chain with transition matrix T_m. The mode is never
observed by the learner. Because we know all T_m and the prior over modes,
we can compute the exact Bayesian posterior predictive distribution
p_true(x_{t+1} | x_1..t) via a forward (HMM-filtering) recursion. This lets
us measure calibration error against ground truth rather than against a
finite-sample proxy, which is the key advantage of this synthetic setup over
real-LLM calibration studies.

We train a small causal Transformer as a next-token predictor on sampled
sequences (mode hidden), then compare its predictive distribution to
p_true, before and after two standard LLM calibration techniques:
  (1) temperature scaling (single scalar, fit by NLL on a val split)
  (2) split conformal prediction (distribution-free coverage guarantee)

All computation is CPU, from scratch, no downloaded weights.
"""
import time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------- Generative process ----------------
V = 10          # vocab size
K = 3           # number of latent modes
T_LEN = 40      # sequence length
N_TRAIN, N_VAL, N_CAL, N_TEST = 4000, 800, 800, 1200

def sample_transition_matrix(v, alpha, rng):
    return rng.dirichlet(alpha * np.ones(v), size=v)  # rows sum to 1

# make modes reasonably distinguishable: skewed dirichlet
alphas = [0.3, 0.3, 0.3]
mode_T = [sample_transition_matrix(V, a, rng) for a in alphas]
mode_prior = np.ones(K) / K

def gen_sequence(rng):
    m = rng.integers(K)
    x = [rng.integers(V)]
    for _ in range(T_LEN - 1):
        p = mode_T[m][x[-1]]
        x.append(rng.choice(V, p=p))
    return np.array(x, dtype=np.int64), m

def gen_dataset(n, rng):
    seqs, modes = [], []
    for _ in range(n):
        x, m = gen_sequence(rng)
        seqs.append(x); modes.append(m)
    return np.stack(seqs), np.array(modes)

train_x, _ = gen_dataset(N_TRAIN, rng)
val_x, _   = gen_dataset(N_VAL, rng)
cal_x, _   = gen_dataset(N_CAL, rng)
test_x, _  = gen_dataset(N_TEST, rng)

# ---------------- Exact posterior predictive (ground truth) ----------------
def true_predictive_probs(x):
    """For a sequence x (length T_LEN), return array [T_LEN-1, V] of exact
    p_true(x_{t+1} | x_1..t) for t=1..T_LEN-1 (0-indexed positions 0..T-2)."""
    T = len(x)
    post = mode_prior.copy()  # p(mode | x_1) = prior (no info from single token)
    out = np.zeros((T - 1, V))
    for t in range(T - 1):
        # predictive at this step uses current posterior over modes given x_1..t
        pred = np.zeros(V)
        for m in range(K):
            pred += post[m] * mode_T[m][x[t]]
        out[t] = pred
        # observe x_{t+1}, update posterior over modes
        lik = np.array([mode_T[m][x[t], x[t + 1]] for m in range(K)])
        post = post * lik
        post = post / post.sum()
    return out

def batch_true_predictive(X):
    return np.stack([true_predictive_probs(x) for x in X])  # [N, T-1, V]

print(f"[{time.time()-t_start:.1f}s] generated data + ground truth setup")

# ---------------- Small causal Transformer ----------------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=48, nhead=4, nlayers=2, max_len=T_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                            dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)  # [B, T, V] logits predicting next token at each pos

device = "cpu"
model = TinyTransformerLM(V).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

train_t = torch.from_numpy(train_x).to(device)
val_t = torch.from_numpy(val_x).to(device)

def train_epoch_step(batch_x):
    model.train()
    inp = batch_x[:, :-1]
    tgt = batch_x[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

BATCH = 64
N_STEPS = 600
for step in range(N_STEPS):
    idx = rng.choice(N_TRAIN, BATCH, replace=False)
    b = train_t[idx]
    loss = train_epoch_step(b)
    if step % 100 == 0 or step == N_STEPS - 1:
        print(f"[{time.time()-t_start:.1f}s] step {step} train_loss {loss:.4f}")

print(f"[{time.time()-t_start:.1f}s] training done")

# ---------------- Get model predictive distributions ----------------
@torch.no_grad()
def model_predictive_probs(X, T_scale=1.0):
    model.eval()
    xt = torch.from_numpy(X).to(device)
    inp = xt[:, :-1]
    logits = model(inp) / T_scale
    probs = F.softmax(logits, dim=-1).numpy()  # [N, T-1, V]
    return probs

val_true = batch_true_predictive(val_x)
val_model_raw = model_predictive_probs(val_x, 1.0)

# ---------------- Temperature scaling (fit on val, minimize NLL vs actual next tokens) ----------------
def nll_for_T(Tval, X):
    xt = torch.from_numpy(X)
    inp = xt[:, :-1]; tgt = xt[:, 1:]
    with torch.no_grad():
        logits = model(inp) / Tval
        logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp.reshape(-1, V), tgt.reshape(-1))
    return nll.item()

Ts = np.linspace(0.5, 2.5, 41)
nlls = [nll_for_T(float(Tv), val_x) for Tv in Ts]
best_T = float(Ts[int(np.argmin(nlls))])
print(f"[{time.time()-t_start:.1f}s] best temperature T={best_T:.3f} (val NLL={min(nlls):.4f} vs T=1 NLL={nlls[list(Ts).index(1.0) if 1.0 in Ts else 20]:.4f})")

# ---------------- Evaluation on test set ----------------
test_true = batch_true_predictive(test_x)          # [N, T-1, V]
test_model_raw = model_predictive_probs(test_x, 1.0)
test_model_scaled = model_predictive_probs(test_x, best_T)

def kl_div(p, q, eps=1e-9):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)

def true_ece(model_probs, true_probs, tokens_next, n_bins=10):
    """'Oracle' ECE: confidence = model's predicted prob of the token it
    considers most likely; true correctness rate replaced by the EXACT true
    probability that a token equals argmax under p_true (soft correctness),
    binned by model confidence. Since we know p_true exactly we use it
    directly rather than a finite-sample empirical accuracy."""
    conf = model_probs.max(axis=-1)                    # model's top confidence
    top_tok = model_probs.argmax(axis=-1)
    true_prob_of_top = np.take_along_axis(true_probs, top_tok[..., None], axis=-1).squeeze(-1)
    conf = conf.flatten(); true_prob_of_top = true_prob_of_top.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / N) * abs(conf[m].mean() - true_prob_of_top[m].mean())
    return ece

def position_bucket_metrics(model_probs, true_probs, label):
    kl = kl_div(model_probs, true_probs)  # [N, T-1]
    early = kl[:, :10].mean(); late = kl[:, 10:].mean(); overall = kl.mean()
    print(f"  {label}: KL early(t<10)={early:.4f} late(t>=10)={late:.4f} overall={overall:.4f}")
    return dict(early=float(early), late=float(late), overall=float(overall))

print(f"[{time.time()-t_start:.1f}s] === KL(true || model) : raw vs temperature-scaled ===")
kl_raw = position_bucket_metrics(test_model_raw, test_true, "raw (T=1)")
kl_scaled = position_bucket_metrics(test_model_scaled, test_true, f"scaled (T={best_T:.2f})")

ece_raw = true_ece(test_model_raw, test_true, None)
ece_scaled = true_ece(test_model_scaled, test_true, None)
print(f"[{time.time()-t_start:.1f}s] Oracle-ECE raw={ece_raw:.4f} scaled={ece_scaled:.4f}")

# ---------------- Split conformal prediction ----------------
def conformal_calibrate(cal_x, target_cov=0.9, T_scale=1.0):
    probs = model_predictive_probs(cal_x, T_scale)   # [N, T-1, V]
    xt = cal_x[:, 1:]
    N, Tm1 = xt.shape
    scores = 1.0 - np.take_along_axis(probs, xt[..., None], axis=-1).squeeze(-1)
    scores = scores.flatten()
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * target_cov) / n)
    qhat = np.quantile(scores, q_level)
    return qhat

def conformal_eval(test_x, qhat, T_scale=1.0):
    probs = model_predictive_probs(test_x, T_scale)
    xt = test_x[:, 1:]
    true_probs_here = batch_true_predictive(test_x)
    set_mask = probs >= (1 - qhat)   # prediction set: tokens with score <= qhat
    set_size = set_mask.sum(axis=-1).mean()
    covered = np.take_along_axis(set_mask, xt[..., None], axis=-1).squeeze(-1)
    coverage = covered.mean()
    # oracle: what set size would exact p_true need for same target coverage,
    # using an ideal conformal-style threshold on p_true itself
    return float(coverage), float(set_size)

target_cov = 0.9
qhat_raw = conformal_calibrate(cal_x, target_cov, 1.0)
qhat_scaled = conformal_calibrate(cal_x, target_cov, best_T)
cov_raw, size_raw = conformal_eval(test_x, qhat_raw, 1.0)
cov_scaled, size_scaled = conformal_eval(test_x, qhat_scaled, best_T)
print(f"[{time.time()-t_start:.1f}s] Conformal target={target_cov}: "
      f"raw coverage={cov_raw:.3f} size={size_raw:.2f} | "
      f"scaled coverage={cov_scaled:.3f} size={size_scaled:.2f}")

# baseline: uniform-random predictor (sanity floor) and true-distribution predictor (ceiling)
uniform_probs = np.ones_like(test_model_raw) / V
kl_uniform = position_bucket_metrics(uniform_probs, test_true, "uniform baseline")
kl_oracle = position_bucket_metrics(test_true, test_true, "oracle (true dist as predictor)")

results = dict(
    V=V, K=K, T_LEN=T_LEN, N_STEPS=N_STEPS, best_T=best_T,
    kl_raw=kl_raw, kl_scaled=kl_scaled, kl_uniform=kl_uniform, kl_oracle=kl_oracle,
    ece_raw=float(ece_raw), ece_scaled=float(ece_scaled),
    conformal=dict(target=target_cov,
                    raw=dict(coverage=cov_raw, size=size_raw, qhat=float(qhat_raw)),
                    scaled=dict(coverage=cov_scaled, size=size_scaled, qhat=float(qhat_scaled))),
    elapsed_sec=time.time() - t_start,
)
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[{time.time()-t_start:.1f}s] DONE. Results saved to results.json")
print(json.dumps(results, indent=2))
