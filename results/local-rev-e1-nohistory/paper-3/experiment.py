"""
Synthetic study of probabilistic calibration in a tiny transformer LM.

We construct a controlled generative process with KNOWN ground-truth
aleatoric uncertainty (some "keys" map deterministically to a value,
others map stochastically according to a known categorical distribution),
and KNOWN epistemic exposure (each key is seen a different number of times
during training, some keys are held out entirely).

We then ask:
  (A) Does the trained model's predictive distribution match the true
      generating distribution (calibration), and does this degrade with
      lower training frequency (epistemic contamination of aleatoric
      estimates)?
  (B) How many Monte-Carlo samples are needed for sampling-based entropy
      estimates (as used in semantic-entropy-style UQ) to match the exact
      analytic entropy computable from the softmax -- i.e. what is the
      sample-efficiency cost of the "expensive" black-box approach vs the
      "cheap" white-box (single forward pass) approach?
  (C) Can single-pass entropy / max-prob detect prediction errors
      (AUROC), and does that detection quality depend on epistemic
      exposure (training frequency)?

All results are written to results.json and results.log in this directory.
"""
import json, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t_start = time.time()
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)

# ---------------------------------------------------------------------------
# 1. Vocabulary & synthetic generative process
# ---------------------------------------------------------------------------
N_KEYS = 60
N_VALS = 8
BOS, SEP, EOS, PAD = 0, 1, 2, 3
KEY_BASE = 4
VAL_BASE = KEY_BASE + N_KEYS
VOCAB_SIZE = VAL_BASE + N_VALS

def key_tok(i): return KEY_BASE + i
def val_tok(j): return VAL_BASE + j

# Frequency tiers (training exposure count) -> epistemic axis
FREQ_TIERS = {"unseen": 0, "rare": 3, "medium": 20, "frequent": 150}
tier_names = list(FREQ_TIERS.keys())

# Assign each key: type (deterministic / stochastic), true distribution,
# and frequency tier. Split evenly across tiers and types.
keys_meta = {}
tier_cycle = (tier_names * (N_KEYS // len(tier_names) + 1))[:N_KEYS]
random.shuffle(tier_cycle)
for i in range(N_KEYS):
    tier = tier_cycle[i]
    is_stochastic = (i % 2 == 0)
    cand_vals = random.sample(range(N_VALS), 3)
    if is_stochastic:
        # known categorical distribution over 3 candidate values
        raw = np.array([0.6, 0.3, 0.1])
        np.random.shuffle(raw)
        dist = {v: float(p) for v, p in zip(cand_vals, raw)}
    else:
        dist = {cand_vals[0]: 1.0}
    keys_meta[i] = dict(type="stochastic" if is_stochastic else "deterministic",
                         tier=tier, freq=FREQ_TIERS[tier], dist=dist)

def sample_value(i):
    dist = keys_meta[i]["dist"]
    vs, ps = zip(*dist.items())
    return int(np.random.choice(vs, p=ps))

# ---------------------------------------------------------------------------
# 2. Build training corpus: sequence = BOS key SEP val EOS  (len 5)
# ---------------------------------------------------------------------------
SEQ_LEN = 5
train_seqs = []
for i in range(N_KEYS):
    for _ in range(keys_meta[i]["freq"]):
        v = sample_value(i)
        train_seqs.append([BOS, key_tok(i), SEP, val_tok(v), EOS])

random.shuffle(train_seqs)
train_tensor = torch.tensor(train_seqs, dtype=torch.long)
log(f"Training examples: {len(train_seqs)}  (keys={N_KEYS}, vals={N_VALS}, vocab={VOCAB_SIZE})")
for t in tier_names:
    n = sum(1 for i in keys_meta if keys_meta[i]["tier"] == t)
    log(f"  tier={t:9s} freq={FREQ_TIERS[t]:4d}  n_keys={n}")

# ---------------------------------------------------------------------------
# 3. Tiny causal transformer LM
# ---------------------------------------------------------------------------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, nlayers=2, dim_ff=128, max_len=SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1).to(x.device)
        h = self.enc(h, mask=mask)
        return self.out(h)

device = "cpu"
model = TinyTransformerLM(VOCAB_SIZE).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

BATCH = 128
EPOCHS = 25
n = train_tensor.shape[0]
log(f"Model params: {sum(p.numel() for p in model.parameters())}")

for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot_loss, nb = 0.0, 0
    for s in range(0, n, BATCH):
        idx = perm[s:s+BATCH]
        batch = train_tensor[idx].to(device)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item(); nb += 1
    if ep % 5 == 0 or ep == EPOCHS - 1:
        log(f"epoch {ep:3d}  loss {tot_loss/nb:.4f}  elapsed {time.time()-t_start:.1f}s")

log(f"Training done at {time.time()-t_start:.1f}s")

# ---------------------------------------------------------------------------
# 4. Experiment A: calibration to true aleatoric distribution
# ---------------------------------------------------------------------------
model.eval()

def analytic_value_dist(key_i):
    """Exact softmax over value tokens at the prediction position, one fwd pass."""
    seq = torch.tensor([[BOS, key_tok(key_i), SEP]], dtype=torch.long)
    with torch.no_grad():
        logits = model(seq)[0, -1]  # predict token after SEP
    val_logits = logits[VAL_BASE:VAL_BASE + N_VALS]
    probs = F.softmax(val_logits, dim=-1).numpy()
    return probs

def true_dist_vec(key_i):
    d = keys_meta[key_i]["dist"]
    v = np.zeros(N_VALS)
    for k_, p_ in d.items():
        v[k_] = p_
    return v

def kl(p, q, eps=1e-9):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())

def entropy(p, eps=1e-9):
    p = p + eps
    return float(-np.sum(p * np.log(p)))

rows = []
for i in range(N_KEYS):
    model_p = analytic_value_dist(i)
    true_p = true_dist_vec(i)
    rows.append(dict(
        key=i, type=keys_meta[i]["type"], tier=keys_meta[i]["tier"], freq=keys_meta[i]["freq"],
        kl_true_model=kl(true_p, model_p), tv=tv(true_p, model_p),
        true_entropy=entropy(true_p), model_entropy=entropy(model_p),
        model_top_p=float(model_p.max()), model_argmax=int(model_p.argmax()),
        true_argmax=int(true_p.argmax()),
    ))

import statistics as st
log("\n=== Experiment A: calibration vs training frequency tier ===")
summaryA = {}
for typ in ["deterministic", "stochastic"]:
    for tier in tier_names:
        sub = [r for r in rows if r["type"] == typ and r["tier"] == tier]
        if not sub: continue
        mkl = st.mean(r["kl_true_model"] for r in sub)
        mtv = st.mean(r["tv"] for r in sub)
        mte = st.mean(r["true_entropy"] for r in sub)
        mme = st.mean(r["model_entropy"] for r in sub)
        acc = st.mean(1.0 if r["model_argmax"] == r["true_argmax"] else 0.0 for r in sub)
        key = f"{typ}/{tier}"
        summaryA[key] = dict(n=len(sub), mean_kl=mkl, mean_tv=mtv, mean_true_entropy=mte,
                              mean_model_entropy=mme, argmax_acc=acc)
        log(f"  {key:22s} n={len(sub):2d}  KL(true||model)={mkl:.3f}  TV={mtv:.3f}  "
            f"H_true={mte:.3f}  H_model={mme:.3f}  argmax_acc={acc:.2f}")

# ---------------------------------------------------------------------------
# 5. Experiment B: Monte-Carlo sample efficiency vs analytic entropy
# ---------------------------------------------------------------------------
log("\n=== Experiment B: MC-sampled entropy vs analytic (single forward pass) entropy ===")
sample_sizes = [5, 20, 100, 400]
N_TRIALS = 30
mc_keys = [i for i in range(N_KEYS) if keys_meta[i]["tier"] in ("medium", "frequent")]

def mc_sample_entropy(key_i, T):
    probs = analytic_value_dist(key_i)  # generation is single-token here, so we sample from the exact
    # categorical the model defines, exactly mirroring autoregressive sampling.
    samples = np.random.choice(N_VALS, size=T, p=probs)
    counts = np.bincount(samples, minlength=N_VALS).astype(float)
    emp_p = counts / counts.sum()
    return entropy(emp_p)

summaryB = {}
for T in sample_sizes:
    errs = []
    for i in mc_keys:
        h_true = entropy(analytic_value_dist(i))
        trial_hs = [mc_sample_entropy(i, T) for _ in range(N_TRIALS)]
        errs.extend([abs(h - h_true) for h in trial_hs])
    mae = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    summaryB[T] = dict(mae=mae, rmse=rmse)
    log(f"  T={T:4d} samples  MAE(entropy)={mae:.4f}  RMSE={rmse:.4f}  "
        f"(analytic entropy needs 1 forward pass, 0 sampling error)")

# ---------------------------------------------------------------------------
# 6. Experiment C: error detection AUROC (deterministic keys only)
# ---------------------------------------------------------------------------
log("\n=== Experiment C: AUROC of uncertainty scores for detecting argmax errors (deterministic keys) ===")
det_rows = [r for r in rows if r["type"] == "deterministic"]
y_err = [1 if r["model_argmax"] != r["true_argmax"] else 0 for r in det_rows]
scores_entropy = [r["model_entropy"] for r in det_rows]
scores_negmaxp = [1 - r["model_top_p"] for r in det_rows]
summaryC = {}
if len(set(y_err)) > 1:
    auroc_ent = roc_auc_score(y_err, scores_entropy)
    auroc_negmaxp = roc_auc_score(y_err, scores_negmaxp)
    summaryC = dict(n_det_keys=len(det_rows), n_errors=sum(y_err),
                     auroc_entropy=float(auroc_ent), auroc_1_minus_maxprob=float(auroc_negmaxp))
    log(f"  n_det_keys={len(det_rows)} n_errors={sum(y_err)}  "
        f"AUROC(entropy)={auroc_ent:.3f}  AUROC(1-maxprob)={auroc_negmaxp:.3f}")
else:
    log(f"  n_det_keys={len(det_rows)} n_errors={sum(y_err)} -- degenerate (all correct or all wrong), AUROC undefined")
    summaryC = dict(n_det_keys=len(det_rows), n_errors=sum(y_err), note="degenerate, AUROC undefined")

# also break down accuracy of deterministic keys by tier for context
log("\n  deterministic-key argmax accuracy by tier:")
for tier in tier_names:
    sub = [r for r in det_rows if r["tier"] == tier]
    if not sub: continue
    acc = st.mean(1.0 if r["model_argmax"] == r["true_argmax"] else 0.0 for r in sub)
    log(f"    {tier:9s} n={len(sub):2d} acc={acc:.2f}")

# ---------------------------------------------------------------------------
# 7. Save everything
# ---------------------------------------------------------------------------
elapsed = time.time() - t_start
log(f"\nTotal elapsed: {elapsed:.1f}s")

out = dict(
    n_keys=N_KEYS, n_vals=N_VALS, vocab_size=VOCAB_SIZE,
    freq_tiers=FREQ_TIERS, n_train_examples=len(train_seqs),
    model_params=sum(p.numel() for p in model.parameters()),
    epochs=EPOCHS, elapsed_seconds=elapsed,
    per_key_rows=rows,
    experimentA_summary=summaryA,
    experimentB_summary=summaryB,
    experimentC_summary=summaryC,
)
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
with open("results.log", "w") as f:
    f.write("\n".join(LOG))

print("\nSaved results.json and results.log")
