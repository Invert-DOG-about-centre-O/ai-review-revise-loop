"""
Scaled-up robustness check, addressing reviewer round-2 questions 2 and 3:

Q3: "would scaling up N_KEYS (e.g. 200-600) at similar per-key compute be
    feasible within your CPU budget to shrink the per-cell n" -- we scale
    N_KEYS from 60 to 240 (4x), keeping the same 8 seeds and frequency
    tiers, so each (type x tier) cell now holds ~20-40 keys instead of 5-10.

Q2: "did you check whether the per-seed key/tier/type assignment ...
    correlates with the AUROC spread" -- for every seed we also record a
    per-seed covariate (mean number of *other* keys sharing a candidate
    value with each unseen deterministic key's true value token -- i.e.
    "value-token collision rate") and report its correlation with that
    seed's AUROC across the 8 seeds.
"""
import json, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t_start = time.time()
N_SEEDS = 8

N_KEYS = 240
N_VALS = 8
BOS, SEP, EOS, PAD = 0, 1, 2, 3
KEY_BASE = 4
VAL_BASE = KEY_BASE + N_KEYS
VOCAB_SIZE = VAL_BASE + N_VALS
SEQ_LEN = 5
FREQ_TIERS = {"unseen": 0, "rare": 3, "medium": 20, "frequent": 150}
tier_names = list(FREQ_TIERS.keys())

def key_tok(i): return KEY_BASE + i
def val_tok(j): return VAL_BASE + j

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

def kl(p, q, eps=1e-9):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def entropy(p, eps=1e-9):
    p = p + eps
    return float(-np.sum(p * np.log(p)))

def run_seed(SEED):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    keys_meta = {}
    tier_cycle = (tier_names * (N_KEYS // len(tier_names) + 1))[:N_KEYS]
    random.shuffle(tier_cycle)
    for i in range(N_KEYS):
        tier = tier_cycle[i]
        is_stochastic = (i % 2 == 0)
        cand_vals = random.sample(range(N_VALS), 3)
        if is_stochastic:
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

    train_seqs = []
    for i in range(N_KEYS):
        for _ in range(keys_meta[i]["freq"]):
            v = sample_value(i)
            train_seqs.append([BOS, key_tok(i), SEP, val_tok(v), EOS])
    random.shuffle(train_seqs)
    train_tensor = torch.tensor(train_seqs, dtype=torch.long)

    model = TinyTransformerLM(VOCAB_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    BATCH, EPOCHS = 128, 25
    n = train_tensor.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            batch = train_tensor[idx]
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()

    def analytic_value_dist(key_i):
        seq = torch.tensor([[BOS, key_tok(key_i), SEP]], dtype=torch.long)
        with torch.no_grad():
            logits = model(seq)[0, -1]
        val_logits = logits[VAL_BASE:VAL_BASE + N_VALS]
        return F.softmax(val_logits, dim=-1).numpy()

    def true_dist_vec(key_i):
        d = keys_meta[key_i]["dist"]
        v = np.zeros(N_VALS)
        for k_, p_ in d.items():
            v[k_] = p_
        return v

    rows = []
    for i in range(N_KEYS):
        model_p = analytic_value_dist(i)
        true_p = true_dist_vec(i)
        rows.append(dict(
            key=i, type=keys_meta[i]["type"], tier=keys_meta[i]["tier"],
            kl=kl(true_p, model_p),
            true_entropy=entropy(true_p), model_entropy=entropy(model_p),
            model_top_p=float(model_p.max()), model_argmax=int(model_p.argmax()),
            true_argmax=int(true_p.argmax()), model_p=model_p,
        ))

    import statistics as st
    cell_n = {}
    for typ in ["deterministic", "stochastic"]:
        for tier in tier_names:
            sub = [r for r in rows if r["type"] == typ and r["tier"] == tier]
            cell_n[f"{typ}/{tier}"] = len(sub)

    det_rows = [r for r in rows if r["type"] == "deterministic"]
    y_err = [1 if r["model_argmax"] != r["true_argmax"] else 0 for r in det_rows]
    scores_entropy = [r["model_entropy"] for r in det_rows]
    auroc = roc_auc_score(y_err, scores_entropy) if len(set(y_err)) > 1 else float("nan")

    global_prior = np.mean([r["model_p"] for r in rows], axis=0)
    global_prior_entropy = entropy(global_prior)

    unseen_det = [r for r in rows if r["type"] == "deterministic" and r["tier"] == "unseen"]
    unseen_sto = [r for r in rows if r["type"] == "stochastic" and r["tier"] == "unseen"]
    h_unseen_det = st.mean(r["model_entropy"] for r in unseen_det) if unseen_det else float("nan")
    h_unseen_sto = st.mean(r["model_entropy"] for r in unseen_sto) if unseen_sto else float("nan")

    det_freq_150 = [r for r in rows if r["type"] == "deterministic" and r["tier"] == "frequent"]
    sto_freq_150 = [r for r in rows if r["type"] == "stochastic" and r["tier"] == "frequent"]
    det_unseen_kl = st.mean(r["kl"] for r in unseen_det) if unseen_det else float("nan")
    det_freq_kl = st.mean(r["kl"] for r in det_freq_150) if det_freq_150 else float("nan")
    sto_unseen_kl = st.mean(r["kl"] for r in unseen_sto) if unseen_sto else float("nan")
    sto_freq_kl = st.mean(r["kl"] for r in sto_freq_150) if sto_freq_150 else float("nan")

    # Reviewer Q2 covariate: for each unseen deterministic key, does any OTHER
    # key (any tier/type) have the same true-argmax value token? This "value
    # token collision rate" is a per-seed property of the random key/tier
    # assignment that could plausibly drive AUROC spread (more collisions ->
    # the value-token embedding gets pulled in multiple directions by other
    # keys -> noisier softmax on the unseen key -> more argmax errors).
    all_argmax_by_key = {r["key"]: r["true_argmax"] for r in rows}
    collisions = 0
    for r in unseen_det:
        v = r["true_argmax"]
        n_sharing = sum(1 for k2, v2 in all_argmax_by_key.items() if k2 != r["key"] and v2 == v)
        collisions += n_sharing
    collision_rate = collisions / len(unseen_det) if unseen_det else float("nan")

    return dict(
        seed=SEED, auroc=auroc, n_unseen_det=len(unseen_det), cell_n=cell_n,
        h_unseen_det=h_unseen_det, h_unseen_sto=h_unseen_sto,
        global_prior_entropy=global_prior_entropy,
        det_unseen_kl=det_unseen_kl, det_freq_kl=det_freq_kl,
        sto_unseen_kl=sto_unseen_kl, sto_freq_kl=sto_freq_kl,
        det_ratio=(det_unseen_kl / det_freq_kl) if det_freq_kl > 0 else float("inf"),
        sto_ratio=(sto_unseen_kl / sto_freq_kl) if sto_freq_kl > 0 else float("inf"),
        collision_rate=collision_rate,
    )

results = []
for seed in range(N_SEEDS):
    r = run_seed(seed)
    results.append(r)
    print(f"seed={seed} auroc={r['auroc']:.4f} n_unseen_det={r['n_unseen_det']} "
          f"det_ratio={r['det_ratio']:.1f} sto_ratio={r['sto_ratio']:.1f} "
          f"collision_rate={r['collision_rate']:.2f} elapsed={time.time()-t_start:.1f}s")

import statistics as st
def ms(key):
    vals = [r[key] for r in results if not math.isnan(r[key]) and not math.isinf(r[key])]
    return (st.mean(vals), st.pstdev(vals), min(vals), max(vals))

summary = {k: ms(k) for k in ["auroc", "h_unseen_det", "h_unseen_sto",
                               "global_prior_entropy", "det_ratio", "sto_ratio",
                               "collision_rate"]}
print("\n=== Summary over", N_SEEDS, "seeds, N_KEYS=", N_KEYS, "(mean, std, min, max) ===")
for k, (m, s, lo, hi) in summary.items():
    print(f"  {k:22s} {m:.4f} +/- {s:.4f}  [{lo:.4f}, {hi:.4f}]")

print("\nCell sizes (seed 0):", results[0]["cell_n"])

# Pearson correlation between per-seed AUROC and collision_rate (Q2)
aurocs = [r["auroc"] for r in results]
colls = [r["collision_rate"] for r in results]
if len(set(aurocs)) > 1 and len(set(colls)) > 1:
    corr = float(np.corrcoef(aurocs, colls)[0, 1])
else:
    corr = float("nan")
print(f"\nPearson corr(AUROC, collision_rate) across {N_SEEDS} seeds: {corr:.3f}")

elapsed = time.time() - t_start
print(f"\nTotal elapsed: {elapsed:.1f}s")

with open("scaled_results.json", "w") as f:
    json.dump(dict(n_seeds=N_SEEDS, n_keys=N_KEYS, per_seed=results, summary=summary,
                    auroc_collision_corr=corr, elapsed_seconds=elapsed), f, indent=2)
