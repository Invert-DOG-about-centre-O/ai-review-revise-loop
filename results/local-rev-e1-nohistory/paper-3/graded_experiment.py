"""
Round-3 reviewer question 2: "do you have a rough sense of whether the
KL-based calibration signal would show the same step-function pattern [as
argmax-error AUROC], or whether it might already show graded discrimination
that argmax-accuracy-based AUROC is masking?"

We reuse the 240-key, 8-seed setup (identical generative process, model,
training) and, for DETERMINISTIC keys, record mean KL and mean entropy in
*all four* frequency tiers (unseen/rare/medium/frequent), not just
unseen-vs-frequent. If KL is graded (monotonically decreasing rare>medium>
frequent, not just unseen>>rest), that's evidence KL carries within-tier
information that binary argmax-error AUROC cannot see (accuracy is ~100%
already at rare/medium/frequent, so AUROC is blind there by construction).
"""
import json, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            raw = np.array([0.6, 0.3, 0.1]); np.random.shuffle(raw)
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

    tier_kl = {t: [] for t in tier_names}
    tier_ent = {t: [] for t in tier_names}
    for i in range(N_KEYS):
        if keys_meta[i]["type"] != "deterministic":
            continue
        model_p = analytic_value_dist(i)
        true_p = true_dist_vec(i)
        tier_kl[keys_meta[i]["tier"]].append(kl(true_p, model_p))
        tier_ent[keys_meta[i]["tier"]].append(entropy(model_p))

    return dict(
        seed=SEED,
        kl_by_tier={t: float(np.mean(v)) for t, v in tier_kl.items()},
        ent_by_tier={t: float(np.mean(v)) for t, v in tier_ent.items()},
    )

results = [run_seed(s) for s in range(N_SEEDS)]
for r in results:
    print(r["seed"], r["kl_by_tier"], r["ent_by_tier"])

order = ["unseen", "rare", "medium", "frequent"]
kl_summary = {t: (float(np.mean([r["kl_by_tier"][t] for r in results])),
                   float(np.std([r["kl_by_tier"][t] for r in results]))) for t in order}
ent_summary = {t: (float(np.mean([r["ent_by_tier"][t] for r in results])),
                    float(np.std([r["ent_by_tier"][t] for r in results]))) for t in order}

print("\nKL by tier (mean, std):")
for t in order: print(f"  {t:10s} {kl_summary[t][0]:.5f} +/- {kl_summary[t][1]:.5f}")
print("\nEntropy by tier (mean, std):")
for t in order: print(f"  {t:10s} {ent_summary[t][0]:.5f} +/- {ent_summary[t][1]:.5f}")

# is KL monotonically decreasing across rare->medium->frequent (graded within "seen")?
seen_order = ["rare", "medium", "frequent"]
kl_monotone = all(kl_summary[seen_order[i]][0] > kl_summary[seen_order[i+1]][0] for i in range(len(seen_order)-1))
ent_monotone = all(ent_summary[seen_order[i]][0] > ent_summary[seen_order[i+1]][0] for i in range(len(seen_order)-1))
print(f"\nKL monotonically decreasing rare>medium>frequent: {kl_monotone}")
print(f"Entropy monotonically decreasing rare>medium>frequent: {ent_monotone}")
print(f"KL rare/frequent ratio: {kl_summary['rare'][0]/kl_summary['frequent'][0]:.2f}")
print(f"Entropy rare-frequent gap (nats): {ent_summary['rare'][0]-ent_summary['frequent'][0]:.5f}")

elapsed = time.time() - t_start
print(f"\nTotal elapsed: {elapsed:.1f}s")

with open("graded_results.json", "w") as f:
    json.dump(dict(n_seeds=N_SEEDS, n_keys=N_KEYS, per_seed=results,
                    kl_summary=kl_summary, ent_summary=ent_summary,
                    kl_monotone=kl_monotone, ent_monotone=ent_monotone,
                    elapsed_seconds=elapsed), f, indent=2)
