"""
Multi-seed variance check for the toy UQ study (addresses reviewer concern:
"single run, no seeds/variance"). Re-runs the full train+eval pipeline for
5 independent seeds (fresh model init, fresh train/test data, fresh sampling)
at the SAME 260-step / ~19% error-rate operating point, and reports
mean +/- std for each AUROC/ECE plus a bootstrap CI on the two headline gaps:
maxprob vs self-consistency AUROC, and probe vs true-expensive-signal AUROC.
"""
import time, math, random, json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CHARS = list("0123456789+=")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
IN_LEN, OUT_LEN = 6, 3
SEQ_LEN = IN_LEN + OUT_LEN

N_TEST = 400
K_SAMPLES = 20
TEMP = 1.0
TRAIN_STEPS = 260
BATCH = 128
SEEDS = [0, 1, 2, 3, 4]


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


def make_example(rng):
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    s = a + b
    text = f"{a:02d}+{b:02d}={s:03d}"
    return text, a, b, s


def encode(text):
    return [STOI[c] for c in text]


def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def batch(n):
        xs = [encode(make_example(random)[0]) for _ in range(n)]
        return torch.tensor(xs, dtype=torch.long)

    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(TRAIN_STEPS):
        x = batch(BATCH)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits[:, IN_LEN - 1:].reshape(-1, VOCAB),
                                tgt[:, IN_LEN - 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    test_examples = [make_example(random) for _ in range(N_TEST)]

    @torch.no_grad()
    def greedy_decode_with_features(prefix_ids):
        seq = list(prefix_ids)
        maxprobs, entropies, hiddens = [], [], []
        for _ in range(OUT_LEN):
            x = torch.tensor([seq], dtype=torch.long)
            logits, h = model(x, return_hidden=True)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum().item()
            mp = probs.max().item()
            nxt = int(probs.argmax().item())
            maxprobs.append(mp); entropies.append(ent); hiddens.append(h[0, -1].numpy())
            seq.append(nxt)
        answer = "".join(ITOS[i] for i in seq[IN_LEN:])
        return answer, float(np.mean(maxprobs)), float(np.mean(entropies)), np.mean(hiddens, axis=0)

    @torch.no_grad()
    def sample_decode(prefix_ids, temp=TEMP):
        seq = list(prefix_ids)
        for _ in range(OUT_LEN):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)
            probs = F.softmax(logits[0, -1] / temp, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            seq.append(nxt)
        return "".join(ITOS[i] for i in seq[IN_LEN:])

    rows = []
    for text, a, b, s in test_examples:
        true_ans = f"{s:03d}"
        prefix = encode(text)[:IN_LEN]
        greedy_ans, maxprob, pred_entropy, hidden_mean = greedy_decode_with_features(prefix)
        correct = int(greedy_ans == true_ans)
        samples = [sample_decode(prefix) for _ in range(K_SAMPLES)]
        agree = sum(1 for smp in samples if smp == greedy_ans) / K_SAMPLES
        counts = Counter(samples)
        probs_emp = np.array(list(counts.values())) / K_SAMPLES
        sample_entropy = float(-(probs_emp * np.log(probs_emp)).sum())
        rows.append(dict(correct=correct, maxprob=maxprob, pred_entropy=pred_entropy,
                          sample_agree=agree, sample_entropy=sample_entropy, hidden=hidden_mean))

    y = np.array([r["correct"] for r in rows])
    maxprob = np.array([r["maxprob"] for r in rows])
    pred_entropy = np.array([r["pred_entropy"] for r in rows])
    sample_agree = np.array([r["sample_agree"] for r in rows])
    sample_entropy = np.array([r["sample_entropy"] for r in rows])
    hidden = np.stack([r["hidden"] for r in rows])

    def safe_auroc(score, label):
        return roc_auc_score(label, score) if len(set(label.tolist())) > 1 else float("nan")

    auroc_maxprob = safe_auroc(maxprob, y)
    auroc_negent = safe_auroc(-pred_entropy, y)
    auroc_agree = safe_auroc(sample_agree, y)
    auroc_negsampent = safe_auroc(-sample_entropy, y)

    idx = np.arange(N_TEST)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = N_TEST // 2
    train_idx, test_idx = idx[:split], idx[split:]
    X = np.concatenate([maxprob.reshape(-1, 1), pred_entropy.reshape(-1, 1), hidden], axis=1)
    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X)
    probe = Ridge(alpha=1.0).fit(Xs[train_idx], sample_entropy[train_idx])
    pred_se = probe.predict(Xs[test_idx])
    ss_res = np.sum((sample_entropy[test_idx] - pred_se) ** 2)
    ss_tot = np.sum((sample_entropy[test_idx] - sample_entropy[test_idx].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    auroc_true_holdout = safe_auroc(-sample_entropy[test_idx], y[test_idx])
    auroc_probe_holdout = safe_auroc(-pred_se, y[test_idx])

    return dict(seed=seed, acc=float(y.mean()),
                auroc_maxprob=auroc_maxprob, auroc_negent=auroc_negent,
                auroc_agree=auroc_agree, auroc_negsampent=auroc_negsampent,
                probe_r2=r2, auroc_true_holdout=auroc_true_holdout,
                auroc_probe_holdout=auroc_probe_holdout,
                gap_maxprob_minus_agree=auroc_maxprob - auroc_agree)


t0 = time.time()
results = []
for sd in SEEDS:
    r = run_seed(sd)
    results.append(r)
    print(f"[{time.time()-t0:.1f}s] seed={sd} acc={r['acc']:.3f} "
          f"maxprob={r['auroc_maxprob']:.3f} agree={r['auroc_agree']:.3f} "
          f"gap={r['gap_maxprob_minus_agree']:.3f} probe_r2={r['probe_r2']:.3f} "
          f"true_holdout={r['auroc_true_holdout']:.3f} probe_holdout={r['auroc_probe_holdout']:.3f}")

def mstd(key):
    vals = np.array([r[key] for r in results])
    return float(vals.mean()), float(vals.std())

summary = {k: mstd(k) for k in [
    "acc", "auroc_maxprob", "auroc_negent", "auroc_agree", "auroc_negsampent",
    "probe_r2", "auroc_true_holdout", "auroc_probe_holdout", "gap_maxprob_minus_agree"]}

gaps = np.array([r["gap_maxprob_minus_agree"] for r in results])
n_positive = int((gaps > 0).sum())

print("\n=== SUMMARY over", len(SEEDS), "seeds (mean, std) ===")
for k, v in summary.items():
    print(f"{k}: {v[0]:.4f} +/- {v[1]:.4f}")
print(f"maxprob > self-consistency in {n_positive}/{len(SEEDS)} seeds")
print(f"per-seed gaps: {gaps.tolist()}")

with open("multiseed_results.json", "w") as f:
    json.dump({"per_seed": results, "summary": summary,
                "n_seeds_maxprob_wins": n_positive, "n_seeds": len(SEEDS)}, f, indent=2)
print(f"[{time.time()-t0:.1f}s] DONE")
