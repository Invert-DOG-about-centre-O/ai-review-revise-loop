"""
Quick check: does self-consistency's AUROC improve toward maxprob's as K grows?
Same seed-0 model/training as experiment.py, but eval self-consistency at
K=20 and K=50 on the same held examples (N_TEST=200 for speed), reusing the
K=20 samples as the first 20 of the K=50 draws so comparisons are paired.
"""
import time, random, json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t0 = time.time()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

CHARS = list("0123456789+=")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
IN_LEN = 6
OUT_LEN = 3
SEQ_LEN = IN_LEN + OUT_LEN


def make_example():
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    s = a + b
    return f"{a:02d}+{b:02d}={s:03d}", a, b, s


def encode(text):
    return [STOI[c] for c in text]


def batch(n):
    return torch.tensor([encode(make_example()[0]) for _ in range(n)], dtype=torch.long)


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

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.ln(h)
        return self.head(h)


model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
TRAIN_STEPS = 260
BATCH = 128
for step in range(TRAIN_STEPS):
    x = batch(BATCH)
    inp, tgt = x[:, :-1], x[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(logits[:, IN_LEN - 1:].reshape(-1, VOCAB), tgt[:, IN_LEN - 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"[{time.time()-t0:.1f}s] training done, loss {loss.item():.4f}")

model.eval()
N_TEST = 200
K_MAX = 50
TEMP = 1.0
test_examples = [make_example() for _ in range(N_TEST)]


@torch.no_grad()
def greedy_decode(prefix_ids):
    seq = list(prefix_ids)
    maxprobs = []
    for _ in range(OUT_LEN):
        x = torch.tensor([seq], dtype=torch.long)
        logits = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        maxprobs.append(probs.max().item())
        seq.append(int(probs.argmax().item()))
    return "".join(ITOS[i] for i in seq[IN_LEN:]), float(np.mean(maxprobs))


@torch.no_grad()
def sample_decode(prefix_ids, temp=TEMP):
    seq = list(prefix_ids)
    for _ in range(OUT_LEN):
        x = torch.tensor([seq], dtype=torch.long)
        logits = model(x)
        probs = F.softmax(logits[0, -1] / temp, dim=-1)
        seq.append(int(torch.multinomial(probs, 1).item()))
    return "".join(ITOS[i] for i in seq[IN_LEN:])


y, maxprob = [], []
agree_k = {20: [], 50: []}
print(f"[{time.time()-t0:.1f}s] starting eval, N={N_TEST}, K_MAX={K_MAX}")
for text, a, b, s in test_examples:
    true_ans = f"{s:03d}"
    prefix = encode(text)[:IN_LEN]
    greedy_ans, mp = greedy_decode(prefix)
    y.append(int(greedy_ans == true_ans))
    maxprob.append(mp)
    samples = [sample_decode(prefix) for _ in range(K_MAX)]
    for k in (20, 50):
        agree_k[k].append(sum(1 for smp in samples[:k] if smp == greedy_ans) / k)
print(f"[{time.time()-t0:.1f}s] eval done")

y = np.array(y)
maxprob = np.array(maxprob)
out = {"n_test": N_TEST, "auroc_maxprob": roc_auc_score(y, maxprob)}
for k in (20, 50):
    out[f"auroc_agree_K{k}"] = roc_auc_score(y, np.array(agree_k[k]))
print(json.dumps(out, indent=2))
with open("k_sweep_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"[{time.time()-t0:.1f}s] DONE")
