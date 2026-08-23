"""Same as ablation_k_temp.py but seed 1, fewer configs (K=8 vs K=24 @ T=0.9) to fit time budget."""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)

CHARS = list("0123456789+= ") + ["<PAD>", "<EOS>"]
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
PAD, EOS = STOI["<PAD>"], STOI["<EOS>"]
MAXLEN = 16


def make_example(a, b):
    prompt = f"{a}+{b}="
    answer = str(a + b)
    return prompt, answer, prompt + answer


def encode(s):
    return [STOI[c] for c in s]


def build_dataset(n, lo=0, hi=40, seen=None):
    data = []
    seen = seen if seen is not None else set()
    while len(data) < n:
        a, b = random.randint(lo, hi), random.randint(lo, hi)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        data.append(make_example(a, b))
    return data, seen


seen = set()
train_data, seen = build_dataset(700, seen=seen)
test_data, seen = build_dataset(200, seen=seen)


def collate(batch):
    xs, masks = [], []
    for prompt, answer, full in batch:
        seq = encode(full) + [EOS]
        seq = seq[:MAXLEN]
        pad_len = MAXLEN - len(seq)
        mask = [0] * len(encode(prompt)) + [1] * (len(seq) - len(encode(prompt))) + [0] * pad_len
        seq = seq + [PAD] * pad_len
        xs.append(seq)
        masks.append(mask[:MAXLEN])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(masks, dtype=torch.float)


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=64, n_head=4, n_layer=3, maxlen=MAXLEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.maxlen = maxlen
        mask = torch.triu(torch.ones(maxlen, maxlen) * float("-inf"), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.blocks(h, mask=self.causal_mask[:t, :t])
        h = self.ln_f(h)
        return self.head(h)


model = TinyGPT(VOCAB)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
BATCH, EPOCHS = 32, 60

t0 = time.time()
for epoch in range(EPOCHS):
    random.shuffle(train_data)
    for i in range(0, len(train_data), BATCH):
        batch = train_data[i:i + BATCH]
        x, mask = collate(batch)
        inp, tgt = x[:, :-1], x[:, 1:]
        m = mask[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
print(f"training took {time.time()-t0:.1f}s")


@torch.no_grad()
def sample_answer(prompt, temperature=0.8, max_new=4):
    ids = encode(prompt)
    entropies = []
    for _ in range(max_new):
        x = torch.tensor([ids[-MAXLEN:]], dtype=torch.long)
        logits = model(x)[0, -1]
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        entropies.append(ent)
        if temperature == 0:
            nxt = int(torch.argmax(probs).item())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item())
        ids.append(nxt)
        if nxt == EOS:
            break
    gen = ids[len(encode(prompt)):]
    out = []
    for tkn in gen:
        if tkn == EOS:
            break
        out.append(ITOS[tkn])
    return "".join(out), entropies


def parse_int(s):
    s = s.strip()
    if s == "" or not s.lstrip("-").isdigit():
        return None
    try:
        return int(s)
    except ValueError:
        return None


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    c = 0
    for p in pos:
        for n in neg:
            c += 1 if p > n else (0.5 if p == n else 0)
    return c / (len(pos) * len(neg))


is_error, greedy_mte = [], []
for prompt, answer, full in test_data:
    true_val = int(answer)
    ans, ents = sample_answer(prompt, temperature=0.0, max_new=4)
    val = parse_int(ans)
    is_error.append(0 if val == true_val else 1)
    greedy_mte.append(sum(ents) / max(len(ents), 1))

mte_auroc = auroc(greedy_mte, is_error)
print(f"mean_token_entropy AUROC (K-independent baseline): {mte_auroc:.3f}")

configs = [(8, 0.9), (24, 0.9)]
results = {}
for K, T in configs:
    t0 = time.time()
    se_scores, scd_scores = [], []
    for prompt, answer, full in test_data:
        samples = [sample_answer(prompt, temperature=T, max_new=4)[0] for _ in range(K)]
        vals = [parse_int(s) for s in samples]
        counts = Counter(vals)
        probs = [c / K for c in counts.values()]
        se = -sum(p * math.log(p + 1e-12) for p in probs)
        maj_count = counts.most_common(1)[0][1]
        scd = 1.0 - maj_count / K
        se_scores.append(se)
        scd_scores.append(scd)
    dt = time.time() - t0
    se_auroc = auroc(se_scores, is_error)
    scd_auroc = auroc(scd_scores, is_error)
    print(f"K={K:2d} T={T}: semantic_entropy AUROC={se_auroc:.3f}  "
          f"self_consistency AUROC={scd_auroc:.3f}  ({dt:.1f}s, n=200)")
    results[f"K{K}_T{T}"] = {"semantic_entropy_auroc": se_auroc,
                              "self_consistency_auroc": scd_auroc, "time_s": dt}

out = {"seed": SEED, "mean_token_entropy_auroc": mte_auroc, "configs": results}
with open("ablation_k_temp_seed1.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved ablation_k_temp_seed1.json")
