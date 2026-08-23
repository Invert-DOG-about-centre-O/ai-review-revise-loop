"""
Reviewer round-3 Q1: with only 3-4 repeats per (seed,K) we can show the seed-1
K-effect is not separable from noise but not establish its true sign. Redo with
REPEATS=15 per operating point (still cheap - no retraining, same frozen
seed-1 weights) to see if the sign stabilizes or if K is simply inert here.
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def train_model(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    seen = set()
    train_data, seen = build_dataset(700, seen=seen)
    test_data, seen = build_dataset(200, seen=seen)
    model = TinyGPT(VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    BATCH, EPOCHS = 32, 60
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
    return model, test_data


@torch.no_grad()
def sample_answer(model, prompt, temperature=0.9, max_new=4):
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


REPEATS = 5
out = {}
t_start = time.time()
for seed in [1]:
    model, test_data = train_model(seed)
    is_error, greedy_mte = [], []
    for prompt, answer, full in test_data:
        true_val = int(answer)
        ans, ents = sample_answer(model, prompt, temperature=0.0, max_new=4)
        val = parse_int(ans)
        is_error.append(0 if val == true_val else 1)
        greedy_mte.append(sum(ents) / max(len(ents), 1))
    mte_auroc = auroc(greedy_mte, is_error)
    print(f"seed {seed}: mean_token_entropy AUROC={mte_auroc:.3f}")

    seed_out = {"mean_token_entropy_auroc": mte_auroc, "repeats": {}}
    for K in [8, 24]:
        se_list, scd_list = [], []
        for rep in range(REPEATS):
            rng_seed = 5000 * seed + 10 * K + rep
            random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            se_scores, scd_scores = [], []
            for prompt, answer, full in test_data:
                samples = [sample_answer(model, prompt, temperature=0.9, max_new=4)[0] for _ in range(K)]
                vals = [parse_int(s) for s in samples]
                counts = Counter(vals)
                probs = [c / K for c in counts.values()]
                se = -sum(p * math.log(p + 1e-12) for p in probs)
                maj_count = counts.most_common(1)[0][1]
                scd = 1.0 - maj_count / K
                se_scores.append(se)
                scd_scores.append(scd)
            se_auroc = auroc(se_scores, is_error)
            scd_auroc = auroc(scd_scores, is_error)
            se_list.append(se_auroc)
            scd_list.append(scd_auroc)
            print(f"  seed{seed} K={K} rep{rep}: semantic_entropy={se_auroc:.3f} self_consistency={scd_auroc:.3f} "
                  f"(elapsed {time.time()-t_start:.0f}s)")
        seed_out["repeats"][f"K{K}"] = {
            "semantic_entropy_auroc": se_list,
            "self_consistency_auroc": scd_list,
            "se_mean": sum(se_list) / len(se_list),
            "se_std": (sum((x - sum(se_list) / len(se_list)) ** 2 for x in se_list) / len(se_list)) ** 0.5,
            "scd_mean": sum(scd_list) / len(scd_list),
            "scd_std": (sum((x - sum(scd_list) / len(scd_list)) ** 2 for x in scd_list) / len(scd_list)) ** 0.5,
        }
    out[f"seed{seed}"] = seed_out

with open("noise_decomposition_extended.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("Saved noise_decomposition_extended.json")
