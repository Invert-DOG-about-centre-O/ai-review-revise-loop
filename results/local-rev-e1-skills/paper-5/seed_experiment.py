"""
Follow-up experiment for round-1 review (weakness: single training seed).
Retrains the identical TinyLM architecture/task from 3 different random
seeds (init + training data stream) and re-runs the MAIN condition
(N=10, T=0.8, n=300, all ops/digits) for each, to check whether the
SE ~ self-consistency ~ NLL >> token-entropy pattern is a property of one
lucky training run or holds across independent trainings.
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = list("0123456789+-*=;. ")
stoi = {c: i for i, c in enumerate(CHARS)}
itos = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
PAD = stoi[" "]
EOS = stoi[";"]

def encode(s):
    return [stoi[c] for c in s]

def make_problem(op, ndig, rng):
    a = rng.randint(1, 10**ndig - 1)
    b = rng.randint(1, 10**ndig - 1)
    if op == "+":
        c = a + b
    elif op == "-":
        if b > a:
            a, b = b, a
        c = a - b
    else:
        c = a * b
    q = f"{a}{op}{b}="
    ans = f"{c};"
    return q, ans, c

OPS = ["+", "-", "*"]
DIGS = [1, 2, 3]
MAXLEN = 20

def pad_seq(ids):
    ids = ids[:MAXLEN]
    return ids + [PAD] * (MAXLEN - len(ids))

class TinyLM(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, nlayers=3, maxlen=MAXLEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, vocab)
        self.maxlen = maxlen

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.head(h)

def gen_batch_strings(n, rng):
    out = []
    for _ in range(n):
        op = rng.choice(OPS)
        nd = rng.choice(DIGS)
        q, ans, c = make_problem(op, nd, rng)
        out.append((q, ans, c, op, nd))
    return out

def batch_tensor(strings):
    xs, masks = [], []
    for q, ans, c, op, nd in strings:
        full = q + ans
        ids = encode(full)
        qlen = len(q)
        ids = pad_seq(ids)
        m = [0] * MAXLEN
        for i in range(qlen, min(len(full), MAXLEN)):
            m[i] = 1
        xs.append(ids)
        masks.append(m)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(masks, dtype=torch.float)

def parse_answer(tok_ids):
    s = "".join(itos[t] for t in tok_ids if t in itos)
    s = s.split(";")[0]
    try:
        return int(s)
    except ValueError:
        return None

def auroc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    srt = sorted(range(len(scores)), key=lambda k: scores[k])
    ranks = {}
    i = 0
    r = 1
    while i < len(srt):
        j = i
        while j < len(srt) and scores[srt[j]] == scores[srt[i]]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2
        for k in range(i, j):
            ranks[srt[k]] = avg_rank
        r += (j - i)
        i = j
    sum_ranks_pos = sum(ranks[k] for k in range(len(scores)) if labels[k] == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def run_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = TinyLM(VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train_rng = random.Random(1000 + seed)
    model.train()
    for step in range(900):
        strings = gen_batch_strings(128, train_rng)
        x, mask = batch_tensor(strings)
        inp, tgt = x[:, :-1], x[:, 1:]
        mtgt = mask[:, 1:]
        logits = model(inp)
        loss_tok = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss_tok = loss_tok.view(tgt.shape)
        loss = (loss_tok * mtgt).sum() / mtgt.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    @torch.no_grad()
    def generate(prompt_ids, n_samples, temperature, max_new=8, rng_seed=0):
        g = torch.Generator().manual_seed(rng_seed)
        x = torch.tensor([prompt_ids] * n_samples, dtype=torch.long)
        finished = [False] * n_samples
        gen_tokens = [[] for _ in range(n_samples)]
        first_tok_entropy = None
        for step in range(max_new):
            logits = model(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if step == 0:
                p0 = probs[0]
                first_tok_entropy = float(-(p0 * (p0 + 1e-12).log()).sum())
            if temperature != 1.0:
                logits_t = logits / temperature
                probs = F.softmax(logits_t, dim=-1)
            next_tok = torch.multinomial(probs, 1, generator=g).squeeze(-1)
            for i in range(n_samples):
                if not finished[i]:
                    t = int(next_tok[i])
                    gen_tokens[i].append(t)
                    if t == EOS:
                        finished[i] = True
            x = torch.cat([x, next_tok.unsqueeze(-1)], dim=1)
            if all(finished):
                break
        return gen_tokens, first_tok_entropy

    @torch.no_grad()
    def greedy_nll(prompt_ids, max_new=8):
        x = torch.tensor([prompt_ids], dtype=torch.long)
        total_nll, count = 0.0, 0
        for step in range(max_new):
            logits = model(x)[:, -1, :]
            logp = F.log_softmax(logits, dim=-1)[0]
            next_tok = int(torch.argmax(logp))
            total_nll += -float(logp[next_tok])
            count += 1
            x = torch.cat([x, torch.tensor([[next_tok]])], dim=1)
            if next_tok == EOS:
                break
        return total_nll / max(count, 1)

    rng = random.Random(5)
    rows = []
    for i in range(300):
        op = rng.choice(OPS)
        nd = rng.choice(DIGS)
        q, ans, c = make_problem(op, nd, rng)
        prompt_ids = encode(q)
        gens, tok_ent = generate(prompt_ids, 10, 0.8, rng_seed=1000 + i)
        answers = [parse_answer(g) for g in gens]
        valid = [a for a in answers if a is not None]
        if not valid:
            maj_ans, maj_frac, se = None, 0.0, math.log(max(len(answers), 1))
        else:
            cnt = Counter(valid)
            total = len(valid)
            maj_ans, maj_count = cnt.most_common(1)[0]
            maj_frac = maj_count / total
            se = -sum((c / total) * math.log(c / total) for c in cnt.values())
        nll = greedy_nll(prompt_ids)
        rows.append({
            "se": se, "sc": 1.0 - maj_frac, "te": tok_ent, "nll": nll,
            "correct": maj_ans == c,
        })
    labels = [0 if r["correct"] else 1 for r in rows]
    out = {"seed": seed, "err_rate": sum(labels) / len(labels)}
    for k in ["se", "sc", "te", "nll"]:
        out[f"auroc_{k}"] = auroc([r[k] for r in rows], labels)
    return out

t0 = time.time()
results = []
for seed in [0, 1, 2]:
    r = run_seed(seed)
    r["elapsed"] = time.time() - t0
    print(json.dumps(r, indent=2))
    results.append(r)

with open("seed_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE total_elapsed=", time.time() - t0)
