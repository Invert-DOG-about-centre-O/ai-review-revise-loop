"""
Follow-up for round-3 review question 2: is the digit-length ablation's label
imbalance (near-0% error at 1 digit, near-100% at 3 digits) intrinsic to this
architecture/task, or could a differently-tuned training budget rescue it by
giving more balanced per-digit-length error rates?

We retrain the identical architecture at three training budgets (300, 900
[= main paper's budget], 2700 steps) and, for each, report per-digit-length
error rate and whether AUROC is defined on a single n=200 draw per digit
length. If a shorter or longer training budget produces balanced error rates
across all three digit lengths, the imbalance is a tunable artifact of this
particular checkpoint; if every budget we try still saturates at 0% or 100%
at the extremes, that supports the paper's claim that the imbalance is
intrinsic to how this toy task's difficulty scales with digit length.
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
        nd = rng.choice([1, 2, 3])
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

@torch.no_grad()
def generate(model, prompt_ids, n_samples, temperature, max_new=8, rng_seed=0):
    g = torch.Generator().manual_seed(rng_seed)
    x = torch.tensor([prompt_ids] * n_samples, dtype=torch.long)
    finished = [False] * n_samples
    gen_tokens = [[] for _ in range(n_samples)]
    for step in range(max_new):
        logits = model(x)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
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
    return gen_tokens

def train_model(train_steps, seed=0):
    torch.manual_seed(seed)
    model = TinyLM(VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train_rng = random.Random(1)
    model.train()
    for step in range(train_steps):
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
    return model

def eval_digit(model, nd, draw_seed, n_q=200):
    rng = random.Random(draw_seed)
    rows = []
    for i in range(n_q):
        op = rng.choice(OPS)
        q, ans, c = make_problem(op, nd, rng)
        gens = generate(model, encode(q), 10, 0.8, rng_seed=draw_seed * 100003 + i)
        answers = [parse_answer(g) for g in gens]
        valid = [a for a in answers if a is not None]
        if not valid:
            maj_ans, se = None, math.log(max(len(answers), 1))
        else:
            cnt = Counter(valid)
            total = len(valid)
            maj_ans, maj_count = cnt.most_common(1)[0]
            se = -sum((c / total) * math.log(c / total) for c in cnt.values())
        rows.append({"se": se, "correct": maj_ans == c})
    labels = [0 if r["correct"] else 1 for r in rows]
    se = [r["se"] for r in rows]
    err_rate = sum(labels) / len(labels)
    return auroc(se, labels), err_rate

t0 = time.time()
BUDGETS = [300, 900, 2700]
results = {}
for budget in BUDGETS:
    model = train_model(budget)
    per_digit = {}
    for nd in [1, 2, 3]:
        a, err = eval_digit(model, nd, draw_seed=3000 + nd)
        per_digit[str(nd)] = {"err_rate": err, "auroc": (None if math.isnan(a) else a)}
        print(f"budget={budget} digits={nd} err={err:.3f} auroc={a}")
    results[str(budget)] = per_digit
    print(f"elapsed={time.time()-t0:.1f}s")

results["total_elapsed_sec"] = time.time() - t0
with open("digit_length_rescue_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE total_elapsed=", results["total_elapsed_sec"])
