"""
Follow-up experiment for round-2 review question 3: the digit-length ablation
in experiment.py uses a single random draw of n=120 problems per digit
length, so 1-digit (few errors) and 2-digit (few corrects) AUROCs have no CI
and 3-digit is undefined (100% error). Here we draw MANY independent problem
sets per digit length (same trained model, same main-condition generation
settings) and report a bootstrap-style spread across draws, which is the
resampling the single-draw ablation could not provide.
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

torch.manual_seed(0)
random.seed(0)
model = TinyLM(VOCAB)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
train_rng = random.Random(1)
t0 = time.time()
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
print(f"trained, elapsed={time.time()-t0:.1f}s")

@torch.no_grad()
def generate(prompt_ids, n_samples, temperature, max_new=8, rng_seed=0):
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

def eval_one(nd, draw_seed, n_q=60):
    rng = random.Random(draw_seed)
    rows = []
    for i in range(n_q):
        op = rng.choice(OPS)
        q, ans, c = make_problem(op, nd, rng)
        gens = generate(encode(q), 10, 0.8, rng_seed=draw_seed * 100003 + i)
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
    return auroc(se, labels), sum(labels) / len(labels), sum(1 for l in labels if l == 0)

N_DRAWS = 8
results = {}
for nd in [1, 2, 3]:
    aurocs, err_rates, n_correct = [], [], []
    for d in range(N_DRAWS):
        a, err, ncorr = eval_one(nd, draw_seed=2000 + nd * 100 + d)
        aurocs.append(a)
        err_rates.append(err)
        n_correct.append(ncorr)
    valid_a = [a for a in aurocs if not math.isnan(a)]
    results[str(nd)] = {
        "n_draws": N_DRAWS, "n_per_draw": 60,
        "mean_err_rate": sum(err_rates) / len(err_rates),
        "n_draws_with_defined_auroc": len(valid_a),
        "auroc_mean": (sum(valid_a) / len(valid_a)) if valid_a else None,
        "auroc_min": min(valid_a) if valid_a else None,
        "auroc_max": max(valid_a) if valid_a else None,
        "auroc_per_draw": aurocs,
        "mean_n_correct_per_draw": sum(n_correct) / len(n_correct),
    }
    print(f"digits={nd}: err_rate={results[str(nd)]['mean_err_rate']:.3f} "
          f"defined_draws={len(valid_a)}/{N_DRAWS} "
          f"auroc_mean={results[str(nd)]['auroc_mean']}")

results["total_elapsed_sec"] = time.time() - t0
with open("digit_length_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE total_elapsed=", time.time() - t0)
