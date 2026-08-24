"""
Follow-up experiment for round-2 review question 1: does the paraphrase-noise
SE-over-self-consistency gap (paraphrase_experiment.py) replicate across
independently trained models, the way the noise-free near-tie was shown to
replicate in seed_experiment.py? Trains 3 independent seeds (identical
architecture/task) and re-runs the full noise_p sweep for each.
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

def render_variants(n):
    return [str(n), f"+{n}", f"{n}.0", (f"0{n}" if n >= 0 else str(n))]

def render_surface(n, rng, noise_p):
    if rng.random() < noise_p:
        return rng.choice(render_variants(n))
    return str(n)

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

def paired_bootstrap_diff_p(scoresA, scoresB, labels, n_boot=500, seed=7):
    rng = random.Random(seed)
    n = len(labels)
    idx = list(range(n))
    diffs = []
    for _ in range(n_boot):
        samp = [rng.choice(idx) for _ in range(n)]
        l = [labels[i] for i in samp]
        a = auroc([scoresA[i] for i in samp], l)
        b = auroc([scoresB[i] for i in samp], l)
        if not (math.isnan(a) or math.isnan(b)):
            diffs.append(a - b)
    diffs.sort()
    aA, aB = auroc(scoresA, labels), auroc(scoresB, labels)
    obs = aA - aB
    if obs >= 0:
        p = 2 * (sum(1 for d in diffs if d <= 0) / len(diffs))
    else:
        p = 2 * (sum(1 for d in diffs if d >= 0) / len(diffs))
    return min(p, 1.0), obs

def train_model(seed):
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
    return model

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

def eval_question(model, q, true_c, n_samples, temperature, seed, noise_p):
    prompt_ids = encode(q)
    gens = generate(model, prompt_ids, n_samples, temperature, rng_seed=seed)
    answers = [parse_answer(g) for g in gens]
    valid = [a for a in answers if a is not None]
    rng = random.Random(seed * 7919 + 3)

    if not valid:
        maj_ans, se = None, math.log(max(len(answers), 1))
    else:
        cnt = Counter(valid)
        total = len(valid)
        maj_ans, maj_count = cnt.most_common(1)[0]
        se = -sum((c / total) * math.log(c / total) for c in cnt.values())

    surf = [render_surface(a, rng, noise_p) if a is not None else None for a in answers]
    valid_s = [s for s in surf if s is not None]
    if not valid_s:
        sc_frac = 0.0
    else:
        cnt_s = Counter(valid_s)
        sc_frac = cnt_s.most_common(1)[0][1] / len(valid_s)

    correct = (maj_ans == true_c)
    return {"se": se, "sc_surface": 1.0 - sc_frac, "correct": correct}

def run_noise_sweep(model, seed_tag):
    out = {}
    for noise_p in [0.0, 0.3, 0.6, 0.9]:
        rng = random.Random(5)
        rows = []
        for i in range(300):
            op = rng.choice(OPS)
            nd = rng.choice(DIGS)
            q, ans, c = make_problem(op, nd, rng)
            r = eval_question(model, q, c, 10, 0.8, seed=1000 + i, noise_p=noise_p)
            rows.append(r)
        labels = [0 if r["correct"] else 1 for r in rows]
        se = [r["se"] for r in rows]
        sc = [r["sc_surface"] for r in rows]
        a_se, a_sc = auroc(se, labels), auroc(sc, labels)
        p, diff = paired_bootstrap_diff_p(se, sc, labels)
        out[str(noise_p)] = {
            "err_rate": sum(labels) / len(labels),
            "auroc_SE": a_se, "auroc_SC_surface": a_sc,
            "diff_SE_minus_SC": diff, "p_raw": p,
        }
        print(f"seed={seed_tag} noise_p={noise_p} SE={a_se:.4f} SC={a_sc:.4f} diff={diff:+.4f} p={p:.3f}")
    return out

t0 = time.time()
all_results = {}
for seed in [0, 1, 2]:
    model = train_model(seed)
    print(f"seed {seed} trained, elapsed={time.time()-t0:.1f}s")
    all_results[str(seed)] = run_noise_sweep(model, seed)

all_results["total_elapsed_sec"] = time.time() - t0
with open("paraphrase_seed_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("DONE total_elapsed=", time.time() - t0)
