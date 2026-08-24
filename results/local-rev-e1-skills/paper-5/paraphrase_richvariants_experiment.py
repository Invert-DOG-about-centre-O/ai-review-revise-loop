"""
Follow-up for round-3 review question 3: the original paraphrase renderer
uses only 4 fixed surface variants ("42","+42","42.0","042"). Here we widen
the variant set to test whether the SE-over-self-consistency gap grows,
shrinks, or stays flat under richer surface variation. Identical model,
training, and evaluation pipeline as paraphrase_experiment.py; only
render_variants() changes, adding: scientific notation ("4.2e1"), a leading
"=" echo ("=42"), thousands-style grouping for the ones/tens split
("4,2" is not meaningful here so we use spaced digits "4 2"), and a trailing
period-free decimal ("42.00"). This roughly doubles the variant count and
adds a qualitatively different family (scientific notation) versus the
original set, which was closer to simple affix/format perturbations.
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
torch.manual_seed(0)
random.seed(0)

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

device = "cpu"
model = TinyLM(VOCAB).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

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

TRAIN_STEPS = 900
BATCH = 128
train_rng = random.Random(1)
print("Training TinyLM (identical to paraphrase_experiment.py)...")
model.train()
for step in range(TRAIN_STEPS):
    strings = gen_batch_strings(BATCH, train_rng)
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
print(f"Training done. elapsed={time.time()-t_start:.1f}s")
model.eval()

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

def parse_answer(tok_ids):
    s = "".join(itos[t] for t in tok_ids if t in itos)
    s = s.split(";")[0]
    try:
        return int(s)
    except ValueError:
        return None

# Richer variant set: original 4 (affix/format perturbations) + 4 more
# spanning a qualitatively different family (scientific notation, echoed
# "=", spaced digits, zero-padded decimal) -- 8 variants total, roughly 2x
# the original set and structurally more diverse.
def render_variants_rich(n):
    base = [str(n), f"+{n}", f"{n}.0", (f"0{n}" if n >= 0 else str(n))]
    extra = [f"{n}e0", f"={n}", " ".join(str(n)), f"{n}.00"]
    return base + extra

def render_surface(n, rng, noise_p):
    if rng.random() < noise_p:
        return rng.choice(render_variants_rich(n))
    return str(n)

def eval_question(q, true_c, n_samples, temperature, seed, noise_p):
    prompt_ids = encode(q)
    gens = generate(prompt_ids, n_samples, temperature, rng_seed=seed)
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
    return {"semantic_entropy": se, "self_inconsistency": 1.0 - sc_frac, "correct": correct}

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

def paired_bootstrap_diff_p(scoresA, scoresB, labels, n_boot=1000, seed=7):
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

def run(noise_p, n_q=300, n_samples=10, temperature=0.8, seed=5):
    rng = random.Random(seed)
    rows = []
    for i in range(n_q):
        op = rng.choice(OPS)
        nd = rng.choice(DIGS)
        q, ans, c = make_problem(op, nd, rng)
        r = eval_question(q, c, n_samples, temperature, seed=1000 + i, noise_p=noise_p)
        rows.append(r)
    labels = [0 if r["correct"] else 1 for r in rows]
    se = [r["semantic_entropy"] for r in rows]
    sc = [r["self_inconsistency"] for r in rows]
    a_se = auroc(se, labels)
    a_sc = auroc(sc, labels)
    p, diff = paired_bootstrap_diff_p(se, sc, labels)
    return {
        "noise_p": noise_p, "n": n_q, "err_rate": sum(labels) / len(labels),
        "auroc_SE": a_se, "auroc_SC_surface": a_sc,
        "diff_SE_minus_SC": diff, "p_raw": p,
    }

results = {}
for noise_p in [0.0, 0.3, 0.6, 0.9]:
    out = run(noise_p)
    results[str(noise_p)] = out
    print(json.dumps(out, indent=2))
    print(f"elapsed so far: {time.time()-t_start:.1f}s")

results["total_elapsed_sec"] = time.time() - t_start
with open("paraphrase_richvariants_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE. total_elapsed=", results["total_elapsed_sec"])
