"""
Cheap proxies for semantic uncertainty in a verifiable arithmetic task.

We train a small char-level Transformer LM from scratch (no internet, CPU-only)
on synthetic arithmetic problems (addition / subtraction / multiplication,
1-3 digit operands), deliberately under-capacity/under-trained so it makes a
non-trivial number of errors. We then compare several uncertainty/confidence
signals as predictors of whether the model's answer is WRONG:

  1. Semantic entropy (SE): sample N completions at temperature T, parse the
     numeric answer from each, cluster by exact numeric equality (= semantic
     equivalence in this verifiable domain), compute entropy of the cluster
     distribution. This mirrors Farquhar et al. 2024's semantic entropy but
     uses an exact symbolic equivalence check instead of an NLI clusterer.
  2. Self-consistency confidence (SC): fraction of the N samples agreeing
     with the modal (majority) answer (Wang et al. 2022 style), used as an
     uncertainty score via 1 - modal_fraction.
  3. Token-level predictive entropy (TE): entropy of the model's next-token
     distribution at the FIRST generated position, computed from a single
     forward pass (no sampling). Cheapest possible proxy.
  4. Length-normalized negative log-likelihood (NLL) of the greedy decode
     (single pass, no sampling).

For each measure we compute AUROC for predicting "the greedy/majority answer
is wrong", with bootstrap CIs, and compare measures with paired bootstrap
tests (Bonferroni-corrected for the multiple pairwise comparisons). We ablate
across: operator (+, -, *; three structurally different task types), digit
length (1-3 digits; difficulty axis), number of samples N (5/10/20), and
temperature (0.5/1.0).
"""
import math
import random
import time
import json
import itertools
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# Vocab / data
# ---------------------------------------------------------------------------
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

def gen_batch_strings(n, rng):
    out = []
    for _ in range(n):
        op = rng.choice(OPS)
        nd = rng.choice(DIGS)
        q, ans, c = make_problem(op, nd, rng)
        out.append((q, ans, c, op, nd))
    return out

MAXLEN = 20  # "999*999=998001;" fits comfortably

def pad_seq(ids):
    ids = ids[:MAXLEN]
    return ids + [PAD] * (MAXLEN - len(ids))

# ---------------------------------------------------------------------------
# Tiny transformer decoder LM
# ---------------------------------------------------------------------------
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

def batch_tensor(strings, rng):
    xs, masks = [], []
    for q, ans, c, op, nd in strings:
        full = q + ans
        ids = encode(full)
        qlen = len(q)
        ids = pad_seq(ids)
        m = [0] * MAXLEN
        for i in range(qlen, min(len(full), MAXLEN)):
            m[i] = 1  # only supervise loss on answer chars
        xs.append(ids)
        masks.append(m)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(masks, dtype=torch.float)

# ---------------------------------------------------------------------------
# Train (kept deliberately small/short -> imperfect accuracy)
# ---------------------------------------------------------------------------
TRAIN_STEPS = 900
BATCH = 128
train_rng = random.Random(1)
print("Training TinyLM...")
model.train()
for step in range(TRAIN_STEPS):
    strings = gen_batch_strings(BATCH, train_rng)
    x, mask = batch_tensor(strings, train_rng)
    inp, tgt = x[:, :-1], x[:, 1:]
    mtgt = mask[:, 1:]
    logits = model(inp)
    loss_tok = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
    loss_tok = loss_tok.view(tgt.shape)
    loss = (loss_tok * mtgt).sum() / mtgt.sum().clamp(min=1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 150 == 0:
        print(f"  step {step:4d}  loss {loss.item():.4f}  elapsed {time.time()-t_start:.1f}s")
print(f"Training done. elapsed={time.time()-t_start:.1f}s")

# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------
model.eval()

@torch.no_grad()
def generate(prompt_ids, n_samples, temperature, max_new=8, rng_seed=0):
    """Batch-generate n_samples completions for one prompt. Returns list of
    generated id-lists (answer part only) and the first-token distribution
    entropy (nats) plus greedy length-normalized NLL."""
    g = torch.Generator().manual_seed(rng_seed)
    T = len(prompt_ids)
    x = torch.tensor([prompt_ids] * n_samples, dtype=torch.long)
    finished = [False] * n_samples
    gen_tokens = [[] for _ in range(n_samples)]
    first_tok_entropy = None
    for step in range(max_new):
        logits = model(x)[:, -1, :]  # (n_samples, vocab)
        probs = F.softmax(logits, dim=-1)
        if step == 0:
            # entropy of first-token distribution (identical across samples pre-sampling)
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

def parse_answer(tok_ids):
    s = "".join(itos[t] for t in tok_ids if t in itos)
    s = s.split(";")[0]
    try:
        return int(s)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Uncertainty measures for one question
# ---------------------------------------------------------------------------
def eval_question(q, true_c, n_samples, temperature, seed):
    prompt_ids = encode(q)
    gens, first_tok_ent = generate(prompt_ids, n_samples, temperature, rng_seed=seed)
    answers = [parse_answer(g) for g in gens]
    valid = [a for a in answers if a is not None]
    if not valid:
        maj_ans, maj_frac = None, 0.0
        se = math.log(max(len(answers), 1))
    else:
        cnt = Counter(valid)
        total = len(valid)
        maj_ans, maj_count = cnt.most_common(1)[0]
        maj_frac = maj_count / total
        se = -sum((c / total) * math.log(c / total) for c in cnt.values())
    nll = greedy_nll(prompt_ids)
    correct = (maj_ans == true_c)
    return {
        "semantic_entropy": se,
        "self_inconsistency": 1.0 - maj_frac,
        "token_entropy": first_tok_ent,
        "nll": nll,
        "correct": correct,
        "maj_ans": maj_ans,
        "true_c": true_c,
    }

# ---------------------------------------------------------------------------
# AUROC + bootstrap
# ---------------------------------------------------------------------------
def auroc(scores, labels):
    """labels: 1 = error (positive class), scores: higher = more uncertain."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = {}
    i = 0
    srt = sorted(range(len(scores)), key=lambda k: scores[k])
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
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc

def bootstrap_auroc_ci(scores, labels, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(scores)
    idx = list(range(n))
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(idx) for _ in range(n)]
        s = [scores[i] for i in samp]
        l = [labels[i] for i in samp]
        a = auroc(s, l)
        if not math.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi

def paired_bootstrap_diff_p(scoresA, scoresB, labels, n_boot=2000, seed=0):
    """Two-sided bootstrap p-value for AUROC(A) - AUROC(B) != 0, paired over
    the same resampled question indices."""
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
    if math.isnan(aA) or math.isnan(aB) or not diffs:
        return float("nan"), float("nan")
    obs = aA - aB
    # p-value: fraction of bootstrap diffs on the other side of 0 from obs, doubled
    if obs >= 0:
        p = 2 * (sum(1 for d in diffs if d <= 0) / len(diffs))
    else:
        p = 2 * (sum(1 for d in diffs if d >= 0) / len(diffs))
    return min(p, 1.0), obs

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
results = {}

def run_condition(n_q, n_samples, temperature, seed, ops=None, digs=None, label=""):
    rng = random.Random(seed)
    ops = ops or OPS
    digs = digs or DIGS
    rows = []
    for i in range(n_q):
        op = rng.choice(ops)
        nd = rng.choice(digs)
        q, ans, c = make_problem(op, nd, rng)
        r = eval_question(q, c, n_samples, temperature, seed=1000 + i)
        r["op"] = op
        r["nd"] = nd
        rows.append(r)
    n_err = sum(1 for r in rows if not r["correct"])
    print(f"[{label}] n_q={n_q} N={n_samples} T={temperature} err_rate={n_err/n_q:.3f} elapsed={time.time()-t_start:.1f}s")
    return rows

def summarize(rows, label):
    labels = [0 if r["correct"] else 1 for r in rows]
    out = {"label": label, "n": len(rows), "err_rate": sum(labels) / len(labels)}
    measures = {
        "semantic_entropy": [r["semantic_entropy"] for r in rows],
        "self_inconsistency": [r["self_inconsistency"] for r in rows],
        "token_entropy": [r["token_entropy"] for r in rows],
        "nll": [r["nll"] for r in rows],
    }
    aucs = {}
    for name, scores in measures.items():
        a = auroc(scores, labels)
        lo, hi = bootstrap_auroc_ci(scores, labels, n_boot=1000, seed=42)
        aucs[name] = {"auroc": a, "ci_lo": lo, "ci_hi": hi}
    out["aucs"] = aucs
    # pairwise comparisons vs semantic_entropy
    comps = {}
    for name in ["self_inconsistency", "token_entropy", "nll"]:
        p, obs = paired_bootstrap_diff_p(measures["semantic_entropy"], measures[name], labels,
                                          n_boot=1000, seed=7)
        comps[f"SE_vs_{name}"] = {"diff": obs, "p_raw": p}
    out["comparisons"] = comps
    return out

# Main condition: N=10, T=0.8, all ops/digits, larger n_q
MAIN_NQ = 300
main_rows = run_condition(MAIN_NQ, n_samples=10, temperature=0.8, seed=5, label="MAIN")
main_summary = summarize(main_rows, "MAIN (N=10,T=0.8,all ops/digits)")
results["main"] = main_summary

# Bonferroni correction across the 3 pairwise comparisons in main condition
m = len(main_summary["comparisons"])
for k, v in main_summary["comparisons"].items():
    v["p_bonferroni"] = min(v["p_raw"] * m, 1.0)

print(json.dumps(main_summary, indent=2))
print(f"elapsed so far: {time.time()-t_start:.1f}s")

# Ablation 1: per-operator breakdown (task-structure diversity axis)
op_results = {}
for op in OPS:
    rows = run_condition(120, n_samples=10, temperature=0.8, seed=11 + OPS.index(op),
                          ops=[op], label=f"OP={op}")
    op_results[op] = summarize(rows, f"op={op}")
results["by_operator"] = op_results
print(f"elapsed so far: {time.time()-t_start:.1f}s")

# Ablation 2: per-digit-length breakdown (difficulty axis)
dig_results = {}
for nd in DIGS:
    rows = run_condition(120, n_samples=10, temperature=0.8, seed=21 + nd, digs=[nd],
                          label=f"NDIG={nd}")
    dig_results[str(nd)] = summarize(rows, f"ndig={nd}")
results["by_digits"] = dig_results
print(f"elapsed so far: {time.time()-t_start:.1f}s")

# Ablation 3: number of samples N (cost axis)
n_results = {}
for N in [5, 10, 20]:
    rows = run_condition(150, n_samples=N, temperature=0.8, seed=31 + N, label=f"N={N}")
    n_results[str(N)] = summarize(rows, f"N={N}")
results["by_N"] = n_results
print(f"elapsed so far: {time.time()-t_start:.1f}s")

# Ablation 4: temperature
temp_results = {}
for T in [0.5, 1.0]:
    rows = run_condition(150, n_samples=10, temperature=T, seed=41 + int(T * 10), label=f"T={T}")
    temp_results[str(T)] = summarize(rows, f"T={T}")
results["by_temperature"] = temp_results
print(f"elapsed so far: {time.time()-t_start:.1f}s")

results["total_elapsed_sec"] = time.time() - t_start

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("DONE. total_elapsed=", results["total_elapsed_sec"])
