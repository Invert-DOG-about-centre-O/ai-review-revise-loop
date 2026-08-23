"""
Cost-accuracy comparison of uncertainty signals for detecting incorrect
answers ("hallucinations") in a small char-level Transformer LM trained
from scratch (offline, CPU-only) on a two-digit addition task.

We deliberately under-train / under-size the model so it gets a
non-trivial fraction of problems wrong -- this gives us a realistic
mix of correct/incorrect greedy answers to test uncertainty signals on.

Signals compared:
  1. first_token_entropy: entropy of the model's distribution over the
     first answer character (1 forward pass).
  2. mean_logp: length-normalized log-probability of the greedy answer
     string (comes for free with greedy decoding, 1 pass equivalent).
  3. self_consistency_conf: fraction of K sampled completions that agree
     with the modal parsed answer (K forward passes' worth of decoding).
  4. semantic_entropy: Shannon entropy of the distribution over distinct
     *parsed integer* answers among K samples (exact-value clustering is
     semantic clustering here, since two strings denote the same meaning
     iff they parse to the same integer).

All are evaluated as predictors of "greedy answer is wrong" via AUROC.
self_consistency_conf and a logp-derived confidence are also evaluated
as calibrated probabilities via ECE and Brier score.
"""
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()


def elapsed():
    return time.time() - T0


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- data ----
VOCAB = list("0123456789+=. ") + ["<pad>", "<eos>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
PAD, EOS = STOI["<pad>"], STOI["<eos>"]
MAXLEN = 12  # "12+34=46" + eos, generously padded


def make_example(rng):
    a = rng.randint(1, 98)
    b = rng.randint(1, 98)
    s = f"{a}+{b}="
    ans = str(a + b)
    return s, ans, a + b


def encode(s):
    return [STOI[c] for c in s]


def build_batch(rng, bs):
    """Full teacher-forcing sequences: prompt+answer+eos, padded."""
    seqs, prompt_lens = [], []
    for _ in range(bs):
        prompt, ans, _ = make_example(rng)
        full = encode(prompt) + encode(ans) + [EOS]
        prompt_lens.append(len(prompt))
        seqs.append(full)
    L = max(len(s) for s in seqs)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, prompt_lens


# --------------------------------------------------------------- model ----
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=48, nhead=4, nlayers=2, maxlen=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab_size)
        self.maxlen = maxlen

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.enc(h, mask=mask)
        return self.head(h)


model = TinyTransformerLM(len(VOCAB), maxlen=MAXLEN)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

# Under-train on purpose: small model + limited steps -> imperfect accuracy.
TRAIN_STEPS = 600
BATCH_SIZE = 64
rng_train = random.Random(1)
print("Training tiny char-transformer on 2-digit addition ...")
model.train()
for step in range(TRAIN_STEPS):
    x, plens = build_batch(rng_train, BATCH_SIZE)
    logits = model(x[:, :-1])
    targets = x[:, 1:].clone()
    targets[targets == PAD] = -100
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}  t={elapsed():.1f}s")
model.eval()
print(f"Training done at t={elapsed():.1f}s")


# ------------------------------------------------------------ decoding ----
@torch.no_grad()
def first_answer_char_entropy(prompt_ids):
    x = torch.tensor([prompt_ids])
    logits = model(x)[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()


@torch.no_grad()
def greedy_decode(prompt_ids, max_new=6):
    ids = list(prompt_ids)
    logps = []
    for _ in range(max_new):
        x = torch.tensor([ids])
        logits = model(x)[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        nid = int(torch.argmax(probs).item())
        logps.append(math.log(max(probs[nid].item(), 1e-12)))
        if nid == EOS:
            break
        ids.append(nid)
    ans = "".join(ITOS[i] for i in ids[len(prompt_ids):])
    mean_logp = float(np.mean(logps)) if logps else -20.0
    return ans, mean_logp


@torch.no_grad()
def sample_decode(prompt_ids, temperature=1.0, max_new=6):
    ids = list(prompt_ids)
    for _ in range(max_new):
        x = torch.tensor([ids])
        logits = model(x)[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        nid = int(torch.multinomial(probs, 1).item())
        if nid == EOS:
            break
        ids.append(nid)
    return "".join(ITOS[i] for i in ids[len(prompt_ids):])


def parse_int(s):
    s = s.strip()
    if s == "" or not all(c.isdigit() for c in s):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def semantic_entropy(answers):
    n = len(answers)
    counts = {}
    for a in answers:
        key = a if a is not None else "PARSE_FAIL"
        counts[key] = counts.get(key, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p)
    return ent


def self_consistency_conf(answers):
    n = len(answers)
    counts = {}
    for a in answers:
        key = a if a is not None else "PARSE_FAIL"
        counts[key] = counts.get(key, 0) + 1
    _, modal_count = max(counts.items(), key=lambda kv: kv[1])
    return modal_count / n


# -------------------------------------------------------------- eval ------
N_QUESTIONS = 400
K_SAMPLES = 8
TEMPERATURE = 1.0
rng_eval = random.Random(42)

records = []
eval_start = elapsed()
for i in range(N_QUESTIONS):
    prompt, ans_str, gt = make_example(rng_eval)
    pids = encode(prompt)

    ent1 = first_answer_char_entropy(pids)
    greedy_ans_str, mean_logp = greedy_decode(pids)
    greedy_ans = parse_int(greedy_ans_str)
    correct = (greedy_ans == gt)

    samples = [parse_int(sample_decode(pids, TEMPERATURE)) for _ in range(K_SAMPLES)]
    sem_ent = semantic_entropy(samples)
    sc_conf = self_consistency_conf(samples)

    records.append(dict(
        prompt=prompt, gt=gt, greedy_ans=greedy_ans, correct=bool(correct),
        first_token_entropy=ent1, mean_logp=mean_logp,
        semantic_entropy=sem_ent, self_consistency_conf=sc_conf,
        samples=samples,
    ))
    if (i + 1) % 100 == 0:
        print(f"  eval {i+1}/{N_QUESTIONS}  t={elapsed():.1f}s")

acc = float(np.mean([r["correct"] for r in records]))
print(f"Greedy accuracy: {acc:.3f}")
print(f"Eval time: {elapsed()-eval_start:.1f}s, total: {elapsed():.1f}s")

with open("raw_results.json", "w") as f:
    json.dump(dict(
        n_questions=N_QUESTIONS, k_samples=K_SAMPLES, temperature=TEMPERATURE,
        train_steps=TRAIN_STEPS, accuracy=acc, total_time_sec=elapsed(),
        records=records,
    ), f, indent=2)

print("Saved raw_results.json")
