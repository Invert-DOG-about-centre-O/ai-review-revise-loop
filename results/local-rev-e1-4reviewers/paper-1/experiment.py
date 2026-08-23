"""
Semantic entropy vs. token-level predictive entropy vs. self-consistency as
cheap error/hallucination predictors, studied on a small Transformer language
model TRAINED FROM SCRATCH (no internet / pretrained-weight download needed)
on a synthetic arithmetic task.

Pipeline:
  1. Train a tiny char-level decoder-only Transformer on "a+b=c" / "a-b=c"
     strings for a short, fixed compute budget -- deliberately undertrained
     so it makes real, varied mistakes on held-out problems.
  2. For each held-out problem, greedy-decode the answer and record the mean
     per-token predictive entropy of that greedy decode.
  3. Draw K stochastic samples (temperature sampling) of the answer and
     compute:
       - semantic entropy: Shannon entropy over clusters of samples that
         parse to the same integer value (numeric answers -> exact-match
         clustering is a valid semantic equivalence relation).
       - self-consistency: fraction of the K samples agreeing with the
         greedy answer.
  4. Score each of the 3 signals as a predictor of whether the greedy answer
     is correct, using AUROC.
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
random.seed(0)
torch.manual_seed(0)

DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Vocabulary / data
# ---------------------------------------------------------------------------
VOCAB = list("0123456789+-=? ") + ["<pad>", "<bos>", "<eos>"]
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for c, i in stoi.items()}
PAD, BOS, EOS = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]


def make_example():
    a = random.randint(1, 99)
    b = random.randint(1, 99)
    op = random.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    prompt = f"{a}{op}{b}="
    target = f"{ans}"
    return prompt, target, ans


def encode(s):
    return [stoi[c] for c in s]


MAX_LEN = 16  # prompt + answer + eos, generously bounded


def make_batch(bs):
    """Full teacher-forcing batch: BOS + prompt + target + EOS, padded."""
    seqs = []
    for _ in range(bs):
        prompt, target, _ = make_example()
        ids = [BOS] + encode(prompt) + encode(target) + [EOS]
        seqs.append(ids)
    maxlen = max(len(s) for s in seqs)
    x = torch.full((bs, maxlen), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x


# ---------------------------------------------------------------------------
# Tiny decoder-only Transformer
# ---------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=4, n_layer=3, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(t, t, device=idx.device) * float("-inf"), diagonal=1)
        h = self.blocks(h, mask=mask)
        h = self.ln_f(h)
        return self.head(h)


model = TinyGPT(len(VOCAB), d_model=64, n_head=4, n_layer=3).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

# ---------------------------------------------------------------------------
# Train (short, fixed budget -> deliberately leaves real errors on the table)
# ---------------------------------------------------------------------------
TRAIN_STEPS = 1500
BATCH_SIZE = 64
print("Training tiny model...")
model.train()
for step in range(TRAIN_STEPS):
    x = make_batch(BATCH_SIZE)
    logits = model(x[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=PAD
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 500 == 0:
        print(f"step {step+1}/{TRAIN_STEPS} loss={loss.item():.3f} elapsed={time.time()-t0:.1f}s")
print(f"Training done in {time.time()-t0:.1f}s")
model.eval()

# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------
def prompt_ids(prompt):
    return [BOS] + encode(prompt)


def greedy_decode_with_entropy(prompt, max_new=6):
    ids = prompt_ids(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    entropies = []
    out_chars = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-12)).sum().item()
            entropies.append(ent)
            nxt = torch.argmax(probs).item()
            if nxt == EOS:
                break
            out_chars.append(itos[nxt])
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    text = "".join(out_chars)
    return text, sum(entropies) / len(entropies)


def sample_decode(prompt, temperature=1.0, max_new=6):
    ids = prompt_ids(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out_chars = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x)[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == EOS:
                break
            out_chars.append(itos[nxt])
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    return "".join(out_chars)


def parse_int(s):
    s = s.strip()
    if s == "" or s in ("-",):
        return None
    try:
        return int(s)
    except ValueError:
        # take a valid leading integer prefix if any
        j = 1 if s[0] == "-" else 0
        k = j
        while k < len(s) and s[k].isdigit():
            k += 1
        if k == j:
            return None
        try:
            return int(s[:k])
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Evaluation on held-out problems
# ---------------------------------------------------------------------------
N_PROBLEMS = 400
K_SAMPLES = 10
TEMPERATURE = 0.9

records = []
t_eval0 = time.time()
for i in range(N_PROBLEMS):
    prompt, target_str, true_ans = make_example()

    greedy_text, mean_tok_entropy = greedy_decode_with_entropy(prompt)
    greedy_ans = parse_int(greedy_text)
    correct = (greedy_ans == true_ans)

    sampled_texts = [sample_decode(prompt, TEMPERATURE) for _ in range(K_SAMPLES)]
    sampled_answers = [parse_int(t) for t in sampled_texts]

    counts = Counter(sampled_answers)
    total = len(sampled_answers)
    semantic_entropy = -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())
    self_consistency = sum(1 for a in sampled_answers if a == greedy_ans) / total

    records.append({
        "prompt": prompt,
        "true_ans": true_ans,
        "greedy_text": greedy_text,
        "greedy_ans": greedy_ans,
        "correct": correct,
        "mean_tok_entropy": mean_tok_entropy,
        "semantic_entropy": semantic_entropy,
        "self_consistency": self_consistency,
        "sampled_answers": sampled_answers,
    })

    if (i + 1) % 100 == 0:
        print(f"eval {i+1}/{N_PROBLEMS} elapsed={time.time()-t_eval0:.1f}s")

t_eval = time.time() - t_eval0
print(f"Eval loop took {t_eval:.1f}s")

with open("raw_results.json", "w") as f:
    json.dump(records, f, indent=1)


# ---------------------------------------------------------------------------
# AUROC analysis
# ---------------------------------------------------------------------------
def auroc(scores, labels):
    pairs = list(zip(scores, labels))
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


labels = [1 if r["correct"] else 0 for r in records]
n_correct = sum(labels)

score_neg_tok_entropy = [-r["mean_tok_entropy"] for r in records]
score_neg_sem_entropy = [-r["semantic_entropy"] for r in records]
score_self_consistency = [r["self_consistency"] for r in records]

results = {
    "n_problems": N_PROBLEMS,
    "k_samples": K_SAMPLES,
    "temperature": TEMPERATURE,
    "train_steps": TRAIN_STEPS,
    "greedy_accuracy": n_correct / N_PROBLEMS,
    "auroc_token_entropy": auroc(score_neg_tok_entropy, labels),
    "auroc_semantic_entropy": auroc(score_neg_sem_entropy, labels),
    "auroc_self_consistency": auroc(score_self_consistency, labels),
    "total_time_sec": time.time() - t0,
}
print(json.dumps(results, indent=2))

with open("results_summary.json", "w") as f:
    json.dump(results, f, indent=2)
