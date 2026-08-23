"""
Probabilistic uncertainty signals for LLM error detection: a controlled study.

We train a small GPT-style transformer from scratch (CPU, no internet access
available in this environment) on synthetic integer-addition strings
"a+b=c<EOS>". Because ground-truth correctness of "c" is exactly checkable by
re-computing a+b, we get noise-free labels for whether a generation is right
or wrong -- letting us cleanly evaluate several uncertainty/confidence
signals as error detectors, without depending on an external judge model.

Signals compared (computed from a single greedy decode unless noted):
  - mean token log-prob of the generated answer
  - min token log-prob of the generated answer  (weakest-link)
  - perplexity of the generated answer
  - entropy of the first answer-token's predictive distribution
  - sampling-based semantic entropy: entropy over K temperature samples'
    *parsed numeric values* (Kuhn et al. 2023 style, but exact instead of an
    NLI/LLM equivalence judge, which is possible here because equivalence of
    two answers is just integer equality)
  - self-consistency agreement: fraction of K samples equal to the greedy
    answer

We report each signal's AUROC for detecting whether the greedy decode is
wrong, an Expected Calibration Error (ECE) for the token-probability
confidence, and compare greedy accuracy vs. self-consistency (majority-vote)
accuracy.
"""
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t_start = time.time()
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Data: "a+b=c" strings over a small char vocabulary.
# ---------------------------------------------------------------------------
VOCAB = list("0123456789+=") + ["<PAD>", "<EOS>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
PAD_ID = STOI["<PAD>"]
EOS_ID = STOI["<EOS>"]
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 16  # "999+999=1998<EOS>" fits comfortably


def make_example(max_digits=3):
    a = random.randint(0, 10 ** max_digits - 1)
    b = random.randint(0, 10 ** max_digits - 1)
    c = a + b
    prompt = f"{a}+{b}="
    answer = f"{c}"
    return prompt, answer


def encode(s):
    return [STOI[ch] for ch in s]


def make_batch(batch_size, max_digits=3):
    seqs = []
    prompt_lens = []
    for _ in range(batch_size):
        prompt, answer = make_example(max_digits)
        full = prompt + answer
        ids = encode(full) + [EOS_ID]
        prompt_lens.append(len(prompt))
        seqs.append(ids)
    maxlen = max(len(s) for s in seqs)
    x = torch.full((batch_size, maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return x, prompt_lens


# ---------------------------------------------------------------------------
# Tiny GPT
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=3, d_ff=256, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1).to(idx.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)


model = TinyGPT(VOCAB_SIZE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ---------------------------------------------------------------------------
# Training loop, time-boxed.
# ---------------------------------------------------------------------------
TRAIN_SECONDS_BUDGET = 240
BATCH_SIZE = 128
step = 0
losses = []
while time.time() - t_start < TRAIN_SECONDS_BUDGET:
    x, prompt_lens = make_batch(BATCH_SIZE)
    inp = x[:, :-1]
    target = x[:, 1:].clone()
    # mask loss on prompt tokens and padding
    loss_mask = torch.zeros_like(target, dtype=torch.bool)
    for i, pl in enumerate(prompt_lens):
        loss_mask[i, pl - 1 :] = True
    target_masked = target.masked_fill(~loss_mask, -100)
    target_masked = target_masked.masked_fill(x[:, 1:] == PAD_ID, -100)

    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target_masked.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    step += 1
    if step % 200 == 0:
        print(f"step {step} loss {loss.item():.4f} elapsed {time.time()-t_start:.1f}s")

print(f"Training finished: {step} steps, final loss {np.mean(losses[-50:]):.4f}, "
      f"time {time.time()-t_start:.1f}s")

# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_with_stats(prompt):
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    logprobs = []
    first_token_entropy = None
    for step_i in range(8):
        logits = model(x)[0, -1]
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        ent = -(probs * logp).sum().item()
        if step_i == 0:
            first_token_entropy = ent
        next_id = int(torch.argmax(logp).item())
        logprobs.append(logp[next_id].item())
        if next_id == EOS_ID:
            break
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    gen_ids = x[0, len(ids):].tolist()
    answer = "".join(ITOS[i] for i in gen_ids if i not in (EOS_ID, PAD_ID))
    return answer, logprobs, first_token_entropy


@torch.no_grad()
def sample_generate(prompt, temperature=0.9):
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(8):
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        if next_id == EOS_ID:
            break
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    gen_ids = x[0, len(ids):].tolist()
    answer = "".join(ITOS[i] for i in gen_ids if i not in (EOS_ID, PAD_ID))
    return answer


def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Build eval set and collect signals.
# ---------------------------------------------------------------------------
N_EVAL = 400
K_SAMPLES = 8
eval_examples = [make_example(3) for _ in range(N_EVAL)]

records = []
t_eval_start = time.time()
for i, (prompt, true_answer) in enumerate(eval_examples):
    true_val = int(true_answer)
    greedy_ans, logprobs, first_ent = greedy_generate_with_stats(prompt)
    pred_val = parse_int(greedy_ans)
    correct = (pred_val == true_val)

    mean_lp = float(np.mean(logprobs)) if logprobs else -100.0
    min_lp = float(np.min(logprobs)) if logprobs else -100.0
    ppl = math.exp(-mean_lp)

    samples = [sample_generate(prompt) for _ in range(K_SAMPLES)]
    sample_vals = [parse_int(s) for s in samples]
    valid_vals = [v for v in sample_vals if v is not None]
    agreement = np.mean([v == pred_val for v in sample_vals]) if sample_vals else 0.0

    if valid_vals:
        vals, counts = np.unique(valid_vals, return_counts=True)
        probs_s = counts / counts.sum()
        sem_entropy = float(-(probs_s * np.log(probs_s + 1e-12)).sum())
        majority_val = int(vals[np.argmax(counts)])
    else:
        sem_entropy = math.log(K_SAMPLES)
        majority_val = None
    majority_correct = (majority_val == true_val)

    records.append(dict(
        prompt=prompt, true_val=true_val, pred_val=pred_val, correct=bool(correct),
        mean_logprob=mean_lp, min_logprob=min_lp, perplexity=ppl,
        first_token_entropy=first_ent, sample_agreement=float(agreement),
        semantic_entropy=sem_entropy, majority_val=majority_val,
        majority_correct=bool(majority_correct),
    ))
    if (i + 1) % 100 == 0:
        print(f"eval {i+1}/{N_EVAL} elapsed {time.time()-t_eval_start:.1f}s")

print(f"Eval finished in {time.time()-t_eval_start:.1f}s")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
correct_arr = np.array([r["correct"] for r in records], dtype=int)
error_arr = 1 - correct_arr
greedy_acc = correct_arr.mean()
majority_acc = np.mean([r["majority_correct"] for r in records])

signals = {
    "neg_mean_logprob": -np.array([r["mean_logprob"] for r in records]),
    "neg_min_logprob": -np.array([r["min_logprob"] for r in records]),
    "perplexity": np.array([r["perplexity"] for r in records]),
    "first_token_entropy": np.array([r["first_token_entropy"] for r in records]),
    "semantic_entropy": np.array([r["semantic_entropy"] for r in records]),
    "neg_sample_agreement": -np.array([r["sample_agreement"] for r in records]),
}

auroc_results = {}
if error_arr.sum() == 0 or error_arr.sum() == len(error_arr):
    auroc_results["note"] = "degenerate label distribution; AUROC undefined"
else:
    for name, scores in signals.items():
        try:
            auroc = roc_auc_score(error_arr, scores)
        except ValueError:
            auroc = float("nan")
        auroc_results[name] = float(auroc)

# ECE for token-probability confidence (confidence = exp(mean_logprob), i.e. per-token geometric mean prob)
def compute_ece(confidences, corrects, n_bins=10):
    confidences = np.array(confidences)
    corrects = np.array(corrects, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = corrects[mask].mean()
        conf_bin = confidences[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(acc_bin - conf_bin)
        bin_stats.append(dict(lo=float(lo), hi=float(hi), n=int(mask.sum()),
                               acc=float(acc_bin), conf=float(conf_bin)))
    return float(ece), bin_stats


token_conf = np.array([math.exp(r["mean_logprob"]) for r in records])
ece_token, bins_token = compute_ece(token_conf, correct_arr)

sample_conf = np.array([r["sample_agreement"] for r in records])
ece_sample, bins_sample = compute_ece(sample_conf, correct_arr)

results = dict(
    n_train_steps=step,
    train_time_s=round(time.time() - t_start, 1),
    n_eval=N_EVAL,
    k_samples=K_SAMPLES,
    greedy_accuracy=float(greedy_acc),
    self_consistency_majority_accuracy=float(majority_acc),
    auroc_error_detection=auroc_results,
    ece_token_probability_confidence=ece_token,
    ece_sample_agreement_confidence=ece_sample,
    reliability_bins_token_prob=bins_token,
    reliability_bins_sample_agreement=bins_sample,
)

print(json.dumps({k: v for k, v in results.items() if k not in
                   ("reliability_bins_token_prob", "reliability_bins_sample_agreement")}, indent=2))

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("records.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"TOTAL SCRIPT TIME: {time.time()-t_start:.1f}s")
