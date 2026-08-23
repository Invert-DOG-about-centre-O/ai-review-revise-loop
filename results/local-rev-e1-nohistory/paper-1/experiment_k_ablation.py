"""
K-sample ablation (K=8/16/32) for semantic entropy and self-consistency,
addressing round2 review weakness: "no evidence a larger K wouldn't close
the token-probability/sampling gap." Trains one model (same architecture/
budget regime as the main run, scaled to this revision's time budget),
then re-evaluates the sampling-based signals at increasing K on a shared
held-out set, holding the model and eval set fixed so only K varies.
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

t_script_start = time.time()

VOCAB = list("0123456789+=") + ["<PAD>", "<EOS>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
PAD_ID = STOI["<PAD>"]
EOS_ID = STOI["<EOS>"]
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 16

TRAIN_SECONDS_BUDGET = 200
BATCH_SIZE = 128
N_EVAL_K = 150
K_ABLATION = [8, 16, 32]
SEED = 42


def make_example(max_digits=3, rng=random):
    a = rng.randint(0, 10 ** max_digits - 1)
    b = rng.randint(0, 10 ** max_digits - 1)
    c = a + b
    return f"{a}+{b}=", f"{c}"


def encode(s):
    return [STOI[ch] for ch in s]


def make_batch(batch_size, rng):
    seqs, prompt_lens = [], []
    for _ in range(batch_size):
        prompt, answer = make_example(3, rng)
        full = prompt + answer
        ids = encode(full) + [EOS_ID]
        prompt_lens.append(len(prompt))
        seqs.append(ids)
    maxlen = max(len(s) for s in seqs)
    x = torch.full((batch_size, maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return x, prompt_lens


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


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = random.Random(SEED)

model = TinyGPT(VOCAB_SIZE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
t0 = time.time()
step, losses = 0, []
while time.time() - t0 < TRAIN_SECONDS_BUDGET:
    x, prompt_lens = make_batch(BATCH_SIZE, rng)
    inp = x[:, :-1]
    target = x[:, 1:].clone()
    loss_mask = torch.zeros_like(target, dtype=torch.bool)
    for i, pl in enumerate(prompt_lens):
        loss_mask[i, pl - 1:] = True
    target_masked = target.masked_fill(~loss_mask, -100)
    target_masked = target_masked.masked_fill(x[:, 1:] == PAD_ID, -100)
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target_masked.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    step += 1
final_loss = float(np.mean(losses[-50:])) if losses else float("nan")
print(f"Trained {step} steps, final loss {final_loss:.4f}, {time.time()-t0:.1f}s")


@torch.no_grad()
def greedy_generate_with_stats(prompt):
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    logprobs = []
    for step_i in range(8):
        logits = model(x)[0, -1]
        logp = F.log_softmax(logits, dim=-1)
        next_id = int(torch.argmax(logp).item())
        logprobs.append(logp[next_id].item())
        if next_id == EOS_ID:
            break
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    gen_ids = x[0, len(ids):].tolist()
    answer = "".join(ITOS[i] for i in gen_ids if i not in (EOS_ID, PAD_ID))
    return answer, logprobs


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
    return "".join(ITOS[i] for i in gen_ids if i not in (EOS_ID, PAD_ID))


def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None


eval_rng = random.Random(999)
eval_examples = [make_example(3, eval_rng) for _ in range(N_EVAL_K)]

# Greedy pass once (shared across all K)
greedy_info = []
for prompt, true_answer in eval_examples:
    true_val = int(true_answer)
    greedy_ans, logprobs = greedy_generate_with_stats(prompt)
    pred_val = parse_int(greedy_ans)
    mean_lp = float(np.mean(logprobs)) if logprobs else -100.0
    greedy_info.append(dict(prompt=prompt, true_val=true_val, pred_val=pred_val,
                             correct=(pred_val == true_val), mean_logprob=mean_lp))

greedy_acc = float(np.mean([g["correct"] for g in greedy_info]))
n_err = int(sum(1 for g in greedy_info if not g["correct"]))
print(f"Greedy accuracy on K-ablation eval set: {greedy_acc:.4f} ({n_err} errors / {N_EVAL_K})")

# For the largest K, draw samples once and reuse prefixes for smaller K (nested sampling)
K_MAX = max(K_ABLATION)
all_samples = []
for g in greedy_info:
    samples = [sample_generate(g["prompt"]) for _ in range(K_MAX)]
    all_samples.append([parse_int(s) for s in samples])

k_ablation_results = {}
for K in K_ABLATION:
    sem_entropies, agreements, majority_corrects = [], [], []
    for g, sample_vals_full in zip(greedy_info, all_samples):
        sample_vals = sample_vals_full[:K]
        valid_vals = [v for v in sample_vals if v is not None]
        agreement = float(np.mean([v == g["pred_val"] for v in sample_vals])) if sample_vals else 0.0
        if valid_vals:
            vals, counts = np.unique(valid_vals, return_counts=True)
            probs_s = counts / counts.sum()
            sem_entropy = float(-(probs_s * np.log(probs_s + 1e-12)).sum())
            majority_val = int(vals[np.argmax(counts)])
        else:
            sem_entropy = math.log(K)
            majority_val = None
        sem_entropies.append(sem_entropy)
        agreements.append(agreement)
        majority_corrects.append(majority_val == g["true_val"])

    error_arr = np.array([0 if g["correct"] else 1 for g in greedy_info])
    if 0 < error_arr.sum() < len(error_arr):
        sem_auroc = float(roc_auc_score(error_arr, np.array(sem_entropies)))
        selfcons_auroc = float(roc_auc_score(error_arr, -np.array(agreements)))
    else:
        sem_auroc = selfcons_auroc = float("nan")
    token_auroc = float(roc_auc_score(error_arr, -np.array([g["mean_logprob"] for g in greedy_info])))
    majority_acc = float(np.mean(majority_corrects))

    k_ablation_results[K] = dict(
        semantic_entropy_auroc=sem_auroc,
        self_consistency_auroc=selfcons_auroc,
        token_prob_auroc=token_auroc,
        majority_accuracy=majority_acc,
    )
    print(f"K={K}: sem_entropy_auroc={sem_auroc:.4f} self_cons_auroc={selfcons_auroc:.4f} "
          f"token_auroc={token_auroc:.4f} majority_acc={majority_acc:.4f} "
          f"gap_vs_sem={token_auroc-sem_auroc:.4f} gap_vs_selfcons={token_auroc-selfcons_auroc:.4f}")

out = dict(
    seed=SEED, train_steps=step, final_loss=final_loss,
    n_eval=N_EVAL_K, greedy_accuracy=greedy_acc, n_errors=n_err,
    k_values=K_ABLATION, k_ablation=k_ablation_results,
    total_script_time_s=round(time.time() - t_script_start, 1),
)
with open("results_k_ablation.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"TOTAL SCRIPT TIME: {time.time()-t_script_start:.1f}s")
