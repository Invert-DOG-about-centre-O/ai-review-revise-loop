"""
Revision experiment: multi-seed variance + K-ablation for the addition-task
UQ comparison in v1.md, addressing round1 review requests for (1) more than
one seed / significance quantification and (2) a K ablation for sampling-
based signals. Reuses the same TinyGPT architecture and signal definitions
as experiment.py, but with a smaller per-seed training budget so that 3
independent seeds fit in the revision time budget (the K-ablation is run
separately in experiment_k_ablation.py on a properly-trained model).
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

TRAIN_SECONDS_BUDGET = 130
BATCH_SIZE = 128
N_EVAL = 150
N_EVAL_K = 120
K_DEFAULT = 8
K_ABLATION = [8, 16, 32]
SEEDS = [0, 1, 2]
RUN_K_ABLATION = False


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


def train_model(seed, budget_s):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyGPT(VOCAB_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    step, losses = 0, []
    while time.time() - t0 < budget_s:
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
    return model, step, float(np.mean(losses[-50:])) if losses else float("nan")


@torch.no_grad()
def greedy_generate_with_stats(model, prompt):
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    logprobs, first_ent = [], None
    for step_i in range(8):
        logits = model(x)[0, -1]
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        ent = -(probs * logp).sum().item()
        if step_i == 0:
            first_ent = ent
        next_id = int(torch.argmax(logp).item())
        logprobs.append(logp[next_id].item())
        if next_id == EOS_ID:
            break
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    gen_ids = x[0, len(ids):].tolist()
    answer = "".join(ITOS[i] for i in gen_ids if i not in (EOS_ID, PAD_ID))
    return answer, logprobs, first_ent


@torch.no_grad()
def sample_generate(model, prompt, temperature=0.9):
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


def evaluate(model, eval_examples, k_samples):
    records = []
    for prompt, true_answer in eval_examples:
        true_val = int(true_answer)
        greedy_ans, logprobs, first_ent = greedy_generate_with_stats(model, prompt)
        pred_val = parse_int(greedy_ans)
        correct = (pred_val == true_val)
        mean_lp = float(np.mean(logprobs)) if logprobs else -100.0
        min_lp = float(np.min(logprobs)) if logprobs else -100.0
        ppl = math.exp(-mean_lp)
        samples = [sample_generate(model, prompt) for _ in range(k_samples)]
        sample_vals = [parse_int(s) for s in samples]
        valid_vals = [v for v in sample_vals if v is not None]
        agreement = float(np.mean([v == pred_val for v in sample_vals])) if sample_vals else 0.0
        if valid_vals:
            vals, counts = np.unique(valid_vals, return_counts=True)
            probs_s = counts / counts.sum()
            sem_entropy = float(-(probs_s * np.log(probs_s + 1e-12)).sum())
            majority_val = int(vals[np.argmax(counts)])
        else:
            sem_entropy = math.log(k_samples)
            majority_val = None
        records.append(dict(
            correct=bool(correct), mean_logprob=mean_lp, min_logprob=min_lp, perplexity=ppl,
            first_token_entropy=first_ent, sample_agreement=agreement, semantic_entropy=sem_entropy,
            majority_correct=bool(majority_val == true_val),
        ))
    return records


def compute_aurocs(records):
    correct_arr = np.array([r["correct"] for r in records], dtype=int)
    error_arr = 1 - correct_arr
    signals = {
        "neg_mean_logprob": -np.array([r["mean_logprob"] for r in records]),
        "neg_min_logprob": -np.array([r["min_logprob"] for r in records]),
        "perplexity": np.array([r["perplexity"] for r in records]),
        "first_token_entropy": np.array([r["first_token_entropy"] for r in records]),
        "semantic_entropy": np.array([r["semantic_entropy"] for r in records]),
        "neg_sample_agreement": -np.array([r["sample_agreement"] for r in records]),
    }
    out = {}
    if 0 < error_arr.sum() < len(error_arr):
        for name, scores in signals.items():
            try:
                out[name] = float(roc_auc_score(error_arr, scores))
            except ValueError:
                out[name] = float("nan")
    return out, int(error_arr.sum()), float(correct_arr.mean())


# ---------------------------------------------------------------------------
# Multi-seed runs
# ---------------------------------------------------------------------------
seed_results = []
seed_models = []
for seed in SEEDS:
    t0 = time.time()
    model, steps, final_loss = train_model(seed, TRAIN_SECONDS_BUDGET)
    seed_models.append(model)
    rng = random.Random(1000 + seed)
    eval_examples = [make_example(3, rng) for _ in range(N_EVAL)]
    records = evaluate(model, eval_examples, K_DEFAULT)
    aurocs, n_err, acc = compute_aurocs(records)
    majority_acc = float(np.mean([r["majority_correct"] for r in records]))
    seed_results.append(dict(seed=seed, steps=steps, final_loss=final_loss, greedy_accuracy=acc,
                              n_errors=n_err, majority_accuracy=majority_acc, aurocs=aurocs))
    print(f"seed {seed}: steps={steps} acc={acc:.4f} n_err={n_err} "
          f"aurocs={ {k: round(v,3) for k,v in aurocs.items()} } "
          f"elapsed={time.time()-t0:.1f}s total={time.time()-t_script_start:.1f}s")

# Aggregate mean/std across seeds for each signal (only seeds with valid AUROC)
signal_names = ["neg_mean_logprob", "neg_min_logprob", "perplexity",
                 "first_token_entropy", "semantic_entropy", "neg_sample_agreement"]
agg = {}
for name in signal_names:
    vals = [sr["aurocs"][name] for sr in seed_results if name in sr["aurocs"]]
    agg[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), n=len(vals), values=vals)

# Paired comparison: token-prob (neg_mean_logprob) vs each sampling-based signal, per seed
paired_diff_semantic = [sr["aurocs"]["neg_mean_logprob"] - sr["aurocs"]["semantic_entropy"]
                         for sr in seed_results if "neg_mean_logprob" in sr["aurocs"] and "semantic_entropy" in sr["aurocs"]]
paired_diff_selfcons = [sr["aurocs"]["neg_mean_logprob"] - sr["aurocs"]["neg_sample_agreement"]
                         for sr in seed_results if "neg_mean_logprob" in sr["aurocs"] and "neg_sample_agreement" in sr["aurocs"]]

print(f"\nPaired diff (token-prob - semantic entropy) across seeds: {paired_diff_semantic}, "
      f"mean={np.mean(paired_diff_semantic):.4f} std={np.std(paired_diff_semantic):.4f}")
print(f"Paired diff (token-prob - self-consistency) across seeds: {paired_diff_selfcons}, "
      f"mean={np.mean(paired_diff_selfcons):.4f} std={np.std(paired_diff_selfcons):.4f}")

# ---------------------------------------------------------------------------
# K ablation: reuse seed-0 model (already trained above), vary K
# ---------------------------------------------------------------------------
k_ablation_results = {}
if RUN_K_ABLATION:
    t0 = time.time()
    model0 = seed_models[0]
    rng = random.Random(2000)
    eval_examples_k = [make_example(3, rng) for _ in range(N_EVAL_K)]
    for K in K_ABLATION:
        records_k = evaluate(model0, eval_examples_k, K)
        aurocs_k, n_err_k, acc_k = compute_aurocs(records_k)
        majority_acc_k = float(np.mean([r["majority_correct"] for r in records_k]))
        k_ablation_results[K] = dict(
            semantic_entropy_auroc=aurocs_k.get("semantic_entropy"),
            self_consistency_auroc=aurocs_k.get("neg_sample_agreement"),
            majority_accuracy=majority_acc_k, greedy_accuracy=acc_k, n_errors=n_err_k,
        )
        print(f"K={K}: sem_entropy_auroc={aurocs_k.get('semantic_entropy'):.4f} "
              f"self_cons_auroc={aurocs_k.get('neg_sample_agreement'):.4f} "
              f"majority_acc={majority_acc_k:.4f} greedy_acc={acc_k:.4f} elapsed={time.time()-t0:.1f}s")

out = dict(
    train_seconds_budget_per_seed=TRAIN_SECONDS_BUDGET,
    n_eval=N_EVAL, k_default=K_DEFAULT, seeds=SEEDS,
    seed_results=seed_results, aggregate_auroc=agg,
    paired_diff_token_minus_semantic=paired_diff_semantic,
    paired_diff_token_minus_selfcons=paired_diff_selfcons,
    k_ablation=k_ablation_results,
    total_script_time_s=round(time.time() - t_script_start, 1),
)
with open("results_v2.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nTOTAL SCRIPT TIME: {time.time()-t_script_start:.1f}s")
