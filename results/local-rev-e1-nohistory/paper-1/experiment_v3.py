"""
Revision experiment addressing round3 review: (1) extend the seed sweep
(experiment_v2.py protocol) to include higher-accuracy seeds under the SAME
protocol used for seeds 0-2, so the K-ablation reversal can be checked
against accuracy level without changing training budget/eval-set-size
confounds (review Q3, Limitation 3); (2) bootstrap confidence intervals on
the paired AUROC gaps, including in the low-error regime the K-ablation
reversal rests on (review Q2).
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

BATCH_SIZE = 128
N_EVAL = 150
K_DEFAULT = 8
N_BOOT = 2000
# New seeds use a LARGER budget than seeds 0-2 (130s) specifically to reach
# the high-accuracy regime the K-ablation model (88%) occupied, so we can
# check the reversal within the identical seed-sweep protocol.
NEW_SEEDS_BUDGETS = [(3, 200), (4, 230)]


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
        greedy_ans, logprobs = greedy_generate_with_stats(model, prompt)
        pred_val = parse_int(greedy_ans)
        correct = (pred_val == true_val)
        mean_lp = float(np.mean(logprobs)) if logprobs else -100.0
        samples = [sample_generate(model, prompt) for _ in range(k_samples)]
        sample_vals = [parse_int(s) for s in samples]
        valid_vals = [v for v in sample_vals if v is not None]
        agreement = float(np.mean([v == pred_val for v in sample_vals])) if sample_vals else 0.0
        if valid_vals:
            vals, counts = np.unique(valid_vals, return_counts=True)
            probs_s = counts / counts.sum()
            sem_entropy = float(-(probs_s * np.log(probs_s + 1e-12)).sum())
        else:
            sem_entropy = math.log(k_samples)
        records.append(dict(correct=bool(correct), mean_logprob=mean_lp,
                             sample_agreement=agreement, semantic_entropy=sem_entropy))
    return records


def aurocs_from_records(records):
    error_arr = np.array([0 if r["correct"] else 1 for r in records])
    neg_mean_lp = -np.array([r["mean_logprob"] for r in records])
    sem_ent = np.array([r["semantic_entropy"] for r in records])
    neg_agree = -np.array([r["sample_agreement"] for r in records])
    if not (0 < error_arr.sum() < len(error_arr)):
        return None
    return dict(
        token_prob=float(roc_auc_score(error_arr, neg_mean_lp)),
        semantic_entropy=float(roc_auc_score(error_arr, sem_ent)),
        self_consistency=float(roc_auc_score(error_arr, neg_agree)),
    )


def bootstrap_gap_ci(records, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap CI on (token_prob AUROC - other AUROC), resampling
    eval examples with replacement. Returns None if degenerate."""
    rng = np.random.RandomState(seed)
    n = len(records)
    error_arr = np.array([0 if r["correct"] else 1 for r in records])
    neg_mean_lp = -np.array([r["mean_logprob"] for r in records])
    sem_ent = np.array([r["semantic_entropy"] for r in records])
    neg_agree = -np.array([r["sample_agreement"] for r in records])
    diffs_sem, diffs_sc = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        e = error_arr[idx]
        if not (0 < e.sum() < n):
            continue
        try:
            a_tok = roc_auc_score(e, neg_mean_lp[idx])
            a_sem = roc_auc_score(e, sem_ent[idx])
            a_sc = roc_auc_score(e, neg_agree[idx])
        except ValueError:
            continue
        diffs_sem.append(a_tok - a_sem)
        diffs_sc.append(a_tok - a_sc)
    if len(diffs_sem) < n_boot * 0.5:
        return None
    return dict(
        vs_semantic_entropy=dict(lo=float(np.percentile(diffs_sem, 2.5)),
                                  hi=float(np.percentile(diffs_sem, 97.5)),
                                  median=float(np.median(diffs_sem))),
        vs_self_consistency=dict(lo=float(np.percentile(diffs_sc, 2.5)),
                                  hi=float(np.percentile(diffs_sc, 97.5)),
                                  median=float(np.median(diffs_sc))),
        n_boot_valid=len(diffs_sem),
    )


# ---------------------------------------------------------------------------
# Train new higher-accuracy seeds under the identical eval protocol as
# experiment_v2.py (N_EVAL=150, K=8), only the training budget differs.
# ---------------------------------------------------------------------------
new_seed_results = []
for seed, budget in NEW_SEEDS_BUDGETS:
    t0 = time.time()
    model, steps, final_loss = train_model(seed, budget)
    rng = random.Random(1000 + seed)
    eval_examples = [make_example(3, rng) for _ in range(N_EVAL)]
    records = evaluate(model, eval_examples, K_DEFAULT)
    aurocs = aurocs_from_records(records)
    n_err = int(sum(1 for r in records if not r["correct"]))
    acc = float(np.mean([r["correct"] for r in records]))
    ci = bootstrap_gap_ci(records, seed=seed)
    new_seed_results.append(dict(seed=seed, budget_s=budget, steps=steps, final_loss=final_loss,
                                  greedy_accuracy=acc, n_errors=n_err, aurocs=aurocs, bootstrap_ci=ci))
    print(f"seed {seed} (budget={budget}s): steps={steps} acc={acc:.4f} n_err={n_err} "
          f"aurocs={aurocs} elapsed={time.time()-t0:.1f}s total={time.time()-t_script_start:.1f}s")

# ---------------------------------------------------------------------------
# Also bootstrap-CI the original seeds 0-2 (results_v2.json) by retraining
# is expensive; instead re-derive CIs is not possible without raw records
# there, so we only report CIs for the newly trained seeds here plus note
# this in the writeup. (records for seeds 0-2 were not persisted.)
# ---------------------------------------------------------------------------

out = dict(
    n_eval=N_EVAL, k_default=K_DEFAULT, n_boot=N_BOOT,
    new_seeds_budgets=NEW_SEEDS_BUDGETS,
    new_seed_results=new_seed_results,
    total_script_time_s=round(time.time() - t_script_start, 1),
)
with open("results_v3.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nTOTAL SCRIPT TIME: {time.time()-t_script_start:.1f}s")
