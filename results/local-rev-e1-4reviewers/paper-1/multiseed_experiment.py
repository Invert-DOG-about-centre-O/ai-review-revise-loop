"""
Addresses reviewer requests: (1) bootstrap CIs on AUROC differences,
(2) multi-seed variance of the token-entropy vs. sampling-based-signal gap.

Refactors experiment.py's pipeline into a function run_seed(seed) and runs it
for 5 seeds, then bootstraps AUROC differences (paired, per-example resampling)
for the seed=0 run to match what's reported in the paper.
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = list("0123456789+-=? ") + ["<pad>", "<bos>", "<eos>"]
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for c, i in stoi.items()}
PAD, BOS, EOS = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]
MAX_LEN = 16
N_PROBLEMS = 400
K_SAMPLES = 10
TEMPERATURE = 0.9
TRAIN_STEPS = 1500
BATCH_SIZE = 64


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


def make_example(rng):
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    prompt = f"{a}{op}{b}="
    target = f"{ans}"
    return prompt, target, ans


def encode(s):
    return [stoi[c] for c in s]


def make_batch(bs, rng):
    seqs = []
    for _ in range(bs):
        prompt, target, _ = make_example(rng)
        ids = [BOS] + encode(prompt) + encode(target) + [EOS]
        seqs.append(ids)
    maxlen = max(len(s) for s in seqs)
    x = torch.full((bs, maxlen), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x


def parse_int(s):
    s = s.strip()
    if s == "" or s in ("-",):
        return None
    try:
        return int(s)
    except ValueError:
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


def run_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + 10000)  # separate stream for data gen determinism per seed
    model = TinyGPT(len(VOCAB)).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(TRAIN_STEPS):
        x = make_batch(BATCH_SIZE, rng)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=PAD
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

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

    records = []
    for i in range(N_PROBLEMS):
        prompt, target_str, true_ans = make_example(rng)
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
            "correct": correct,
            "mean_tok_entropy": mean_tok_entropy,
            "semantic_entropy": semantic_entropy,
            "self_consistency": self_consistency,
        })

    labels = [1 if r["correct"] else 0 for r in records]
    n_correct = sum(labels)
    score_tok = [-r["mean_tok_entropy"] for r in records]
    score_sem = [-r["semantic_entropy"] for r in records]
    score_sc = [r["self_consistency"] for r in records]
    return {
        "seed": seed,
        "greedy_accuracy": n_correct / N_PROBLEMS,
        "auroc_token_entropy": auroc(score_tok, labels),
        "auroc_semantic_entropy": auroc(score_sem, labels),
        "auroc_self_consistency": auroc(score_sc, labels),
    }, records


def bootstrap_ci(records, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(records)
    diffs_sem_tok = []
    diffs_sc_tok = []
    diffs_sc_sem = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [records[i] for i in idx]
        labels = [1 if r["correct"] else 0 for r in sub]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue
        a_tok = auroc([-r["mean_tok_entropy"] for r in sub], labels)
        a_sem = auroc([-r["semantic_entropy"] for r in sub], labels)
        a_sc = auroc([r["self_consistency"] for r in sub], labels)
        diffs_sem_tok.append(a_sem - a_tok)
        diffs_sc_tok.append(a_sc - a_tok)
        diffs_sc_sem.append(a_sc - a_sem)

    def summarize(diffs):
        diffs.sort()
        n = len(diffs)
        lo = diffs[int(0.025 * n)]
        hi = diffs[int(0.975 * n)]
        frac_pos = sum(1 for d in diffs if d > 0) / n
        return {"mean_diff": sum(diffs) / n, "ci95": [lo, hi], "frac_gt_0": frac_pos}

    return {
        "semantic_minus_token": summarize(diffs_sem_tok),
        "selfconsistency_minus_token": summarize(diffs_sc_tok),
        "selfconsistency_minus_semantic": summarize(diffs_sc_sem),
    }


if __name__ == "__main__":
    t0 = time.time()
    seed_results = []
    seed0_records = None
    for seed in [0, 1, 2, 3, 4]:
        summary, records = run_seed(seed)
        print(json.dumps(summary), f"elapsed={time.time()-t0:.1f}s")
        seed_results.append(summary)
        if seed == 0:
            seed0_records = records

    ci = bootstrap_ci(seed0_records, n_boot=2000, seed=0)
    print("Bootstrap CI (seed 0, 2000 resamples):")
    print(json.dumps(ci, indent=2))

    out = {"seed_results": seed_results, "bootstrap_ci_seed0": ci, "total_time_sec": time.time() - t0}
    with open("multiseed_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Total time: {time.time()-t0:.1f}s")
