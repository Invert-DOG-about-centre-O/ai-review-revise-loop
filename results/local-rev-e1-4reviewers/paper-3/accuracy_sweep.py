"""
Designed accuracy sweep (round-3 revision): instead of relying on the
natural, non-uniform accuracy spread produced by 15 random seeds at a fixed
600 training steps, we deliberately vary the number of training steps to
place models at controlled points across the accuracy range, with 2 seeds
per step-budget. This tests whether the ECE-gap reversal (self-consistency
loses its calibration edge at high accuracy) is a stable function of
accuracy, or an artifact of which few random seeds happened to land at the
high-accuracy end in the natural-spread replication.
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

VOCAB = list("0123456789+=. ") + ["<pad>", "<eos>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
PAD, EOS = STOI["<pad>"], STOI["<eos>"]
MAXLEN = 12


def make_example(rng):
    a = rng.randint(1, 98)
    b = rng.randint(1, 98)
    s = f"{a}+{b}="
    ans = str(a + b)
    return s, ans, a + b


def encode(s):
    return [STOI[c] for c in s]


def build_batch(rng, bs):
    seqs = []
    for _ in range(bs):
        prompt, ans, _ = make_example(rng)
        full = encode(prompt) + encode(ans) + [EOS]
        seqs.append(full)
    L = max(len(s) for s in seqs)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x


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


@torch.no_grad()
def greedy_decode(model, prompt_ids, max_new=6):
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
def sample_decode(model, prompt_ids, temperature=1.0, max_new=6):
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


def self_consistency_conf(answers):
    n = len(answers)
    counts = {}
    for a in answers:
        key = a if a is not None else "PARSE_FAIL"
        counts[key] = counts.get(key, 0) + 1
    _, modal_count = max(counts.items(), key=lambda kv: kv[1])
    return modal_count / n


def ece(confidences, corrects, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = corrects[mask].mean()
        conf_bin = confidences[mask].mean()
        e += (mask.sum() / total) * abs(acc_bin - conf_bin)
    return float(e)


N_QUESTIONS = 150  # smaller than main 15-seed run, to fit time budget
K_SAMPLES = 8
TEMPERATURE = 1.0
BATCH_SIZE = 64
STEP_BUDGETS = [100, 400, 900, 1800]
SEEDS_PER_BUDGET = [0]

results = []
for steps in STEP_BUDGETS:
    for s in SEEDS_PER_BUDGET:
        seed = steps * 10 + s  # unique seed per (budget, replicate)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = TinyTransformerLM(len(VOCAB), maxlen=MAXLEN)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        rng_train = random.Random(seed + 1)
        model.train()
        for step in range(steps):
            x = build_batch(rng_train, BATCH_SIZE)
            logits = model(x[:, :-1])
            targets = x[:, 1:].clone()
            targets[targets == PAD] = -100
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    targets.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()

        rng_eval = random.Random(42 + seed)
        correct_l, logp_l, sc_l = [], [], []
        for i in range(N_QUESTIONS):
            prompt, ans_str, gt = make_example(rng_eval)
            pids = encode(prompt)
            greedy_ans_str, mean_logp = greedy_decode(model, pids)
            greedy_ans = parse_int(greedy_ans_str)
            correct = (greedy_ans == gt)
            samples = [parse_int(sample_decode(model, pids, TEMPERATURE)) for _ in range(K_SAMPLES)]
            sc_conf = self_consistency_conf(samples)
            correct_l.append(correct)
            logp_l.append(mean_logp)
            sc_l.append(sc_conf)

        correct_arr = np.array(correct_l, dtype=bool)
        acc = float(correct_arr.mean())
        logp_conf = np.exp(np.array(logp_l))
        ece_logp = ece(logp_conf, correct_arr)
        ece_sc = ece(np.array(sc_l), correct_arr)

        results.append(dict(steps=steps, seed=seed, accuracy=acc,
                             ece_logp=ece_logp, ece_self_cons=ece_sc,
                             ece_gap=ece_logp - ece_sc))
        print(f"steps={steps:5d} seed={seed:5d} acc={acc:.3f} "
              f"ece_logp={ece_logp:.3f} ece_sc={ece_sc:.3f} gap={ece_logp-ece_sc:+.3f} "
              f"t={time.time()-T0:.1f}s")

acc_vals = np.array([r["accuracy"] for r in results])
gap_vals = np.array([r["ece_gap"] for r in results])
corr = float(np.corrcoef(acc_vals, gap_vals)[0, 1])
n_reversed = int((gap_vals < 0).sum())  # logp better calibrated than self-cons
print(f"\nAccuracy range: {acc_vals.min():.3f} - {acc_vals.max():.3f}")
print(f"corr(accuracy, ece_logp - ece_self_cons) = {corr:.3f}")
print(f"logp better-calibrated (gap<0) in {n_reversed}/{len(results)} designed runs")
# crude reversal threshold: smallest accuracy at which gap < 0, among sorted results
sorted_by_acc = sorted(results, key=lambda r: r["accuracy"])
reversal_acc = None
for r in sorted_by_acc:
    if r["ece_gap"] < 0:
        reversal_acc = r["accuracy"]
        break
print(f"lowest-accuracy run with logp better calibrated: acc={reversal_acc}")

with open("accuracy_sweep_results.json", "w") as f:
    json.dump(dict(results=results, corr_acc_gap=corr,
                    n_reversed=n_reversed, n_total=len(results),
                    reversal_acc=reversal_acc), f, indent=2)
print(f"Total time: {time.time()-T0:.1f}s")
