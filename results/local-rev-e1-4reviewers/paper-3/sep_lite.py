"""
Lightweight Semantic-Entropy-Probe (SEP) baseline (round-3 revision).
Kossen et al. (2024) train a probe on hidden states to predict semantic
entropy from a single forward pass, avoiding K-sample generation at test
time. We implement a minimal version: linear regression from the model's
final-layer hidden state at the last prompt token to the K=8 semantic
entropy target, fit on half the eval questions and tested (as an AUROC
wrongness predictor) on the other half, disjoint from probe training.
"""
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

T0 = time.time()

VOCAB = list("0123456789+=. ") + ["<pad>", "<eos>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
PAD, EOS = STOI["<pad>"], STOI["<eos>"]
MAXLEN = 12


def make_example(rng):
    a = rng.randint(1, 98)
    b = rng.randint(1, 98)
    return f"{a}+{b}=", str(a + b), a + b


def encode(s):
    return [STOI[c] for c in s]


def build_batch(rng, bs):
    seqs = []
    for _ in range(bs):
        prompt, ans, _ = make_example(rng)
        seqs.append(encode(prompt) + encode(ans) + [EOS])
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

    def forward(self, x, return_hidden=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.enc(h, mask=mask)
        if return_hidden:
            return self.head(h), h
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
    return ans, (float(np.mean(logps)) if logps else -20.0)


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


@torch.no_grad()
def prompt_hidden(model, prompt_ids):
    x = torch.tensor([prompt_ids])
    _, h = model(x, return_hidden=True)
    return h[0, -1, :].numpy()  # hidden state at last prompt token


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


def auroc(scores, is_positive):
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[is_positive], scores[~is_positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


N_QUESTIONS = 400
K_SAMPLES = 8
TEMPERATURE = 1.0
TRAIN_STEPS = 600
BATCH_SIZE = 64
SEEDS = [0, 1, 2]

seed_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = TinyTransformerLM(len(VOCAB), maxlen=MAXLEN)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng_train = random.Random(seed + 1)
    model.train()
    for step in range(TRAIN_STEPS):
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
    correct_l, logp_l, sem_l, hid_l = [], [], [], []
    for i in range(N_QUESTIONS):
        prompt, ans_str, gt = make_example(rng_eval)
        pids = encode(prompt)
        greedy_ans_str, mean_logp = greedy_decode(model, pids)
        greedy_ans = parse_int(greedy_ans_str)
        correct = (greedy_ans == gt)
        samples = [parse_int(sample_decode(model, pids, TEMPERATURE)) for _ in range(K_SAMPLES)]
        sem_ent = semantic_entropy(samples)
        hid = prompt_hidden(model, pids)
        correct_l.append(correct)
        logp_l.append(mean_logp)
        sem_l.append(sem_ent)
        hid_l.append(hid)

    correct_arr = np.array(correct_l, dtype=bool)
    wrong_arr = ~correct_arr
    logp_arr = np.array(logp_l)
    sem_arr = np.array(sem_l)
    hid_arr = np.stack(hid_l)

    # split: first half trains the probe (labels = true K=8 semantic entropy),
    # second half is held out for AUROC evaluation (probe never sees these labels)
    half = N_QUESTIONS // 2
    probe = LinearRegression().fit(hid_arr[:half], sem_arr[:half])
    pred_sem_test = probe.predict(hid_arr[half:])

    a_probe = auroc(pred_sem_test, wrong_arr[half:])
    a_logp_test = auroc(-logp_arr[half:], wrong_arr[half:])
    a_sem_test = auroc(sem_arr[half:], wrong_arr[half:])
    acc = float(correct_arr.mean())

    seed_results.append(dict(seed=seed, accuracy=acc, auroc_probe=a_probe,
                              auroc_logp=a_logp_test, auroc_sem_ent_true=a_sem_test))
    print(f"seed={seed} acc={acc:.3f} AUROC(held-out 200) probe={a_probe:.3f} "
          f"logp={a_logp_test:.3f} true_sem_ent(K=8)={a_sem_test:.3f} t={time.time()-T0:.1f}s")

probe_vals = np.array([r["auroc_probe"] for r in seed_results])
logp_vals = np.array([r["auroc_logp"] for r in seed_results])
sem_vals = np.array([r["auroc_sem_ent_true"] for r in seed_results])
print(f"\nMean AUROC (held-out half, n={len(SEEDS)} seeds): "
      f"probe(1 pass)={probe_vals.mean():.3f} logp(1 pass)={logp_vals.mean():.3f} "
      f"true_sem_ent(K=8 passes)={sem_vals.mean():.3f}")

with open("sep_lite_results.json", "w") as f:
    json.dump(dict(seeds=SEEDS, results=seed_results,
                    probe_mean=float(probe_vals.mean()), logp_mean=float(logp_vals.mean()),
                    sem_ent_mean=float(sem_vals.mean())), f, indent=2)
print(f"Total time: {time.time()-T0:.1f}s")
