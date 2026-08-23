"""
Edit-distance clustering ablation, requested independently by all 4 round-3
reviewers: does clustering self-consistency samples by edit distance (rather
than exact string match) close the maxprob-vs-self-consistency AUROC gap?

Reuses the exact seed-0 training pipeline from experiment.py, then recomputes
self-consistency agreement/entropy under edit-distance clustering (threshold
<=1 edit on the 3-digit answer, i.e. off-by-one-digit strings count as the
same cluster) instead of exact match, and compares AUROC to the maxprob
baseline and to the original exact-match self-consistency.
"""
import time
import random
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t0 = time.time()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

CHARS = list("0123456789+=")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
IN_LEN = 6
OUT_LEN = 3
SEQ_LEN = IN_LEN + OUT_LEN


def make_example():
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    s = a + b
    text = f"{a:02d}+{b:02d}={s:03d}"
    return text, a, b, s


def encode(text):
    return [STOI[c] for c in text]


def batch(n):
    return torch.tensor([encode(make_example()[0]) for _ in range(n)], dtype=torch.long)


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, seq_len=SEQ_LEN, d=64, nhead=4, layers=2, ff=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                                batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.ln(h)
        return self.head(h)


device = "cpu"
model = TinyGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
TRAIN_STEPS = 260
BATCH = 128

for step in range(TRAIN_STEPS):
    x = batch(BATCH).to(device)
    inp, tgt = x[:, :-1], x[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(logits[:, IN_LEN - 1:].reshape(-1, VOCAB), tgt[:, IN_LEN - 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
N_TEST = 700
K_SAMPLES = 20
TEMP = 1.0
test_examples = [make_example() for _ in range(N_TEST)]


@torch.no_grad()
def greedy_decode(prefix_ids):
    seq = list(prefix_ids)
    maxprobs, entropies = [], []
    for _ in range(OUT_LEN):
        x = torch.tensor([seq], dtype=torch.long)
        logits = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum().item()
        mp = probs.max().item()
        nxt = int(probs.argmax().item())
        maxprobs.append(mp)
        entropies.append(ent)
        seq.append(nxt)
    answer = "".join(ITOS[i] for i in seq[IN_LEN:])
    return answer, float(np.mean(maxprobs)), float(np.mean(entropies))


@torch.no_grad()
def sample_decode(prefix_ids, temp=TEMP):
    seq = list(prefix_ids)
    for _ in range(OUT_LEN):
        x = torch.tensor([seq], dtype=torch.long)
        logits = model(x)
        probs = F.softmax(logits[0, -1] / temp, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        seq.append(nxt)
    return "".join(ITOS[i] for i in seq[IN_LEN:])


def edit_distance(a, b):
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def edit_cluster(samples, thresh=1):
    """Greedy clustering: assign each sample to first existing cluster rep
    within `thresh` Hamming distance (answers are fixed 3-char strings)."""
    reps = []  # list of [rep_string, members]
    for s in samples:
        placed = False
        for rep in reps:
            if edit_distance(s, rep[0]) <= thresh:
                rep[1].append(s)
                placed = True
                break
        if not placed:
            reps.append([s, [s]])
    return reps


rows = []
for text, a, b, s in test_examples:
    true_ans = f"{s:03d}"
    prefix = encode(text)[:IN_LEN]
    greedy_ans, maxprob, pred_entropy = greedy_decode(prefix)
    correct = int(greedy_ans == true_ans)
    samples = [sample_decode(prefix) for _ in range(K_SAMPLES)]

    # exact-match (original) signals
    exact_counts = Counter(samples)
    agree_exact = sum(1 for smp in samples if smp == greedy_ans) / K_SAMPLES
    p_exact = np.array(list(exact_counts.values())) / K_SAMPLES
    ent_exact = float(-(p_exact * np.log(p_exact)).sum())

    # edit-distance clustering (threshold 1): agreement = fraction of samples
    # in the same cluster as the greedy answer; entropy = Shannon entropy of
    # cluster-size distribution.
    clusters = edit_cluster(samples, thresh=1)
    greedy_cluster_size = 0
    for rep, members in clusters:
        if edit_distance(rep, greedy_ans) <= 1 or greedy_ans in members:
            greedy_cluster_size = max(greedy_cluster_size, len(members))
    agree_edit = greedy_cluster_size / K_SAMPLES
    sizes = np.array([len(m) for _, m in clusters]) / K_SAMPLES
    ent_edit = float(-(sizes * np.log(sizes)).sum())

    rows.append(dict(correct=correct, maxprob=maxprob, agree_exact=agree_exact,
                      ent_exact=ent_exact, agree_edit=agree_edit, ent_edit=ent_edit))

y = np.array([r["correct"] for r in rows])
maxprob = np.array([r["maxprob"] for r in rows])
agree_exact = np.array([r["agree_exact"] for r in rows])
ent_exact = np.array([r["ent_exact"] for r in rows])
agree_edit = np.array([r["agree_edit"] for r in rows])
ent_edit = np.array([r["ent_edit"] for r in rows])

acc = float(y.mean())


def auroc(score, label):
    return float(roc_auc_score(label, score)) if len(set(label.tolist())) > 1 else None


out = {
    "accuracy": acc,
    "n_test": N_TEST,
    "k_samples": K_SAMPLES,
    "auroc": {
        "maxprob": auroc(maxprob, y),
        "sample_agree_exact_match": auroc(agree_exact, y),
        "neg_sample_entropy_exact_match": auroc(-ent_exact, y),
        "sample_agree_editdist1_cluster": auroc(agree_edit, y),
        "neg_sample_entropy_editdist1_cluster": auroc(-ent_edit, y),
    },
    "elapsed_s": time.time() - t0,
}
print(json.dumps(out, indent=2))
with open("editdist_results.json", "w") as f:
    json.dump(out, f, indent=2)
