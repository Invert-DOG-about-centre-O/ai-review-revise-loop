"""
Cheap proxies vs. sampling-based semantic uncertainty for error detection
in a small autoregressive LM trained from scratch on synthetic arithmetic QA.

Fully offline (no pretrained-model download needed): trains a tiny
character-level Transformer decoder on "a+b=" -> "sum" strings, then
compares several uncertainty-quantification (UQ) methods at predicting
whether the model's greedy answer is wrong, and reports their compute cost.

Run: python experiment.py
"""
import math
import random
import time
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Vocab & data
# ---------------------------------------------------------------------------
CHARS = list("0123456789+= ") + ["<PAD>", "<EOS>"]
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
PAD = STOI["<PAD>"]
EOS = STOI["<EOS>"]

MAXLEN = 16  # "12+34=" (6) + answer up to 3 digits + EOS -> pad to 16


def make_example(a, b):
    prompt = f"{a}+{b}="
    answer = str(a + b)
    full = prompt + answer
    return prompt, answer, full


def encode(s):
    return [STOI[c] for c in s]


def build_dataset(n, lo=0, hi=40, seen=None):
    data = []
    seen = seen if seen is not None else set()
    while len(data) < n:
        a, b = random.randint(lo, hi), random.randint(lo, hi)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        data.append(make_example(a, b))
    return data, seen


N_TRAIN = 700
N_TEST = 200

seen = set()
train_data, seen = build_dataset(N_TRAIN, seen=seen)
test_data, seen = build_dataset(N_TEST, seen=seen)

print(f"train={len(train_data)} test={len(test_data)} (disjoint (a,b) pairs)")


def collate(batch):
    xs, masks = [], []
    for prompt, answer, full in batch:
        seq = encode(full) + [EOS]
        seq = seq[:MAXLEN]
        pad_len = MAXLEN - len(seq)
        mask = [0] * len(encode(prompt)) + [1] * (len(seq) - len(encode(prompt))) + [0] * pad_len
        seq = seq + [PAD] * pad_len
        xs.append(seq)
        masks.append(mask[:MAXLEN])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(masks, dtype=torch.float)


# ---------------------------------------------------------------------------
# Tiny GPT-style decoder
# ---------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=64, n_head=4, n_layer=3, maxlen=MAXLEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.maxlen = maxlen
        mask = torch.triu(torch.ones(maxlen, maxlen) * float("-inf"), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.blocks(h, mask=self.causal_mask[:t, :t])
        h = self.ln_f(h)
        return self.head(h)


model = TinyGPT(VOCAB)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

BATCH = 32
EPOCHS = 60

t0 = time.time()
for epoch in range(EPOCHS):
    random.shuffle(train_data)
    total_loss, nb = 0.0, 0
    for i in range(0, len(train_data), BATCH):
        batch = train_data[i:i + BATCH]
        x, mask = collate(batch)
        inp, tgt = x[:, :-1], x[:, 1:]
        m = mask[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none"
        )
        loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        nb += 1
    print(f"epoch {epoch} loss {total_loss / nb:.4f} elapsed {time.time() - t0:.1f}s")

print(f"training took {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_answer(prompt, temperature=0.8, max_new=4):
    """Autoregressively sample an answer continuation; return (answer_str, token_entropies)."""
    ids = encode(prompt)
    entropies = []
    for _ in range(max_new):
        x = torch.tensor([ids[-MAXLEN:]], dtype=torch.long)
        logits = model(x)[0, -1]
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        entropies.append(ent)
        if temperature == 0:
            nxt = int(torch.argmax(probs).item())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item())
        ids.append(nxt)
        if nxt == EOS:
            break
    gen = ids[len(encode(prompt)):]
    out_chars = []
    for tkn in gen:
        if tkn == EOS:
            break
        out_chars.append(ITOS[tkn])
    return "".join(out_chars), entropies


def parse_int(s):
    s = s.strip()
    if s == "" or not (s.lstrip("-").isdigit()):
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Evaluate UQ methods on the held-out test set
# ---------------------------------------------------------------------------
K = 8  # number of samples for sampling-based methods
TEMP = 0.9

records = []
t_greedy_total = 0.0
t_sampling_total = 0.0

for prompt, answer, full in test_data:
    true_val = int(answer)

    # --- cheap, single-pass method: greedy decode + mean token entropy ---
    t0 = time.time()
    greedy_ans, greedy_entropies = sample_answer(prompt, temperature=0.0, max_new=4)
    t_greedy_total += time.time() - t0
    greedy_val = parse_int(greedy_ans)
    correct = (greedy_val == true_val)
    mean_token_entropy = sum(greedy_entropies) / max(len(greedy_entropies), 1)
    max_token_entropy = max(greedy_entropies) if greedy_entropies else 0.0

    # --- expensive, sampling-based methods: K stochastic samples ---
    t0 = time.time()
    samples = [sample_answer(prompt, temperature=TEMP, max_new=4)[0] for _ in range(K)]
    t_sampling_total += time.time() - t0

    raw_strs = samples
    vals = [parse_int(s) for s in samples]

    # lexical diversity: fraction of distinct raw strings (syntactic, cheap-ish clustering)
    lexical_diversity = len(set(raw_strs)) / K

    # semantic clustering: group by parsed numeric value (None counts as own cluster)
    from collections import Counter
    cluster_counts = Counter(vals)
    cluster_probs = [c / K for c in cluster_counts.values()]
    semantic_entropy = -sum(p * math.log(p + 1e-12) for p in cluster_probs)

    majority_val, majority_count = cluster_counts.most_common(1)[0]
    self_consistency_disagreement = 1.0 - majority_count / K

    records.append(dict(
        prompt=prompt, true_val=true_val, greedy_val=greedy_val, correct=correct,
        mean_token_entropy=mean_token_entropy, max_token_entropy=max_token_entropy,
        lexical_diversity=lexical_diversity, semantic_entropy=semantic_entropy,
        self_consistency_disagreement=self_consistency_disagreement,
        majority_val=majority_val,
    ))

n_correct = sum(r["correct"] for r in records)
print(f"greedy accuracy: {n_correct}/{len(records)} = {n_correct/len(records):.3f}")
print(f"total time: greedy={t_greedy_total:.2f}s sampling(K={K})={t_sampling_total:.2f}s "
      f"(ratio {t_sampling_total/max(t_greedy_total,1e-9):.1f}x)")


def auroc(scores, labels_is_error):
    """labels_is_error: 1 if this is an error (positive class we want high score to predict)."""
    pairs = list(zip(scores, labels_is_error))
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    count = 0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


is_error = [0 if r["correct"] else 1 for r in records]

metrics = {
    "mean_token_entropy (cheap, 1 pass)": [r["mean_token_entropy"] for r in records],
    "max_token_entropy (cheap, 1 pass)": [r["max_token_entropy"] for r in records],
    "lexical_diversity (K samples, syntactic)": [r["lexical_diversity"] for r in records],
    "self_consistency_disagreement (K samples, semantic)": [r["self_consistency_disagreement"] for r in records],
    "semantic_entropy (K samples, semantic)": [r["semantic_entropy"] for r in records],
}

results = {}
for name, scores in metrics.items():
    results[name] = auroc(scores, is_error)

print("\nAUROC for predicting greedy-answer error:")
for name, val in results.items():
    print(f"  {name}: {val:.3f}")

# Also: does majority-vote (self-consistency) answer improve accuracy over greedy?
n_correct_majority = sum(1 for r in records if r["majority_val"] == r["true_val"])
print(f"\nmajority-vote (K={K}) accuracy: {n_correct_majority}/{len(records)} = {n_correct_majority/len(records):.3f}")

output = dict(
    n_train=len(train_data), n_test=len(test_data), n_params=n_params,
    epochs=EPOCHS, K=K, temperature=TEMP,
    greedy_accuracy=n_correct / len(records),
    majority_accuracy=n_correct_majority / len(records),
    time_greedy_s=t_greedy_total, time_sampling_s=t_sampling_total,
    cost_ratio=t_sampling_total / max(t_greedy_total, 1e-9),
    auroc=results,
)
with open(f"results_seed{SEED}.json", "w") as f:
    json.dump(output, f, indent=2)

with open(f"records_seed{SEED}.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"\nSaved results_seed{SEED}.json and records_seed{SEED}.json")
