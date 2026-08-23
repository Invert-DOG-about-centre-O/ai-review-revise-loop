"""
Toy controlled study of lexical vs. semantic predictive entropy as
uncertainty/hallucination signals for a small autoregressive transformer LM,
including a probe for the length-confound reported in real-LLM UQ evaluations.

Fully self-contained (no internet, no pretrained weights): we build a tiny
synthetic "capital-city" QA world with a KNOWN ground-truth ambiguity
structure, train a small causal Transformer on it, then sample from the
trained model and analyze uncertainty metrics against that ground truth.
"""
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Synthetic world
# ---------------------------------------------------------------------------
N_SUBTOK = 24          # subtoken vocabulary used to build city name strings
N_CITY = 15            # number of distinct semantic "city" entities
N_DET = 25             # deterministic countries (exactly one true city)
N_AMB = 25             # ambiguous countries (2-3 true cities, genuine aleatoric uncertainty)
N_OOD = 15             # out-of-distribution countries, never seen during training

PAD, BOS, EOS, Q_TOK, SEP = 0, 1, 2, 3, 4
SUBTOK_OFFSET = 5
COUNTRY_OFFSET = SUBTOK_OFFSET + N_SUBTOK
N_COUNTRY = N_DET + N_AMB + N_OOD
VOCAB_SIZE = COUNTRY_OFFSET + N_COUNTRY

MAX_ANSWER_LEN = 3
MAX_SEQ_LEN = 4 + MAX_ANSWER_LEN + 1  # BOS Q country SEP <answer...> EOS

# Each city has 2 surface "aliases" (different subtoken strings, same meaning).
# Alias lengths vary 1..MAX_ANSWER_LEN tokens -> gives us controlled length variation
# for a fixed semantic identity.
city_aliases = []  # city_aliases[c] = list of alias tuples (subtoken ids)
alias_to_city = {}
for c in range(N_CITY):
    aliases = set()
    while len(aliases) < 2:
        L = random.randint(1, MAX_ANSWER_LEN)
        alias = tuple(random.randint(0, N_SUBTOK - 1) for _ in range(L))
        aliases.add(alias)
    aliases = list(aliases)
    city_aliases.append(aliases)
    for a in aliases:
        alias_to_city[a] = c

country_valid_cities = {}  # country_id -> list of valid city ids
country_ids = list(range(N_COUNTRY))
random.shuffle(country_ids)
det_countries = country_ids[:N_DET]
amb_countries = country_ids[N_DET:N_DET + N_AMB]
ood_countries = country_ids[N_DET + N_AMB:]

for cid in det_countries:
    country_valid_cities[cid] = [random.randrange(N_CITY)]
for cid in amb_countries:
    k = random.choice([2, 3])
    country_valid_cities[cid] = random.sample(range(N_CITY), k)
# ood countries: intentionally NOT added to country_valid_cities / never trained on

def make_example(country_id):
    city = random.choice(country_valid_cities[country_id])
    alias = random.choice(city_aliases[city])
    seq = [BOS, Q_TOK, COUNTRY_OFFSET + country_id, SEP]
    seq += [SUBTOK_OFFSET + t for t in alias]
    seq += [EOS]
    return seq

def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

trainable_countries = det_countries + amb_countries

# ---------------------------------------------------------------------------
# 2. Tiny causal Transformer LM
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, nlayers=2, dff=128, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dff, dropout=0.1,
                                            batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1).to(x.device)
        pad_mask = (x == PAD)
        h = self.enc(h, mask=mask, src_key_padding_mask=pad_mask)
        return self.out(h)

device = torch.device("cpu")
model = TinyLM(VOCAB_SIZE).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
BATCH = 64
STEPS = 1200
log_lines = []
t0 = time.time()
for step in range(1, STEPS + 1):
    batch_countries = [random.choice(trainable_countries) for _ in range(BATCH)]
    seqs = [make_example(c) for c in batch_countries]
    x = pad_batch(seqs).to(device)
    inp, tgt = x[:, :-1], x[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=PAD)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0 or step == 1:
        line = f"step {step:5d}  loss {loss.item():.4f}  elapsed {time.time()-t0:.1f}s"
        print(line)
        log_lines.append(line)

train_time = time.time() - t0
log_lines.append(f"TOTAL TRAIN TIME: {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 4. Sampling-based evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_answer(country_id, temperature=1.0):
    seq = [BOS, Q_TOK, COUNTRY_OFFSET + country_id, SEP]
    for _ in range(MAX_ANSWER_LEN + 1):
        x = torch.tensor([seq], dtype=torch.long)
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == EOS:
            break
        seq.append(nxt)
    ans = tuple(t - SUBTOK_OFFSET for t in seq[4:] if SUBTOK_OFFSET <= t < SUBTOK_OFFSET + N_SUBTOK)
    return ans

def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ps = [c / total for c in counts.values()]
    return -sum(p * math.log(p + 1e-12) for p in ps)

K = 40
TEMPERATURE = 1.0

def eval_group(country_list, group_name):
    rows = []
    for cid in country_list:
        samples = [sample_answer(cid, TEMPERATURE) for _ in range(K)]
        lex_counts, sem_counts = {}, {}
        lengths = []
        unmapped = 0
        for a in samples:
            lex_counts[a] = lex_counts.get(a, 0) + 1
            lengths.append(len(a))
            city = alias_to_city.get(a, None)
            key = city if city is not None else f"UNK"
            if city is None:
                unmapped += 1
            sem_counts[key] = sem_counts.get(key, 0) + 1
        lex_H = entropy(lex_counts)
        sem_H = entropy(sem_counts)
        valid_cities = set(country_valid_cities.get(cid, []))
        mapped_cities = [alias_to_city[a] for a in samples if a in alias_to_city]
        if group_name == "ood":
            correct_rate = float("nan")
        else:
            correct_rate = (sum(1 for c in mapped_cities if c in valid_cities) / K)
        rows.append(dict(
            country=cid, group=group_name,
            lex_entropy=lex_H, sem_entropy=sem_H,
            mean_len=float(np.mean(lengths)),
            unmapped_rate=unmapped / K,
            n_true_answers=len(valid_cities) if group_name != "ood" else 0,
            correct_rate=correct_rate,
        ))
    return rows

results = []
results += eval_group(det_countries, "det")
results += eval_group(amb_countries, "amb")
results += eval_group(ood_countries, "ood")

# ---------------------------------------------------------------------------
# 5. Analysis
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score

det_amb = [r for r in results if r["group"] in ("det", "amb")]
y_true = [1 if r["group"] == "amb" else 0 for r in det_amb]
lex_scores = [r["lex_entropy"] for r in det_amb]
sem_scores = [r["sem_entropy"] for r in det_amb]
auroc_lex = roc_auc_score(y_true, lex_scores)
auroc_sem = roc_auc_score(y_true, sem_scores)

# Length-confound probe: within DETERMINISTIC questions only, true ambiguity
# is constant (=0), so any correlation between entropy and mean answer length
# is a pure artifact/confound, not signal.
det_rows = [r for r in results if r["group"] == "det"]
det_len = np.array([r["mean_len"] for r in det_rows])
det_lex = np.array([r["lex_entropy"] for r in det_rows])
det_sem = np.array([r["sem_entropy"] for r in det_rows])
corr_len_lex = float(np.corrcoef(det_len, det_lex)[0, 1])
corr_len_sem = float(np.corrcoef(det_len, det_sem)[0, 1])

def group_stats(name):
    rows = [r for r in results if r["group"] == name]
    return dict(
        n=len(rows),
        mean_lex_entropy=float(np.mean([r["lex_entropy"] for r in rows])),
        mean_sem_entropy=float(np.mean([r["sem_entropy"] for r in rows])),
        mean_unmapped_rate=float(np.mean([r["unmapped_rate"] for r in rows])),
        mean_correct_rate=float(np.nanmean([r["correct_rate"] for r in rows])),
    )

summary = dict(
    train_steps=STEPS,
    train_time_sec=train_time,
    final_loss=loss.item(),
    K_samples=K,
    temperature=TEMPERATURE,
    auroc_lexical_entropy_vs_ambiguous=auroc_lex,
    auroc_semantic_entropy_vs_ambiguous=auroc_sem,
    length_confound_corr_lexical=corr_len_lex,
    length_confound_corr_semantic=corr_len_sem,
    group_stats={g: group_stats(g) for g in ("det", "amb", "ood")},
)

print(json.dumps(summary, indent=2))

# ---------------------------------------------------------------------------
# 6. Temperature ablation: does the lexical-vs-semantic AUROC gap depend on
#    sampling temperature?
# ---------------------------------------------------------------------------
def eval_group_T(country_list, group_name, T):
    rows = []
    for cid in country_list:
        samples = [sample_answer(cid, T) for _ in range(K)]
        lex_counts, sem_counts = {}, {}
        for a in samples:
            lex_counts[a] = lex_counts.get(a, 0) + 1
            city = alias_to_city.get(a, None)
            key = city if city is not None else "UNK"
            sem_counts[key] = sem_counts.get(key, 0) + 1
        rows.append(dict(country=cid, group=group_name,
                          lex_entropy=entropy(lex_counts), sem_entropy=entropy(sem_counts)))
    return rows

temp_ablation = []
for T in (0.6, 1.0, 1.5):
    rows_t = eval_group_T(det_countries, "det", T) + eval_group_T(amb_countries, "amb", T)
    y_t = [1 if r["group"] == "amb" else 0 for r in rows_t]
    auroc_lex_t = roc_auc_score(y_t, [r["lex_entropy"] for r in rows_t])
    auroc_sem_t = roc_auc_score(y_t, [r["sem_entropy"] for r in rows_t])
    temp_ablation.append(dict(temperature=T, auroc_lexical=auroc_lex_t, auroc_semantic=auroc_sem_t))

summary["temperature_ablation"] = temp_ablation
print(json.dumps(temp_ablation, indent=2))

with open("results.json", "w") as f:
    json.dump(dict(summary=summary, rows=results), f, indent=2)

with open("train_log.txt", "w") as f:
    f.write("\n".join(log_lines) + "\n\n" + json.dumps(summary, indent=2) + "\n")

print("DONE")
