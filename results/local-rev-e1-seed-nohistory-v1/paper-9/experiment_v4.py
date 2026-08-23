"""
Round-3 review follow-up on top of experiment_v3.py:

Reviewer weakness 2 / question 2: the class-exclusive ablation (Experiment C)
changes TWO things relative to the main 15-city design at once -- (i) alias
count becomes correlated with ambiguity class (4 amb-pool aliases/city vs 2
det-pool aliases/city) AND (ii) the city pool is restructured into two
smaller disjoint pools (7 amb-only + 8 det-only) instead of one shared
15-city pool. Experiment C's AUROC could shift because of either change, but
the paper attributed the (small) shift entirely to (i).

Experiment E here is the matched control the reviewer asked for: the SAME
disjoint 7/8 pool structure as Experiment C, but with alias counts
EQUALIZED across the two pools (3 aliases/city each, instead of 4 vs 2), so
alias count is no longer correlated with ambiguity class. This isolates (ii)
-- pool restructuring alone -- so we can tell how much of Experiment C's
shift versus the main result (a) is attributable to alias-count correlation
specifically, versus just having two smaller disjoint pools.

Fully self-contained, CPU-only, no pretrained weights. Duplicates the
model/training/eval logic from experiment_v3.py to keep this file standalone.
"""
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

N_SUBTOK = 24
N_CITY = 15
N_DET = 25
N_AMB = 25
N_OOD = 15

PAD, BOS, EOS, Q_TOK, SEP = 0, 1, 2, 3, 4
SUBTOK_OFFSET = 5
COUNTRY_OFFSET = SUBTOK_OFFSET + N_SUBTOK
N_COUNTRY = N_DET + N_AMB + N_OOD
VOCAB_SIZE = COUNTRY_OFFSET + N_COUNTRY
MAX_ANSWER_LEN = 3
MAX_SEQ_LEN = 4 + MAX_ANSWER_LEN + 1
K = 40
STEPS = 1200
BATCH = 64

AMB_POOL_SIZE = 7
DET_POOL_SIZE = N_CITY - AMB_POOL_SIZE


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


def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ps = [c / total for c in counts.values()]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def build_world_exclusive(seed, n_alias_amb, n_alias_det):
    """Disjoint amb-only / det-only city pools. n_alias_amb/n_alias_det control
    whether alias count is correlated with class (4 vs 2, Experiment C) or
    equalized (3 vs 3, Experiment E matched control)."""
    rng = random.Random(seed)

    city_ids = list(range(N_CITY))
    rng.shuffle(city_ids)
    amb_pool = city_ids[:AMB_POOL_SIZE]
    det_pool = city_ids[AMB_POOL_SIZE:]

    country_valid_cities = {}
    country_ids = list(range(N_COUNTRY))
    rng.shuffle(country_ids)
    det_countries = country_ids[:N_DET]
    amb_countries = country_ids[N_DET:N_DET + N_AMB]
    ood_countries = country_ids[N_DET + N_AMB:]

    for cid in det_countries:
        country_valid_cities[cid] = [rng.choice(det_pool)]
    for cid in amb_countries:
        k = rng.choice([2, 3])
        k = min(k, len(amb_pool))
        country_valid_cities[cid] = rng.sample(amb_pool, k)

    city_aliases = []
    alias_to_city = {}
    for c in range(N_CITY):
        n_alias = n_alias_amb if c in amb_pool else n_alias_det
        aliases = set()
        while len(aliases) < n_alias:
            L = rng.randint(1, MAX_ANSWER_LEN)
            alias = tuple(rng.randint(0, N_SUBTOK - 1) for _ in range(L))
            aliases.add(alias)
        aliases = list(aliases)
        city_aliases.append(aliases)
        for a in aliases:
            alias_to_city[a] = c

    trainable_countries = det_countries + amb_countries
    return dict(city_aliases=city_aliases, alias_to_city=alias_to_city,
                country_valid_cities=country_valid_cities,
                det_countries=det_countries, amb_countries=amb_countries,
                ood_countries=ood_countries, trainable_countries=trainable_countries,
                amb_pool=amb_pool, det_pool=det_pool)


def make_example(rng, world, country_id):
    city = rng.choice(world["country_valid_cities"][country_id])
    alias = rng.choice(world["city_aliases"][city])
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


@torch.no_grad()
def sample_answer(model, country_id, temperature=1.0):
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


def eval_group(model, world, country_list, group_name, T):
    rows = []
    for cid in country_list:
        samples = [sample_answer(model, cid, T) for _ in range(K)]
        lex_counts, sem_counts = {}, {}
        for a in samples:
            lex_counts[a] = lex_counts.get(a, 0) + 1
            city = world["alias_to_city"].get(a, None)
            key = city if city is not None else "UNK"
            sem_counts[key] = sem_counts.get(key, 0) + 1
        rows.append(dict(
            country=cid, group=group_name,
            lex_entropy=entropy(lex_counts), sem_entropy=entropy(sem_counts),
        ))
    return rows


def train_model(seed, world):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyLM(VOCAB_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = random.Random(seed + 999)
    for step in range(1, STEPS + 1):
        batch_countries = [rng.choice(world["trainable_countries"]) for _ in range(BATCH)]
        seqs = [make_example(rng, world, c) for c in batch_countries]
        x = pad_batch(seqs)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def run_ablation(name, seeds, n_alias_amb, n_alias_det):
    runs = []
    for s in seeds:
        world = build_world_exclusive(s, n_alias_amb=n_alias_amb, n_alias_det=n_alias_det)
        model, final_loss = train_model(s, world)
        det_rows = eval_group(model, world, world["det_countries"], "det", 1.0)
        amb_rows = eval_group(model, world, world["amb_countries"], "amb", 1.0)
        det_amb = det_rows + amb_rows
        y = [1 if r["group"] == "amb" else 0 for r in det_amb]
        auroc_lex = roc_auc_score(y, [r["lex_entropy"] for r in det_amb])
        auroc_sem = roc_auc_score(y, [r["sem_entropy"] for r in det_amb])
        row = dict(seed=s, final_loss=final_loss, auroc_lex=auroc_lex, auroc_sem=auroc_sem)
        print(name, json.dumps(row))
        runs.append(row)
    auroc_lex = np.array([r["auroc_lex"] for r in runs])
    auroc_sem = np.array([r["auroc_sem"] for r in runs])
    summary = dict(
        seeds=seeds,
        auroc_lex_mean=float(auroc_lex.mean()), auroc_lex_std=float(auroc_lex.std()),
        auroc_sem_mean=float(auroc_sem.mean()), auroc_sem_std=float(auroc_sem.std()),
        n_seeds_lex_ge_sem=int((auroc_lex >= auroc_sem).sum()),
        gap_per_seed=[float(l - s_) for l, s_ in zip(auroc_lex, auroc_sem)],
    )
    print(f"SUMMARY {name}:", json.dumps(summary, indent=2))
    return runs, summary


print("=== Experiment E: matched control -- disjoint 7/8 pools, EQUALIZED alias count (3/3) ===")
seeds_E = [0, 1, 2, 3, 4]
runs_E, summary_E = run_ablation("E", seeds_E, n_alias_amb=3, n_alias_det=3)

print("=== Re-running Experiment C (class-correlated alias count, 4/2) on the SAME 5 seeds for a fair paired comparison ===")
runs_C5, summary_C5 = run_ablation("C5", seeds_E, n_alias_amb=4, n_alias_det=2)

with open("results_v4.json", "w") as f:
    json.dump(dict(experiment_E_runs=runs_E, experiment_E_summary=summary_E,
                    experiment_C5_runs=runs_C5, experiment_C5_summary=summary_C5), f, indent=2)

print("DONE")
