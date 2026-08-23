"""
Extended toy study: lexical vs. semantic predictive entropy as UQ signals,
now with (1) multi-seed variance estimates, (2) a quantitative OOD-vs-ID
separability metric, and (3) a class-correlated alias-diversity ablation that
directly stress-tests the "additive offset" explanation for the round-1 null
result (round1_review.json weaknesses 1, 2-ish, and 3).

Fully self-contained, CPU-only, no pretrained weights.
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


def build_world(seed, class_correlated_aliases=False, n_alias_amb=4, n_alias_det=2):
    """Builds the synthetic world. If class_correlated_aliases, cities that
    are the *sole/first* answer for an ambiguous country get MORE aliases
    than the alias count used by deterministic-only cities -- i.e. alias
    diversity is now correlated with ambiguity class, the direct opposite of
    the original design's independence assumption."""
    rng = random.Random(seed)

    city_aliases = []
    alias_to_city = {}

    country_valid_cities = {}
    country_ids = list(range(N_COUNTRY))
    rng.shuffle(country_ids)
    det_countries = country_ids[:N_DET]
    amb_countries = country_ids[N_DET:N_DET + N_AMB]
    ood_countries = country_ids[N_DET + N_AMB:]

    for cid in det_countries:
        country_valid_cities[cid] = [rng.randrange(N_CITY)]
    for cid in amb_countries:
        k = rng.choice([2, 3])
        country_valid_cities[cid] = rng.sample(range(N_CITY), k)

    if class_correlated_aliases:
        amb_cities = set()
        for cid in amb_countries:
            amb_cities.update(country_valid_cities[cid])
    else:
        amb_cities = set()

    for c in range(N_CITY):
        n_alias = n_alias_amb if (class_correlated_aliases and c in amb_cities) else n_alias_det
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
                ood_countries=ood_countries, trainable_countries=trainable_countries)


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
def sample_answer(model, rng_torch, country_id, temperature=1.0):
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
        samples = [sample_answer(model, None, cid, T) for _ in range(K)]
        lex_counts, sem_counts = {}, {}
        lengths, unmapped = [], 0
        for a in samples:
            lex_counts[a] = lex_counts.get(a, 0) + 1
            lengths.append(len(a))
            city = world["alias_to_city"].get(a, None)
            key = city if city is not None else "UNK"
            if city is None:
                unmapped += 1
            sem_counts[key] = sem_counts.get(key, 0) + 1
        rows.append(dict(
            country=cid, group=group_name,
            lex_entropy=entropy(lex_counts), sem_entropy=entropy(sem_counts),
            mean_len=float(np.mean(lengths)), unmapped_rate=unmapped / K,
        ))
    return rows


def run_one(seed, class_correlated_aliases=False, n_alias_amb=4, n_alias_det=2, verbose=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    world = build_world(seed, class_correlated_aliases, n_alias_amb, n_alias_det)
    model = TinyLM(VOCAB_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = random.Random(seed + 999)

    t0 = time.time()
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
    train_time = time.time() - t0

    det_rows = eval_group(model, world, world["det_countries"], "det", 1.0)
    amb_rows = eval_group(model, world, world["amb_countries"], "amb", 1.0)
    ood_rows = eval_group(model, world, world["ood_countries"], "ood", 1.0)

    det_amb = det_rows + amb_rows
    y = [1 if r["group"] == "amb" else 0 for r in det_amb]
    auroc_lex = roc_auc_score(y, [r["lex_entropy"] for r in det_amb])
    auroc_sem = roc_auc_score(y, [r["sem_entropy"] for r in det_amb])

    det_len = np.array([r["mean_len"] for r in det_rows])
    det_lex = np.array([r["lex_entropy"] for r in det_rows])
    det_sem = np.array([r["sem_entropy"] for r in det_rows])
    corr_len_lex = float(np.corrcoef(det_len, det_lex)[0, 1])
    corr_len_sem = float(np.corrcoef(det_len, det_sem)[0, 1])

    id_rows = det_rows + amb_rows
    ood_y = [0] * len(id_rows) + [1] * len(ood_rows)
    ood_sem_scores = [r["sem_entropy"] for r in id_rows] + [r["sem_entropy"] for r in ood_rows]
    ood_unmapped_scores = [r["unmapped_rate"] for r in id_rows] + [r["unmapped_rate"] for r in ood_rows]
    auroc_ood_sem = roc_auc_score(ood_y, ood_sem_scores)
    auroc_ood_unmapped = roc_auc_score(ood_y, ood_unmapped_scores)

    out = dict(seed=seed, train_time=train_time, final_loss=loss.item(),
               auroc_lex=auroc_lex, auroc_sem=auroc_sem,
               corr_len_lex=corr_len_lex, corr_len_sem=corr_len_sem,
               auroc_ood_sem_vs_id=auroc_ood_sem, auroc_ood_unmapped_vs_id=auroc_ood_unmapped)
    if verbose:
        print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Experiment A: multi-seed replication of the main comparison (5 seeds)
# ---------------------------------------------------------------------------
print("=== Experiment A: multi-seed main comparison (original 2-alias design) ===")
seeds = [0, 1, 2, 3, 4]
runs_A = [run_one(s, class_correlated_aliases=False) for s in seeds]
for r in runs_A:
    print(json.dumps(r))

auroc_lex_arr = np.array([r["auroc_lex"] for r in runs_A])
auroc_sem_arr = np.array([r["auroc_sem"] for r in runs_A])
corr_lex_arr = np.array([r["corr_len_lex"] for r in runs_A])
corr_sem_arr = np.array([r["corr_len_sem"] for r in runs_A])
ood_sem_arr = np.array([r["auroc_ood_sem_vs_id"] for r in runs_A])
ood_unm_arr = np.array([r["auroc_ood_unmapped_vs_id"] for r in runs_A])

summary_A = dict(
    seeds=seeds,
    auroc_lex_mean=float(auroc_lex_arr.mean()), auroc_lex_std=float(auroc_lex_arr.std()),
    auroc_sem_mean=float(auroc_sem_arr.mean()), auroc_sem_std=float(auroc_sem_arr.std()),
    n_seeds_lex_ge_sem=int((auroc_lex_arr >= auroc_sem_arr).sum()),
    corr_len_lex_mean=float(corr_lex_arr.mean()), corr_len_lex_std=float(corr_lex_arr.std()),
    corr_len_sem_mean=float(corr_sem_arr.mean()), corr_len_sem_std=float(corr_sem_arr.std()),
    auroc_ood_sem_mean=float(ood_sem_arr.mean()), auroc_ood_sem_std=float(ood_sem_arr.std()),
    auroc_ood_unmapped_mean=float(ood_unm_arr.mean()), auroc_ood_unmapped_std=float(ood_unm_arr.std()),
)
print("SUMMARY A:", json.dumps(summary_A, indent=2))

# ---------------------------------------------------------------------------
# Experiment B: class-correlated alias-diversity ablation (3 seeds; direct
# test of whether the "additive offset" mechanism breaks once alias
# diversity correlates with ambiguity class, per reviewer question 2)
# ---------------------------------------------------------------------------
print("=== Experiment B: class-correlated alias diversity (amb cities get 4 aliases, det cities get 2) ===")
seeds_B = [0, 1, 2]
runs_B = [run_one(s, class_correlated_aliases=True, n_alias_amb=4, n_alias_det=2) for s in seeds_B]
for r in runs_B:
    print(json.dumps(r))

auroc_lex_B = np.array([r["auroc_lex"] for r in runs_B])
auroc_sem_B = np.array([r["auroc_sem"] for r in runs_B])
summary_B = dict(
    seeds=seeds_B,
    auroc_lex_mean=float(auroc_lex_B.mean()), auroc_lex_std=float(auroc_lex_B.std()),
    auroc_sem_mean=float(auroc_sem_B.mean()), auroc_sem_std=float(auroc_sem_B.std()),
    n_seeds_lex_ge_sem=int((auroc_lex_B >= auroc_sem_B).sum()),
)
print("SUMMARY B:", json.dumps(summary_B, indent=2))

with open("results_v2.json", "w") as f:
    json.dump(dict(experiment_A_runs=runs_A, experiment_A_summary=summary_A,
                    experiment_B_runs=runs_B, experiment_B_summary=summary_B), f, indent=2)

print("DONE")
