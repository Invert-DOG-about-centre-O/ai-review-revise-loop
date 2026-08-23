"""
Revision experiment: multi-seed variance + OOD AUROC + alias-count ablation,
built directly on top of experiment.py's synthetic world/model, to address
round-1 review weaknesses:
  (1) single-seed AUROC/correlation numbers have no variance estimate
  (2) the "additive offset" explanation for the null result is a near-tautological
      consequence of exactly-2-aliases-per-city; test with more aliases
  (3) OOD claim (4d) lacks a quantitative separability metric like AUROC
"""
import json, math, random, time, sys
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
TEMPERATURE = 1.0
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


def run_once(seed, n_aliases=2):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    city_aliases = []
    alias_to_city = {}
    for c in range(N_CITY):
        aliases = set()
        while len(aliases) < n_aliases:
            L = random.randint(1, MAX_ANSWER_LEN)
            alias = tuple(random.randint(0, N_SUBTOK - 1) for _ in range(L))
            aliases.add(alias)
        aliases = list(aliases)
        city_aliases.append(aliases)
        for a in aliases:
            alias_to_city[a] = c

    country_valid_cities = {}
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
    device = torch.device("cpu")
    model = TinyLM(VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

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

    @torch.no_grad()
    def sample_answer(country_id, temperature=1.0):
        seq = [BOS, Q_TOK, COUNTRY_OFFSET + country_id, SEP]
        for _ in range(MAX_ANSWER_LEN + 1):
            xx = torch.tensor([seq], dtype=torch.long)
            logits = model(xx)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == EOS:
                break
            seq.append(nxt)
        ans = tuple(t - SUBTOK_OFFSET for t in seq[4:] if SUBTOK_OFFSET <= t < SUBTOK_OFFSET + N_SUBTOK)
        return ans

    def eval_group(country_list, group_name):
        rows = []
        for cid in country_list:
            samples = [sample_answer(cid, TEMPERATURE) for _ in range(K)]
            lex_counts, sem_counts = {}, {}
            lengths, unmapped = [], 0
            for a in samples:
                lex_counts[a] = lex_counts.get(a, 0) + 1
                lengths.append(len(a))
                city = alias_to_city.get(a, None)
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

    results = eval_group(det_countries, "det") + eval_group(amb_countries, "amb") + eval_group(ood_countries, "ood")

    det_amb = [r for r in results if r["group"] in ("det", "amb")]
    y_true = [1 if r["group"] == "amb" else 0 for r in det_amb]
    auroc_lex = roc_auc_score(y_true, [r["lex_entropy"] for r in det_amb])
    auroc_sem = roc_auc_score(y_true, [r["sem_entropy"] for r in det_amb])

    det_rows = [r for r in results if r["group"] == "det"]
    det_len = np.array([r["mean_len"] for r in det_rows])
    corr_len_lex = float(np.corrcoef(det_len, [r["lex_entropy"] for r in det_rows])[0, 1])
    corr_len_sem = float(np.corrcoef(det_len, [r["sem_entropy"] for r in det_rows])[0, 1])

    # OOD vs in-distribution (det+amb) quantitative separability
    id_ood = [r for r in results if r["group"] in ("det", "amb", "ood")]
    y_ood = [1 if r["group"] == "ood" else 0 for r in id_ood]
    auroc_ood_sem = roc_auc_score(y_ood, [r["sem_entropy"] for r in id_ood])
    auroc_ood_unmapped = roc_auc_score(y_ood, [r["unmapped_rate"] for r in id_ood])
    auroc_ood_lex = roc_auc_score(y_ood, [r["lex_entropy"] for r in id_ood])

    return dict(
        seed=seed, n_aliases=n_aliases, final_loss=loss.item(),
        auroc_lex=auroc_lex, auroc_sem=auroc_sem,
        corr_len_lex=corr_len_lex, corr_len_sem=corr_len_sem,
        auroc_ood_sem=auroc_ood_sem, auroc_ood_unmapped=auroc_ood_unmapped, auroc_ood_lex=auroc_ood_lex,
    )


if __name__ == "__main__":
    t0 = time.time()
    out = {"main_multiseed": [], "alias_ablation": []}

    # (1) multi-seed variance for main (n_aliases=2) config
    for seed in range(5):
        r = run_once(seed, n_aliases=2)
        print("seed", seed, r)
        out["main_multiseed"].append(r)

    # (2) alias-count ablation: does the lexical>=semantic result survive
    # more aliases per city (more surface diversity per meaning)?
    for n_al in (2, 4, 8):
        r = run_once(seed=0, n_aliases=n_al)
        print("n_aliases", n_al, r)
        out["alias_ablation"].append(r)

    out["wall_time_sec"] = time.time() - t0
    with open("results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", time.time() - t0)
