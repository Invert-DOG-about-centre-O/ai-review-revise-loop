"""
Round-2 revision experiment, built on experiment_v2.py, addressing round-2
review weaknesses:
  (1) alias-count ablation was single-seed -> now 3 seeds/condition (0,1,2)
  (2) oracle semantic clustering is a best case for "semantic entropy" ->
      add a noisy-clustering ablation that injects clustering errors at
      controlled rates (0%, 5%, 15%, 30%) into the oracle map and re-measures
      AUROC, to see how fast the oracle-semantic advantage erodes toward /
      past the lexical baseline once clustering is imperfect.
"""
import json, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from experiment_v2 import (
    TinyLM, entropy, N_SUBTOK, N_CITY, N_DET, N_AMB, N_OOD, N_COUNTRY,
    VOCAB_SIZE, PAD, BOS, EOS, Q_TOK, SEP, SUBTOK_OFFSET, COUNTRY_OFFSET,
    MAX_ANSWER_LEN, K, TEMPERATURE, STEPS, BATCH,
)


def run_once_v3(seed, n_aliases=2, noise_rates=(0.0,)):
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

    # collect raw samples once per country so noisy-clustering re-analysis is free
    raw = {}
    for cid in det_countries + amb_countries:
        raw[cid] = [sample_answer(cid, TEMPERATURE) for _ in range(K)]

    def lex_entropy_for(samples):
        counts = {}
        for a in samples:
            counts[a] = counts.get(a, 0) + 1
        return entropy(counts)

    def sem_entropy_for(samples, noise_rate, rng):
        counts = {}
        for a in samples:
            true_city = alias_to_city.get(a, None)
            if true_city is None:
                key = "UNK"
            elif rng.random() < noise_rate:
                # simulate an imperfect NLI/judge clustering error: mis-cluster
                # into a uniformly random *wrong* city id instead of the true one
                wrong = rng.randrange(N_CITY - 1)
                if wrong >= true_city:
                    wrong += 1
                key = wrong
            else:
                key = true_city
            counts[key] = counts.get(key, 0) + 1
        return entropy(counts)

    det_amb = det_countries + amb_countries
    y_true = [1 if c in amb_countries else 0 for c in det_amb]
    auroc_lex = roc_auc_score(y_true, [lex_entropy_for(raw[c]) for c in det_amb])

    noisy_results = {}
    rng = random.Random(seed * 1000 + 7)
    for nr in noise_rates:
        sem_ent = [sem_entropy_for(raw[c], nr, rng) for c in det_amb]
        noisy_results[nr] = roc_auc_score(y_true, sem_ent)

    return dict(seed=seed, n_aliases=n_aliases, final_loss=loss.item(),
                auroc_lex=auroc_lex, noisy_auroc_sem=noisy_results)


if __name__ == "__main__":
    t0 = time.time()
    out = {"alias_ablation_multiseed": [], "noisy_clustering": []}
    noise_rates = (0.0, 0.05, 0.15, 0.30)

    # (1) alias-count ablation, now over 3 seeds/condition instead of 1.
    # For n_aliases=2 we also sweep clustering-noise rates in the same run
    # (free re-analysis of the same sampled outputs) to address (2) below.
    for n_al in (2, 4, 8):
        for seed in (0, 1, 2):
            nr = noise_rates if n_al == 2 else (0.0,)
            r = run_once_v3(seed, n_aliases=n_al, noise_rates=nr)
            r["auroc_sem_oracle"] = r["noisy_auroc_sem"][0.0]
            print("alias", n_al, "seed", seed, r)
            out["alias_ablation_multiseed"].append(
                dict(seed=seed, n_aliases=n_al, final_loss=r["final_loss"],
                     auroc_lex=r["auroc_lex"], auroc_sem_oracle=r["auroc_sem_oracle"]))
            if n_al == 2:
                out["noisy_clustering"].append(
                    dict(seed=seed, auroc_lex=r["auroc_lex"], noisy_auroc_sem=r["noisy_auroc_sem"]))

    out["wall_time_sec"] = time.time() - t0
    with open("results_v3.json", "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", time.time() - t0)
