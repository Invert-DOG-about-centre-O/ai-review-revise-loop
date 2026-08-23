"""
Round-3 revision experiment, built on experiment_v3.py, addressing round-3
review weaknesses/questions:
  (1) [reviewer Q1] the noisy-clustering ablation used *uniform* random
      wrong-city relabeling; real NLI-clustering errors concentrate on
      semantically-close near-miss clusters. We add a *structured* noise
      model: with probability `noise_rate`, a sample is mis-clustered, and
      the wrong label is drawn mostly (80%) from a small fixed "neighbor"
      set of nearby city ids (a ring topology over city id, radius 2) and
      only occasionally (20%) uniform over all other cities -- a minimal
      stand-in for "NLI confuses similar answers more than dissimilar ones."
  (2) [reviewer Q2] the 8-aliases/city condition had only 3 seeds and a
      large per-seed spread (0.827-0.955); we add 2 more seeds (3,4) at
      n_aliases=8 to bring it to 5 seeds, matching the main comparison.
"""
import json, time
import random as pyrandom
from experiment_v3 import run_once_v3, N_CITY


def neighbor_set(city, radius=2):
    return {(city + d) % N_CITY for d in range(-radius, radius + 1) if d != 0}


def run_structured(seed, n_aliases, noise_rates, near_prob=0.8, radius=2):
    """Same as run_once_v3's noisy-clustering re-analysis, but with a
    structured (near-miss-biased) wrong-label distribution instead of a
    uniform one, applied to the *same* sampled outputs for a fair comparison."""
    # Reuse run_once_v3's internals by re-deriving raw samples the same way:
    # simplest reliable approach is to reimplement the small bit of logic
    # that differs (the mis-clustering label distribution) by monkey-patching
    # is fragile, so instead we call a light modified copy.
    import experiment_v3 as v3
    pyrandom.seed(seed)
    import torch, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

    city_aliases = []
    alias_to_city = {}
    for c in range(v3.N_CITY):
        aliases = set()
        while len(aliases) < n_aliases:
            L = pyrandom.randint(1, v3.MAX_ANSWER_LEN)
            alias = tuple(pyrandom.randint(0, v3.N_SUBTOK - 1) for _ in range(L))
            aliases.add(alias)
        aliases = list(aliases)
        city_aliases.append(aliases)
        for a in aliases:
            alias_to_city[a] = c

    country_valid_cities = {}
    country_ids = list(range(v3.N_COUNTRY))
    pyrandom.shuffle(country_ids)
    det_countries = country_ids[:v3.N_DET]
    amb_countries = country_ids[v3.N_DET:v3.N_DET + v3.N_AMB]
    for cid in det_countries:
        country_valid_cities[cid] = [pyrandom.randrange(v3.N_CITY)]
    for cid in amb_countries:
        k = pyrandom.choice([2, 3])
        country_valid_cities[cid] = pyrandom.sample(range(v3.N_CITY), k)

    def make_example(country_id):
        city = pyrandom.choice(country_valid_cities[country_id])
        alias = pyrandom.choice(city_aliases[city])
        seq = [v3.BOS, v3.Q_TOK, v3.COUNTRY_OFFSET + country_id, v3.SEP]
        seq += [v3.SUBTOK_OFFSET + t for t in alias]
        seq += [v3.EOS]
        return seq

    def pad_batch(seqs):
        import torch
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), v3.PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    trainable_countries = det_countries + amb_countries
    device = torch.device("cpu")
    model = v3.TinyLM(v3.VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    import torch.nn.functional as F
    loss = None
    for step in range(1, v3.STEPS + 1):
        batch_countries = [pyrandom.choice(trainable_countries) for _ in range(v3.BATCH)]
        seqs = [make_example(c) for c in batch_countries]
        x = pad_batch(seqs).to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, v3.VOCAB_SIZE), tgt.reshape(-1), ignore_index=v3.PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def sample_answer(country_id, temperature=1.0):
        seq = [v3.BOS, v3.Q_TOK, v3.COUNTRY_OFFSET + country_id, v3.SEP]
        for _ in range(v3.MAX_ANSWER_LEN + 1):
            xx = torch.tensor([seq], dtype=torch.long)
            logits = model(xx)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == v3.EOS:
                break
            seq.append(nxt)
        ans = tuple(t - v3.SUBTOK_OFFSET for t in seq[4:] if v3.SUBTOK_OFFSET <= t < v3.SUBTOK_OFFSET + v3.N_SUBTOK)
        return ans

    raw = {}
    for cid in det_countries + amb_countries:
        raw[cid] = [sample_answer(cid, v3.TEMPERATURE) for _ in range(v3.K)]

    def lex_entropy_for(samples):
        counts = {}
        for a in samples:
            counts[a] = counts.get(a, 0) + 1
        return v3.entropy(counts)

    rng = pyrandom.Random(seed * 1000 + 7)

    def sem_entropy_structured(samples, noise_rate):
        counts = {}
        for a in samples:
            true_city = alias_to_city.get(a, None)
            if true_city is None:
                key = "UNK"
            elif rng.random() < noise_rate:
                if rng.random() < near_prob:
                    nbrs = list(neighbor_set(true_city, radius))
                    key = rng.choice(nbrs)
                else:
                    wrong = rng.randrange(v3.N_CITY - 1)
                    if wrong >= true_city:
                        wrong += 1
                    key = wrong
            else:
                key = true_city
            counts[key] = counts.get(key, 0) + 1
        return v3.entropy(counts)

    det_amb = det_countries + amb_countries
    y_true = [1 if c in amb_countries else 0 for c in det_amb]
    from sklearn.metrics import roc_auc_score
    auroc_lex = roc_auc_score(y_true, [lex_entropy_for(raw[c]) for c in det_amb])
    structured_results = {}
    for nr in noise_rates:
        sem_ent = [sem_entropy_structured(raw[c], nr) for c in det_amb]
        structured_results[nr] = roc_auc_score(y_true, sem_ent)

    return dict(seed=seed, n_aliases=n_aliases, final_loss=loss.item(),
                auroc_lex=auroc_lex, structured_auroc_sem=structured_results)


if __name__ == "__main__":
    t0 = time.time()
    out = {"structured_noisy_clustering": [], "alias8_extra_seeds": []}
    noise_rates = (0.0, 0.05, 0.15, 0.30)

    # (1) structured (near-miss) clustering-noise ablation, 3 seeds, 2 aliases/city
    for seed in (0, 1, 2):
        r = run_structured(seed, n_aliases=2, noise_rates=noise_rates)
        print("structured", seed, r)
        out["structured_noisy_clustering"].append(r)

    # (2) extend 8-aliases/city ablation from 3 to 5 seeds
    for seed in (3, 4):
        r = run_once_v3(seed, n_aliases=8, noise_rates=(0.0,))
        r["auroc_sem_oracle"] = r["noisy_auroc_sem"][0.0]
        print("alias8 extra seed", seed, r)
        out["alias8_extra_seeds"].append(
            dict(seed=seed, n_aliases=8, final_loss=r["final_loss"],
                 auroc_lex=r["auroc_lex"], auroc_sem_oracle=r["auroc_sem_oracle"]))

    out["wall_time_sec"] = time.time() - t0
    with open("results_v4.json", "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", time.time() - t0)
