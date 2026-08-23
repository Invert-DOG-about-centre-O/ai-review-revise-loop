"""
Follow-up experiments addressing round-1 review questions:
 (1) determinism / run-to-run variance beyond seed-to-seed variance
 (2) sensitivity of the ablation semantic/lexical/token AUROC gap to the
     underfit-epoch choice (10/12/14/16/18)
 (3) robustness of the semantic-vs-lexical margin to a noisy ("imperfect
     entailment classifier") entity extractor instead of the oracle one
"""
import json
import math
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

torch.set_num_threads(1)

from experiment import run_seed  # noqa: E402

t0 = time.time()
out = {}

# ---------------------------------------------------------------------
# (1) determinism check: same seed, same code, single-threaded -> run twice
# ---------------------------------------------------------------------
det = []
for trial in range(2):
    r = run_seed(100, epochs=14)
    seen = [rec for rec in r["records"] if rec["split"] == "seen"]
    y = np.array([0 if rec["correct"] else 1 for rec in seen])
    aurocs = {}
    if 0 < y.sum() < len(y):
        for m in ["pred_entropy", "lex_entropy", "sem_entropy"]:
            aurocs[m] = float(roc_auc_score(y, [rec[m] for rec in seen]))
    det.append({"trial": trial, "acc_seen": r["acc_seen"], "auroc": aurocs})
out["determinism_same_seed_single_thread"] = det
print(f"[{time.time()-t0:.1f}s] determinism check done:", json.dumps(det))

# ---------------------------------------------------------------------
# (2) epoch sensitivity sweep for the ablation gap (3 seeds per epoch count
#     to keep within budget; seeds distinct from the paper's 100-104)
# ---------------------------------------------------------------------
sweep = {}
for epochs in [10, 12, 14, 16, 18]:
    runs = [run_seed(seed, epochs=epochs) for seed in range(300, 303)]
    seen = [rec for r in runs for rec in r["records"] if rec["split"] == "seen"]
    y = np.array([0 if rec["correct"] else 1 for rec in seen])
    acc_seen_mean = float(np.mean([r["acc_seen"] for r in runs]))
    entry = {"acc_seen_mean": acc_seen_mean, "n_seen": len(seen), "n_wrong": int(y.sum())}
    if 0 < y.sum() < len(y):
        for m in ["pred_entropy", "lex_entropy", "sem_entropy"]:
            entry[m] = float(roc_auc_score(y, [rec[m] for rec in seen]))
    else:
        entry["note"] = "degenerate (all correct or all wrong)"
    sweep[epochs] = entry
    print(f"[{time.time()-t0:.1f}s] epochs={epochs} acc_seen={acc_seen_mean:.3f} n_wrong={int(y.sum())}/{len(seen)} "
          f"{ {k: round(v,3) for k,v in entry.items() if k in ('pred_entropy','lex_entropy','sem_entropy')} }")
out["epoch_sensitivity_sweep"] = sweep

# ---------------------------------------------------------------------
# (3) noisy entity extractor: simulate an imperfect entailment/similarity
# classifier. run_seed() does not persist the K per-sample extracted
# entities, only the aggregated cluster entropy, so we reimplement just the
# K-sample generation + noisy-clustering step here (reusing run_seed's
# internals is not possible without refactoring it, and we don't have
# import-time hooks) by re-running the exact same training/sampling code
# path with an injected corruption step on the extracted entity labels
# before entropy is computed. To keep this within budget we patch
# experiment.py's extract_city indirectly: we call run_seed_with_noise,
# a thin copy of run_seed with one extra corruption line.
# ---------------------------------------------------------------------
import experiment as exp

def run_seed_with_noise(seed, epochs, noise_p, rng_noise):
    """Same as experiment.run_seed but corrupts each sample's extracted
    entity with probability noise_p before computing semantic entropy
    (simulating an imperfect entailment/similarity classifier)."""
    import random as _random
    r = _random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    countries = [f"country{i}" for i in range(exp.N_FACTS)]
    cities = [f"city{i}" for i in range(exp.N_FACTS)]
    shuffled_cities = cities[:]
    r.shuffle(shuffled_cities)
    kb = dict(zip(countries, shuffled_cities))
    n_held = int(exp.N_FACTS * exp.HELD_OUT_FRAC)
    shuffled_countries = countries[:]
    r.shuffle(shuffled_countries)
    held_out = set(shuffled_countries[:n_held])
    train_countries = [c for c in countries if c not in held_out]

    specials = ["<pad>", "<bos>", "<eos>", "<sep>"]
    vocab_words = set()
    for t in exp.q_templates:
        vocab_words.update(t.replace("{c}", "").split())
    for t in exp.a_templates:
        vocab_words.update(t.replace("{a}", "").split())
    vocab_words.update(countries)
    vocab_words.update(cities)
    vocab = specials + sorted(vocab_words)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    PAD, BOS, EOS, SEP = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"], stoi["<sep>"]
    city_token_ids = set(stoi[c] for c in cities)

    train_seqs = []
    for country in train_countries:
        city = kb[country]
        for qt in exp.q_templates:
            q_toks = [stoi[w] for w in qt.replace("{c}", country).split()]
            for at in exp.a_templates:
                a_toks = [stoi[w] for w in at.replace("{a}", city).split()]
                train_seqs.append([BOS] + q_toks + [SEP] + a_toks + [EOS])
    r.shuffle(train_seqs)
    max_len = max(len(s) for s in train_seqs) + 8

    def pad_batch(seqs):
        out = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s)
        return out

    model = exp.TinyLM(len(vocab), PAD, max_len)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n_batches = math.ceil(len(train_seqs) / exp.BATCH)
    model.train()
    for epoch in range(epochs):
        r.shuffle(train_seqs)
        for b in range(n_batches):
            chunk = train_seqs[b * exp.BATCH:(b + 1) * exp.BATCH]
            x = pad_batch(chunk)
            pad_mask = x == PAD
            logits = model(x[:, :-1], pad_mask[:, :-1])
            targets = x[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    @torch.no_grad()
    def sample_completion(q_ids):
        seq = [BOS] + q_ids + [SEP]
        for _ in range(exp.MAX_GEN):
            x = torch.tensor(seq).unsqueeze(0)
            pad_mask = torch.zeros_like(x, dtype=torch.bool)
            logits = model(x, pad_mask)[0, -1] / exp.TEMP
            probs = torch.nn.functional.softmax(logits, dim=-1)
            nxt = torch.distributions.Categorical(probs).sample().item()
            seq.append(nxt)
            if nxt == EOS:
                break
        answer_ids = seq[seq.index(SEP) + 1:]
        if answer_ids and answer_ids[-1] == EOS:
            answer_ids = answer_ids[:-1]
        return answer_ids

    def extract_city(answer_ids):
        for tid in answer_ids:
            if tid in city_token_ids:
                return itos[tid]
        return None

    test_seen = r.sample(train_countries, min(20, len(train_countries)))
    records = []
    for country in test_seen:
        qt = r.choice(exp.q_templates)
        q_ids = [stoi[w] for w in qt.replace("{c}", country).split()]
        samples = [sample_completion(q_ids) for _ in range(exp.K)]
        extracted_true = [extract_city(a) for a in samples]

        # noisy version: with prob noise_p, relabel a sample's extracted
        # entity to a uniformly random *other* city (simulating a wrong
        # entailment/similarity clustering decision)
        extracted_noisy = []
        for c in extracted_true:
            if rng_noise.rand() < noise_p:
                others = [x for x in cities if x != c]
                extracted_noisy.append(others[rng_noise.randint(len(others))])
            else:
                extracted_noisy.append(c)

        sem_counts = {}
        for c in extracted_noisy:
            sem_counts[c] = sem_counts.get(c, 0) + 1
        sem_probs = np.array(list(sem_counts.values())) / exp.K
        sem_entropy_noisy = float(-(sem_probs * np.log(sem_probs + 1e-12)).sum())

        # correctness uses the TRUE (oracle) majority vote, unaffected by the
        # noisy clustering, since noise models an imperfect UQ signal, not an
        # imperfect answer
        true_counts = {}
        for c in extracted_true:
            true_counts[c] = true_counts.get(c, 0) + 1
        maj_city = max(true_counts.items(), key=lambda kv: kv[1])[0]
        correct = (maj_city == kb[country])

        records.append(dict(correct=bool(correct), sem_entropy_noisy=sem_entropy_noisy))
    return records


ablation_runs = [run_seed(seed, epochs=14) for seed in range(100, 105)]
seen_records = [rec for r in ablation_runs for rec in r["records"] if rec["split"] == "seen"]
y_wrong = np.array([0 if rec["correct"] else 1 for rec in seen_records])
lex_auroc = float(roc_auc_score(y_wrong, [rec["lex_entropy"] for rec in seen_records]))
sem_auroc_oracle = float(roc_auc_score(y_wrong, [rec["sem_entropy"] for rec in seen_records]))

rng_noise = np.random.RandomState(42)
noise_results = {"lexical_auroc_reference": lex_auroc, "semantic_oracle_auroc_reference": sem_auroc_oracle}
for p in [0.0, 0.1, 0.2, 0.3, 0.5]:
    noisy_records = []
    for seed in range(400, 403):
        noisy_records += run_seed_with_noise(seed, epochs=14, noise_p=p, rng_noise=rng_noise)
    y = np.array([0 if rec["correct"] else 1 for rec in noisy_records])
    if 0 < y.sum() < len(y):
        auroc = float(roc_auc_score(y, [rec["sem_entropy_noisy"] for rec in noisy_records]))
    else:
        auroc = float("nan")
    noise_results[p] = {"semantic_auroc": auroc, "n": len(noisy_records), "n_wrong": int(y.sum())}
    print(f"[{time.time()-t0:.1f}s] noise_p={p} semantic_auroc={auroc:.3f} n_wrong={int(y.sum())}/{len(noisy_records)}")

out["noisy_extractor_sweep"] = noise_results

with open("followup_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"[{time.time()-t0:.1f}s] DONE, total {time.time()-t0:.1f}s")
