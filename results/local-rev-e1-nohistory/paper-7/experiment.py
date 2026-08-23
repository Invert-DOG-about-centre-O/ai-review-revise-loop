"""
Semantic vs. lexical vs. token-level predictive entropy for selective QA,
in a fully controlled synthetic knowledge-base language model.

Trains a small from-scratch Transformer decoder (no internet / no
pretrained weights -- CPU only) on a synthetic country->capital fact base
expressed through multiple paraphrase templates, then compares three
uncertainty-quantification (UQ) methods for detecting wrong answers at
test time:

  1. Token-level predictive entropy (naive, per-token softmax entropy
     averaged over the generated sequence)
  2. Lexical entropy (Monte-Carlo entropy over CLUSTERS of *exact string
     match* among K sampled completions)
  3. Semantic entropy (Monte-Carlo entropy over clusters keyed by the
     extracted answer ENTITY, i.e. paraphrase-invariant clustering, a la
     Kuhn et al. 2023 "Semantic Uncertainty: Linguistic Invariances for
     Uncertainty Estimation in Natural Language Generation")

The whole pipeline (KB sampling, training, generation, evaluation) is
repeated over multiple random seeds so we can report mean +/- std of
AUROC and selective-prediction (risk-coverage) accuracy rather than a
single run. All randomness is seeded. Runtime target: a few minutes on
CPU total across all seeds.
"""
import math
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t_start = time.time()
N_SEEDS = 5
N_FACTS = 50
HELD_OUT_FRAC = 0.3
K = 12          # Monte-Carlo samples per test question
MAX_GEN = 8
TEMP = 1.0
EPOCHS = 60
BATCH = 64
D_MODEL, N_HEAD, N_LAYERS, FF = 64, 4, 2, 128

q_templates = [
    "capital of {c} ?",
    "{c} capital city is ?",
    "name capital of {c} ?",
    "which city is capital of {c} ?",
]
a_templates = [
    "{a}",
    "it is {a}",
    "the capital is {a}",
    "{a} is the capital",
]


class TinyLM(nn.Module):
    def __init__(self, vocab_size, pad_idx, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_len, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FF,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, x, pad_mask):
        B, T = x.shape
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        h = self.enc(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.head(h)


def risk_coverage(scores, correct_flags, fracs=(1.0, 0.8, 0.6, 0.4, 0.2)):
    order = np.argsort(scores)  # ascending uncertainty -> answered first
    correct_sorted = np.array(correct_flags)[order]
    n = len(correct_sorted)
    return {f: float(correct_sorted[:max(1, int(round(n * f)))].mean()) for f in fracs}


def run_seed(seed, epochs=EPOCHS):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    countries = [f"country{i}" for i in range(N_FACTS)]
    cities = [f"city{i}" for i in range(N_FACTS)]
    shuffled_cities = cities[:]
    rng.shuffle(shuffled_cities)
    kb = dict(zip(countries, shuffled_cities))

    n_held = int(N_FACTS * HELD_OUT_FRAC)
    shuffled_countries = countries[:]
    rng.shuffle(shuffled_countries)
    held_out_countries = set(shuffled_countries[:n_held])
    train_countries = [c for c in countries if c not in held_out_countries]

    specials = ["<pad>", "<bos>", "<eos>", "<sep>"]
    vocab_words = set()
    for t in q_templates:
        vocab_words.update(t.replace("{c}", "").split())
    for t in a_templates:
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
        for qt in q_templates:
            q_toks = [stoi[w] for w in qt.replace("{c}", country).split()]
            for at in a_templates:
                a_toks = [stoi[w] for w in at.replace("{a}", city).split()]
                train_seqs.append([BOS] + q_toks + [SEP] + a_toks + [EOS])
    rng.shuffle(train_seqs)
    max_len = max(len(s) for s in train_seqs) + 8

    def pad_batch(seqs):
        out = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s)
        return out

    model = TinyLM(len(vocab), PAD, max_len)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n_batches = math.ceil(len(train_seqs) / BATCH)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(train_seqs)
        for b in range(n_batches):
            chunk = train_seqs[b * BATCH:(b + 1) * BATCH]
            x = pad_batch(chunk)
            pad_mask = x == PAD
            logits = model(x[:, :-1], pad_mask[:, :-1])
            targets = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    final_loss = loss.item()
    model.eval()

    @torch.no_grad()
    def sample_completion(q_ids):
        seq = [BOS] + q_ids + [SEP]
        entropies = []
        for _ in range(MAX_GEN):
            x = torch.tensor(seq).unsqueeze(0)
            pad_mask = torch.zeros_like(x, dtype=torch.bool)
            logits = model(x, pad_mask)[0, -1] / TEMP
            probs = F.softmax(logits, dim=-1)
            nxt = torch.distributions.Categorical(probs).sample().item()
            entropies.append(-(probs * (probs + 1e-12).log()).sum().item())
            seq.append(nxt)
            if nxt == EOS:
                break
        answer_ids = seq[seq.index(SEP) + 1:]
        if answer_ids and answer_ids[-1] == EOS:
            answer_ids = answer_ids[:-1]
        return answer_ids, entropies

    def extract_city(answer_ids):
        for tid in answer_ids:
            if tid in city_token_ids:
                return itos[tid]
        return None

    def answer_string(answer_ids):
        return " ".join(itos[t] for t in answer_ids)

    test_seen = rng.sample(train_countries, min(20, len(train_countries)))
    test_unseen = list(held_out_countries)
    test_countries = [(c, "seen") for c in test_seen] + [(c, "unseen") for c in test_unseen]
    rng.shuffle(test_countries)

    records = []
    for country, split in test_countries:
        qt = rng.choice(q_templates)
        q_ids = [stoi[w] for w in qt.replace("{c}", country).split()]
        samples = [sample_completion(q_ids) for _ in range(K)]

        all_ent = [e for (_, ents) in samples for e in ents]
        pred_entropy = float(np.mean(all_ent)) if all_ent else 0.0

        strs = [answer_string(a) for (a, _) in samples]
        lex_counts = {}
        for s in strs:
            lex_counts[s] = lex_counts.get(s, 0) + 1
        lex_probs = np.array(list(lex_counts.values())) / K
        lex_entropy = float(-(lex_probs * np.log(lex_probs + 1e-12)).sum())

        extracted = [extract_city(a) for (a, _) in samples]
        sem_counts = {}
        for c in extracted:
            sem_counts[c] = sem_counts.get(c, 0) + 1
        sem_probs = np.array(list(sem_counts.values())) / K
        sem_entropy = float(-(sem_probs * np.log(sem_probs + 1e-12)).sum())

        maj_city = max(sem_counts.items(), key=lambda kv: kv[1])[0]
        correct = (maj_city == kb[country])

        records.append(dict(
            country=country, split=split, correct=bool(correct),
            pred_entropy=pred_entropy, lex_entropy=lex_entropy, sem_entropy=sem_entropy,
        ))

    y_wrong = np.array([0 if r["correct"] else 1 for r in records])
    correct_flags = [r["correct"] for r in records]
    methods = {
        "predictive_entropy_token": np.array([r["pred_entropy"] for r in records]),
        "lexical_entropy": np.array([r["lex_entropy"] for r in records]),
        "semantic_entropy": np.array([r["sem_entropy"] for r in records]),
        "random_baseline": np.random.RandomState(seed).rand(len(records)),
    }
    auroc = {name: (float(roc_auc_score(y_wrong, s)) if len(set(y_wrong)) > 1 else float("nan"))
             for name, s in methods.items()}
    rc = {name: risk_coverage(s, correct_flags) for name, s in methods.items()}
    acc = 1 - y_wrong.mean()
    acc_seen = np.mean([r["correct"] for r in records if r["split"] == "seen"])
    acc_unseen = np.mean([r["correct"] for r in records if r["split"] == "unseen"])

    return dict(seed=seed, final_train_loss=final_loss, n_test=len(records),
                overall_accuracy=float(acc), acc_seen=float(acc_seen), acc_unseen=float(acc_unseen),
                auroc=auroc, risk_coverage=rc, records=records)


def main():
    all_runs = []
    for seed in range(N_SEEDS):
        r = run_seed(seed)
        all_runs.append(r)
        print(f"[{time.time()-t_start:.1f}s] seed={seed} loss={r['final_train_loss']:.3f} "
              f"acc={r['overall_accuracy']:.3f} (seen={r['acc_seen']:.2f}, unseen={r['acc_unseen']:.2f}) "
              f"auroc={ {k: round(v,3) for k,v in r['auroc'].items()} }")
    _run_rest(all_runs)


def _run_rest(all_runs):
    # ---------------------------------------------------------------------------
    # Aggregate across seeds
    # ---------------------------------------------------------------------------
    method_names = ["predictive_entropy_token", "lexical_entropy", "semantic_entropy", "random_baseline"]
    agg = {"n_seeds": N_SEEDS, "n_test_per_seed": all_runs[0]["n_test"]}
    agg["overall_accuracy_mean"] = float(np.mean([r["overall_accuracy"] for r in all_runs]))
    agg["acc_seen_mean"] = float(np.mean([r["acc_seen"] for r in all_runs]))
    agg["acc_unseen_mean"] = float(np.mean([r["acc_unseen"] for r in all_runs]))

    agg["auroc"] = {}
    for m in method_names:
        vals = [r["auroc"][m] for r in all_runs if not math.isnan(r["auroc"][m])]
        agg["auroc"][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    agg["selective_accuracy_at_coverage"] = {}
    for m in method_names:
        per_frac = {}
        for f in (1.0, 0.8, 0.6, 0.4, 0.2):
            vals = [r["risk_coverage"][m][f] for r in all_runs]
            per_frac[f] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg["selective_accuracy_at_coverage"][m] = per_frac

    print(f"\n[{time.time()-t_start:.1f}s] === AGGREGATE OVER {N_SEEDS} SEEDS ===")
    print(json.dumps(agg, indent=2))

    # -----------------------------------------------------------------------
    # Ablation: underfit the model (fewer epochs) so genuine in-distribution
    # ("seen") errors occur too, not just embedding-unseen errors. This
    # isolates whether semantic-entropy's advantage over token/lexical
    # entropy holds even when we exclude the trivial "never-seen-token" case.
    # -----------------------------------------------------------------------
    ABLATION_EPOCHS = 14
    ABLATION_SEEDS = 5
    ablation_runs = []
    for seed in range(100, 100 + ABLATION_SEEDS):
        r = run_seed(seed, epochs=ABLATION_EPOCHS)
        ablation_runs.append(r)
        print(f"[{time.time()-t_start:.1f}s] [ablation] seed={seed} loss={r['final_train_loss']:.3f} "
              f"acc_seen={r['acc_seen']:.2f} acc_unseen={r['acc_unseen']:.2f}")

    seen_records = [rec for r in ablation_runs for rec in r["records"] if rec["split"] == "seen"]
    y_wrong_seen = np.array([0 if rec["correct"] else 1 for rec in seen_records])
    n_seen_wrong = int(y_wrong_seen.sum())
    ablation_summary = {
        "epochs": ABLATION_EPOCHS, "n_seeds": ABLATION_SEEDS,
        "n_seen_test_total": len(seen_records), "n_seen_wrong": n_seen_wrong,
        "acc_seen_mean": float(np.mean([r["acc_seen"] for r in ablation_runs])),
        "acc_unseen_mean": float(np.mean([r["acc_unseen"] for r in ablation_runs])),
    }
    if 0 < n_seen_wrong < len(seen_records):
        methods_seen = {
            "predictive_entropy_token": np.array([rec["pred_entropy"] for rec in seen_records]),
            "lexical_entropy": np.array([rec["lex_entropy"] for rec in seen_records]),
            "semantic_entropy": np.array([rec["sem_entropy"] for rec in seen_records]),
            "random_baseline": np.random.RandomState(0).rand(len(seen_records)),
        }
        ablation_summary["auroc_in_distribution_only"] = {
            name: float(roc_auc_score(y_wrong_seen, s)) for name, s in methods_seen.items()
        }
    else:
        ablation_summary["auroc_in_distribution_only"] = None
        ablation_summary["note"] = "no variation in seen-set correctness; AUROC undefined"

    print(f"[{time.time()-t_start:.1f}s] === ABLATION (in-distribution only, underfit model) ===")
    print(json.dumps(ablation_summary, indent=2))

    with open("results.json", "w") as f:
        json.dump({
            "aggregate": agg,
            "per_seed_runs": [{k: v for k, v in r.items() if k != "records"} for r in all_runs],
            "ablation_underfit_in_distribution": ablation_summary,
        }, f, indent=2)

    print(f"[{time.time()-t_start:.1f}s] DONE, total runtime {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
