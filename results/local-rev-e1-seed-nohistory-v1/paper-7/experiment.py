import os, sys, json, random, time, math
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
import numpy as np
from sklearn.metrics import roc_auc_score

N_ENTITIES = 50
Q_TEMPLATES = ["capital of {c} ?", "{c} capital city is ?", "what city is the capital of {c} ?", "{c} -> capital ?"]
A_TEMPLATES = ["{a}", "it is {a}", "the capital is {a}", "{a} is the capital"]

def make_kb(seed):
    rng = random.Random(seed)
    countries = [f"country{i}" for i in range(N_ENTITIES)]
    cities = [f"city{i}" for i in range(N_ENTITIES)]
    shuffled = cities[:]
    rng.shuffle(shuffled)
    kb = dict(zip(countries, shuffled))
    idx = list(range(N_ENTITIES))
    rng.shuffle(idx)
    n_unseen = int(round(0.3 * N_ENTITIES))
    unseen_idx = set(idx[:n_unseen])
    seen_countries = [countries[i] for i in idx[n_unseen:]]
    unseen_countries = [countries[i] for i in idx[:n_unseen]]
    return kb, seen_countries, unseen_countries

def build_examples(countries, kb, rng, include_answer=True):
    ex = []
    for c in countries:
        a = kb[c]
        for qt in Q_TEMPLATES:
            q = qt.format(c=c)
            if include_answer:
                at = rng.choice(A_TEMPLATES)
                ans = at.format(a=a)
                ex.append((q, ans, c, a))
            else:
                ex.append((q, None, c, a))
    return ex

class Vocab:
    def __init__(self, tokens):
        self.itos = ["<pad>", "<bos>", "<eos>", "<sep>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def encode(self, toks):
        return [self.stoi[t] for t in toks]
    def __len__(self):
        return len(self.itos)

def tokenize(s):
    return s.replace("?", " ?").split()

class TinyTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, ff=128, max_len=32):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Embedding(max_len, d_model)
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff,
                                                   batch_first=True, activation="gelu")
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = torch.nn.Linear(d_model, vocab_size)
        self.max_len = max_len
    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.out(h)

def make_training_seqs(examples, vocab, max_len):
    seqs = []
    for q, a, c, city in examples:
        toks = ["<bos>"] + tokenize(q) + ["<sep>"] + tokenize(a) + ["<eos>"]
        ids = vocab.encode(toks)
        if len(ids) > max_len:
            ids = ids[:max_len]
        seqs.append(ids)
    return seqs

def pad_batch(seqs, pad_id, max_len):
    b = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        b[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return b

def train_model(train_examples, vocab, max_len, epochs, seed, lr=3e-3):
    torch.manual_seed(seed)
    model = TinyTransformer(len(vocab), max_len=max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = make_training_seqs(train_examples, vocab, max_len)
    pad_id = vocab.stoi["<pad>"]
    batch = pad_batch(seqs, pad_id, max_len)
    final_loss = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(batch.size(0))
        total_loss = 0.0
        for i in range(0, batch.size(0), 16):
            idx = perm[i:i+16]
            x = batch[idx]
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits = model(inp)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                       ignore_index=pad_id)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        final_loss = total_loss / batch.size(0)
    return model, final_loss

@torch.no_grad()
def sample_answer(model, vocab, q, max_len, temperature, gen, extra_noise_p=0.0, rng=None):
    pad_id = vocab.stoi["<pad>"]
    ids = vocab.encode(["<bos>"] + tokenize(q) + ["<sep>"])
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    entropies = []
    for _ in range(max_len - ids.size(1)):
        logits = model(ids)[0, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        entropies.append(ent)
        nxt = torch.multinomial(probs, 1, generator=gen).item()
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if vocab.itos[nxt] == "<eos>":
            break
    gen_toks = [vocab.itos[i] for i in ids[0].tolist()]
    if "<sep>" in gen_toks:
        gen_toks = gen_toks[gen_toks.index("<sep>") + 1:]
    gen_toks = [t for t in gen_toks if t not in ("<eos>", "<pad>", "<bos>")]
    text = " ".join(gen_toks)
    return text, entropies

CITY_SET = set(f"city{i}" for i in range(N_ENTITIES))

def extract_entity(text, noise_p=0.0, rng=None, all_cities=None):
    found = None
    for t in text.split():
        if t in CITY_SET:
            found = t
            break
    if noise_p > 0.0 and rng is not None and all_cities is not None:
        if rng.random() < noise_p:
            found = rng.choice(all_cities)
    return found

def run_eval(model, vocab, test_examples, max_len, K, seed, noise_p=0.0):
    gen = torch.Generator().manual_seed(seed * 100003 + 7)
    py_rng = random.Random(seed * 999 + 3)
    all_cities = [f"city{i}" for i in range(N_ENTITIES)]
    rows = []
    for q, _, c, true_city in test_examples:
        samples = []
        tok_ents = []
        for _ in range(K):
            text, ents = sample_answer(model, vocab, q, max_len, 1.0, gen)
            samples.append(text)
            tok_ents.extend(ents)
        entities = [extract_entity(s, noise_p, py_rng, all_cities) for s in samples]
        # tie-break deterministically: sort candidates (None last) instead of
        # max(set(...)), whose ordering depends on process PYTHONHASHSEED
        pred_ent = max(sorted(set(entities), key=lambda e: (e is None, e)), key=entities.count) if entities else None
        correct = int(pred_ent == true_city)
        token_entropy = float(np.mean(tok_ents)) if tok_ents else 0.0
        from collections import Counter
        lex_counts = Counter(samples)
        lex_probs = np.array(list(lex_counts.values())) / len(samples)
        lexical_entropy = float(-(lex_probs * np.log(lex_probs.clip(1e-12))).sum())
        ent_counts = Counter(entities)
        ent_probs = np.array(list(ent_counts.values())) / len(entities)
        semantic_entropy = float(-(ent_probs * np.log(ent_probs.clip(1e-12))).sum())
        rows.append(dict(country=c, true_city=true_city, correct=correct,
                          token_entropy=token_entropy, lexical_entropy=lexical_entropy,
                          semantic_entropy=semantic_entropy))
    return rows

def auroc_for(rows, key):
    y = [1 - r["correct"] for r in rows]
    if len(set(y)) < 2:
        return float("nan")
    scores = [r[key] for r in rows]
    return roc_auc_score(y, scores)

def selective_acc(rows, key, coverages=(1.0, 0.8, 0.6, 0.4, 0.2)):
    srt = sorted(rows, key=lambda r: r[key])
    out = []
    for cov in coverages:
        n = max(1, int(round(len(srt) * cov)))
        subset = srt[:n]
        out.append(float(np.mean([r["correct"] for r in subset])))
    return out

def full_run(seed, epochs, restrict_to_seen=False, K=12, noise_p=0.0):
    kb, seen_c, unseen_c = make_kb(seed=0)
    rng = random.Random(seed)
    train_ex = build_examples(seen_c, kb, rng, include_answer=True)
    all_toks = set()
    for q, a, c, city in train_ex:
        all_toks.update(tokenize(q))
        all_toks.update(tokenize(a))
    for c in list(kb.keys()):
        all_toks.update(tokenize(kb[c]))
    for c in seen_c + unseen_c:
        all_toks.update(tokenize(c))
    for qt in Q_TEMPLATES:
        all_toks.update(tokenize(qt.format(c="country0")))
    for at in A_TEMPLATES:
        all_toks.update(tokenize(at.format(a="city0")))
    vocab = Vocab(all_toks)
    max_len = 24
    model, final_loss = train_model(train_ex, vocab, max_len, epochs, seed)
    test_c = seen_c if restrict_to_seen else (seen_c + unseen_c)
    test_rng = random.Random(seed + 555)
    if restrict_to_seen:
        test_countries = seen_c
    else:
        test_countries = seen_c + unseen_c
    test_ex = build_examples(test_countries, kb, test_rng, include_answer=False)
    test_ex = [e for e in test_ex if e[0].startswith("capital of")]
    rows = run_eval(model, vocab, test_ex, max_len, K, seed, noise_p=noise_p)
    acc = float(np.mean([r["correct"] for r in rows]))
    aurocs = {k: auroc_for(rows, k) for k in ("token_entropy", "lexical_entropy", "semantic_entropy")}
    sel = {k: selective_acc(rows, k) for k in ("token_entropy", "lexical_entropy", "semantic_entropy")}
    return dict(seed=seed, epochs=epochs, final_loss=final_loss, n=len(rows), acc=acc,
                n_wrong=int(sum(1 - r["correct"] for r in rows)), auroc=aurocs, selective=sel)

def random_baseline_auroc(rows, seed):
    rng = np.random.RandomState(seed)
    y = [1 - r["correct"] for r in rows]
    if len(set(y)) < 2:
        return float("nan")
    scores = rng.rand(len(rows))
    return roc_auc_score(y, scores)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    t0 = time.time()
    results = {}

    if mode == "main":
        main_runs = []
        for seed in range(5):
            r = full_run(seed, epochs=60, restrict_to_seen=False)
            main_runs.append(r)
            print("main seed", seed, r["acc"], r["auroc"])
        results["main"] = main_runs

    elif mode == "ablation":
        abl_runs = []
        for seed in range(100, 105):
            r = full_run(seed, epochs=19, restrict_to_seen=True)
            abl_runs.append(r)
            print("ablation seed", seed, r["acc"], r["n_wrong"], r["auroc"])
        results["ablation"] = abl_runs

    elif mode == "repeat_determinism":
        # rerun same 5 ablation seeds twice under pinned single-thread deterministic settings
        reps = []
        for rep in range(2):
            rr = []
            for seed in range(100, 105):
                r = full_run(seed, epochs=19, restrict_to_seen=True)
                rr.append(r)
            reps.append(rr)
            print("repeat", rep, [r["auroc"] for r in rr])
        results["repeat_determinism"] = reps

    elif mode == "ablation_large":
        # larger seed count at the epoch=19 operating point, to tighten the
        # CI on the semantic-vs-lexical AUROC gap and paired-test it against 0
        from scipy.stats import wilcoxon
        N_SEEDS = 30
        runs = [full_run(seed, epochs=19, restrict_to_seen=True) for seed in range(200, 200 + N_SEEDS)]
        aurocs = {k: [r["auroc"][k] for r in runs] for k in ("token_entropy", "lexical_entropy", "semantic_entropy")}
        means = {k: float(np.nanmean(v)) for k, v in aurocs.items()}
        stds = {k: float(np.nanstd(v)) for k, v in aurocs.items()}
        diffs = np.array(aurocs["semantic_entropy"]) - np.array(aurocs["lexical_entropy"])
        diffs = diffs[~np.isnan(diffs)]
        mean_diff = float(np.mean(diffs))
        # paired bootstrap CI on the mean semantic-lexical gap
        rng = np.random.RandomState(0)
        boot = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(10000)]
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        try:
            wstat, wp = wilcoxon(diffs)
            wp = float(wp)
        except Exception:
            wp = float("nan")
        results["ablation_large"] = dict(n_seeds=N_SEEDS, auroc_mean=means, auroc_std=stds,
                                          semantic_minus_lexical_mean=mean_diff,
                                          bootstrap_ci95=[ci_lo, ci_hi], wilcoxon_p=wp)
        print("ablation_large", means, "diff", mean_diff, "CI", ci_lo, ci_hi, "wilcoxon p", wp)

    elif mode == "epoch_sweep":
        sweep = {}
        for ep in [15, 17, 19, 21, 23]:
            runs = [full_run(seed, epochs=ep, restrict_to_seen=True) for seed in range(100, 105)]
            aurocs = {k: float(np.nanmean([r["auroc"][k] for r in runs])) for k in ("token_entropy", "lexical_entropy", "semantic_entropy")}
            accs = float(np.mean([r["acc"] for r in runs]))
            sweep[ep] = dict(auroc=aurocs, acc=accs)
            print("epoch", ep, "acc", accs, "auroc", aurocs)
        results["epoch_sweep"] = sweep

    elif mode == "noisy_extractor":
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        out = {}
        for p in noise_levels:
            runs = [full_run(seed, epochs=19, restrict_to_seen=True, noise_p=p) for seed in range(100, 105)]
            aurocs = {k: float(np.nanmean([r["auroc"][k] for r in runs])) for k in ("token_entropy", "lexical_entropy", "semantic_entropy")}
            out[p] = aurocs
            print("noise", p, aurocs)
        results["noisy_extractor"] = out

    print("elapsed", time.time() - t0)
    with open(f"results_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)
