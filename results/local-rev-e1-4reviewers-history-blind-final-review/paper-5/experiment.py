import time, json, math
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

t_start = time.time()

DIGITS = "0123456789"
VOCAB = list(DIGITS) + ["+", "="]
STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
DEVICE = "cpu"

def encode_example(a, b):
    s = a + b
    inp = f"{a:02d}+{b:02d}="
    tgt = f"{s:03d}"
    return inp, tgt

def batch_tensor(pairs):
    xs, ys = [], []
    for a, b in pairs:
        inp, tgt = encode_example(a, b)
        full = inp + tgt
        ids = [STOI[c] for c in full]
        xs.append(ids)
    xs = torch.tensor(xs, dtype=torch.long)
    return xs  # (B, 9): 6 input chars + 3 target chars

class TinyTransformer(nn.Module):
    def __init__(self, vocab=V, d_model=64, n_head=4, n_layer=2, d_ff=128, max_len=9):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, mask=self.causal_mask[:L, :L])
        h = self.ln(h)
        logits = self.head(h)
        return logits, h

def sample_pairs(rng, n):
    a = rng.integers(0, 100, size=n)
    b = rng.integers(0, 100, size=n)
    return list(zip(a.tolist(), b.tolist()))

def train_model(seed, steps, batch_size=128, lr=3e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        pairs = sample_pairs(rng, batch_size)
        x = batch_tensor(pairs).to(DEVICE)
        logits, _ = model(x[:, :-1])
        # predict positions 6,7,8 (target chars) from inputs at 5,6,7
        target_logits = logits[:, 5:8, :]
        target = x[:, 6:9]
        loss = nn.functional.cross_entropy(target_logits.reshape(-1, V), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def greedy_eval(model, pairs):
    x = batch_tensor(pairs).to(DEVICE)
    inp = x[:, :6]
    hid_states = []
    generated = inp
    maxprobs, entropies = [], []
    for t in range(3):
        logits, h = model(generated)
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        maxp, next_tok = probs.max(dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        maxprobs.append(maxp)
        entropies.append(ent)
        hid_states.append(h[:, -1, :])
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
    pred_ids = generated[:, 6:9]
    target_ids = x[:, 6:9]
    correct = (pred_ids == target_ids).all(dim=1).cpu().numpy()
    maxprob_mean = torch.stack(maxprobs, dim=1).mean(dim=1).cpu().numpy()
    entropy_mean = torch.stack(entropies, dim=1).mean(dim=1).cpu().numpy()
    mean_hidden = torch.stack(hid_states, dim=1).mean(dim=1).cpu().numpy()
    return correct, maxprob_mean, entropy_mean, mean_hidden, pred_ids.cpu().numpy()

@torch.no_grad()
def sample_k(model, pairs, K, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    x = batch_tensor(pairs).to(DEVICE)
    inp = x[:, :6]
    B = inp.shape[0]
    all_samples = np.zeros((B, K, 3), dtype=np.int64)
    for k in range(K):
        generated = inp
        for t in range(3):
            logits, _ = model(generated)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(1)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
        all_samples[:, k, :] = generated[:, 6:9].cpu().numpy()
    return all_samples  # (B, K, 3)

def sample_agree_and_entropy(all_samples, greedy_pred, k_use):
    B, K, _ = all_samples.shape
    k_use = min(k_use, K)
    sub = all_samples[:, :k_use, :]
    agree = (sub == greedy_pred[:, None, :]).all(axis=2).mean(axis=1)
    entropies = np.zeros(B)
    for i in range(B):
        strs = [tuple(sub[i, k]) for k in range(k_use)]
        vals, counts = np.unique(strs, axis=0, return_counts=True) if False else (None, None)
        from collections import Counter
        c = Counter(strs)
        p = np.array(list(c.values())) / k_use
        entropies[i] = -(p * np.log(p + 1e-12)).sum()
    return agree, entropies

def ece(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(conf)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        e += (mask.sum() / total) * abs(acc - avg_conf)
    return e

def bootstrap_auroc_ci(scores, labels, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    if labels.sum() == 0 or labels.sum() == n:
        return None
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        aucs.append(roc_auc_score(lb, scores[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def bootstrap_auroc_diff_ci(scores_a, scores_b, labels, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        try:
            auc_a = roc_auc_score(lb, scores_a[idx])
            auc_b = roc_auc_score(lb, scores_b[idx])
        except ValueError:
            continue
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

results = {}

# ---- Main experiment: multi-seed at the ~19%-error step budget ----
N_TEST = 500
K = 20
STEPS_MAIN = 260
SEEDS = [0, 1, 2, 3, 4]

main_runs = []
for seed in SEEDS:
    model = train_model(seed=seed, steps=STEPS_MAIN)
    rng = np.random.default_rng(1000 + seed)
    test_pairs = sample_pairs(rng, N_TEST)
    correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
    incorrect = (~correct).astype(int)
    all_samples = sample_k(model, test_pairs, K=K, seed=seed)
    agree20, sent20 = sample_agree_and_entropy(all_samples, greedy_pred, K)

    auc_maxprob = roc_auc_score(incorrect, -maxprob)
    auc_negent = roc_auc_score(incorrect, entropy)
    auc_agree = roc_auc_score(incorrect, -agree20)
    auc_sent = roc_auc_score(incorrect, sent20)

    ece_maxprob = ece(maxprob, correct.astype(float))
    ece_agree = ece(agree20, correct.astype(float))

    # K-sensitivity: reuse the same 20 samples, subsample to K=5
    agree5, sent5 = sample_agree_and_entropy(all_samples, greedy_pred, 5)
    auc_agree5 = roc_auc_score(incorrect, -agree5)
    auc_sent5 = roc_auc_score(incorrect, sent5)

    # Probe: 50/50 split, three feature sets (full, hidden-only, maxprob+entropy-only)
    half = N_TEST // 2
    feat_full = np.concatenate([maxprob[:, None], entropy[:, None], hidden], axis=1)
    feat_hidden = hidden
    feat_cheap2 = np.concatenate([maxprob[:, None], entropy[:, None]], axis=1)

    def fit_probe_auroc(feat):
        mu, sd = feat[:half].mean(0), feat[:half].std(0) + 1e-8
        Xtr = (feat[:half] - mu) / sd
        Xte = (feat[half:] - mu) / sd
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr, sent20[:half])
        pred_te = ridge.predict(Xte)
        r2 = ridge.score(Xte, sent20[half:])
        auc = roc_auc_score(incorrect[half:], pred_te)
        return r2, auc

    r2_full, auc_probe_full = fit_probe_auroc(feat_full)
    r2_hidden, auc_probe_hidden = fit_probe_auroc(feat_hidden)
    r2_cheap2, auc_probe_cheap2 = fit_probe_auroc(feat_cheap2)
    auc_true_holdout = roc_auc_score(incorrect[half:], sent20[half:])
    auc_maxprob_holdout = roc_auc_score(incorrect[half:], -maxprob[half:])

    acc = correct.mean()
    main_runs.append(dict(
        seed=seed, accuracy=float(acc),
        auc_maxprob=float(auc_maxprob), auc_negent=float(auc_negent),
        auc_agree=float(auc_agree), auc_sent=float(auc_sent),
        ece_maxprob=float(ece_maxprob), ece_agree=float(ece_agree),
        auc_agree_k5=float(auc_agree5), auc_sent_k5=float(auc_sent5),
        r2_probe_full=float(r2_full), auc_probe_full=float(auc_probe_full),
        r2_probe_hidden=float(r2_hidden), auc_probe_hidden=float(auc_probe_hidden),
        r2_probe_cheap2=float(r2_cheap2), auc_probe_cheap2=float(auc_probe_cheap2),
        auc_true_holdout=float(auc_true_holdout), auc_maxprob_holdout=float(auc_maxprob_holdout),
    ))

# Bootstrap CI on seed=0 run for the two headline comparisons
seed0_model = train_model(seed=0, steps=STEPS_MAIN)
rng0 = np.random.default_rng(1000)
test_pairs0 = sample_pairs(rng0, N_TEST)
correct0, maxprob0, entropy0, hidden0, greedy_pred0 = greedy_eval(seed0_model, test_pairs0)
incorrect0 = (~correct0).astype(int)
samples0 = sample_k(seed0_model, test_pairs0, K=K, seed=0)
agree0, sent0 = sample_agree_and_entropy(samples0, greedy_pred0, K)

ci_maxprob = bootstrap_auroc_ci(-maxprob0, incorrect0)
ci_agree = bootstrap_auroc_ci(-agree0, incorrect0)
ci_diff_maxprob_vs_agree = bootstrap_auroc_diff_ci(-maxprob0, -agree0, incorrect0)

results["main_multiseed"] = main_runs
results["bootstrap_ci"] = dict(
    auc_maxprob_ci=ci_maxprob, auc_agree_ci=ci_agree,
    diff_maxprob_minus_agree_ci=ci_diff_maxprob_vs_agree,
)

# aggregate across seeds
def agg(key):
    vals = [r[key] for r in main_runs]
    return dict(mean=float(np.mean(vals)), std=float(np.std(vals)))

agg_keys = ["accuracy", "auc_maxprob", "auc_negent", "auc_agree", "auc_sent",
            "ece_maxprob", "ece_agree", "auc_agree_k5", "auc_sent_k5",
            "r2_probe_full", "auc_probe_full", "r2_probe_hidden", "auc_probe_hidden",
            "r2_probe_cheap2", "auc_probe_cheap2", "auc_true_holdout", "auc_maxprob_holdout"]
results["aggregate"] = {k: agg(k) for k in agg_keys}

# ---- Step-budget sweep: does maxprob>self-consistency ranking hold at other error rates? ----
STEP_BUDGETS = [150, 260, 450]
sweep = []
for steps in STEP_BUDGETS:
    model = train_model(seed=0, steps=steps)
    rng = np.random.default_rng(2000 + steps)
    test_pairs = sample_pairs(rng, N_TEST)
    correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
    incorrect = (~correct).astype(int)
    acc = correct.mean()
    if incorrect.sum() == 0 or incorrect.sum() == N_TEST:
        sweep.append(dict(steps=steps, accuracy=float(acc), auroc_undefined=True))
        continue
    samples = sample_k(model, test_pairs, K=K, seed=steps)
    agree, sent = sample_agree_and_entropy(samples, greedy_pred, K)
    auc_maxprob = roc_auc_score(incorrect, -maxprob)
    auc_agree = roc_auc_score(incorrect, -agree)
    auc_sent = roc_auc_score(incorrect, sent)
    sweep.append(dict(steps=steps, accuracy=float(acc), error_rate=float(1 - acc),
                       auc_maxprob=float(auc_maxprob), auc_agree=float(auc_agree),
                       auc_sent=float(auc_sent), maxprob_wins=bool(auc_maxprob > auc_agree)))

results["step_budget_sweep"] = sweep
results["elapsed_seconds"] = time.time() - t_start

with open("results_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results["aggregate"], indent=2))
print("bootstrap:", results["bootstrap_ci"])
print("sweep:", sweep)
print("elapsed:", results["elapsed_seconds"])
