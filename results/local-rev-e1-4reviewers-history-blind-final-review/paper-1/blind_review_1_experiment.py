"""Cheap uncertainty signals for error prediction: controlled synthetic-arithmetic replication."""
import random, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

VOCAB = list("0123456789+-=<>PB") + ["<bos>", "<eos>", "<pad>"]
# simpler: build char vocab dynamically
CHARS = "0123456789+-="
BOS, EOS, PAD = "^", "$", "#"
ALPHABET = CHARS + BOS + EOS + PAD
stoi = {c: i for i, c in enumerate(ALPHABET)}
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(ALPHABET)
MAX_LEN = 12

def make_example(rng):
    a = rng.randint(1, 99)
    b = rng.randint(1, 99)
    if rng.random() < 0.5:
        op = "+"
        ans = a + b
    else:
        op = "-"
        if b > a:
            a, b = b, a
        ans = a - b
    prompt = f"{a}{op}{b}="
    target = str(ans)
    return prompt, target

def encode(s):
    return [stoi[c] for c in s]

class TinyGPT(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=3, vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.blocks(h, mask=mask)
        h = self.ln_f(h)
        return self.head(h)

def train_model(seed, steps=1500, lr=3e-3, batch_size=64):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        prompts, targets = [], []
        for _ in range(batch_size):
            p, t = make_example(rng)
            prompts.append(p); targets.append(t)
        seqs = []
        for p, t in zip(prompts, targets):
            full = BOS + p + t + EOS
            seqs.append(full)
        maxlen = max(len(s) for s in seqs)
        maxlen = min(maxlen, MAX_LEN)
        x_batch, y_batch = [], []
        for s in seqs:
            ids = encode(s)[:maxlen]
            ids = ids + [stoi[PAD]] * (maxlen - len(ids))
            x_batch.append(ids[:-1])
            y_batch.append(ids[1:])
        x = torch.tensor(x_batch, dtype=torch.long)
        y = torch.tensor(y_batch, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=stoi[PAD])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append(loss.item())
    return model, losses

@torch.no_grad()
def generate(model, prompt, greedy=True, temperature=0.9, max_new=6):
    ids = encode(BOS + prompt)
    x = torch.tensor([ids], dtype=torch.long)
    entropies = []
    for _ in range(max_new):
        logits = model(x)[0, -1]
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        entropies.append(ent)
        if greedy:
            nxt = int(torch.argmax(logits))
        else:
            p = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(p, 1))
        if itos.get(nxt) == EOS:
            break
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        if x.shape[1] >= MAX_LEN:
            break
    out = "".join(itos[i] for i in x[0, len(ids):].tolist())
    return out, entropies

def parse_int(s):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None

def run_eval(model, n_problems=400, K=10, eval_seed=123, temperature=0.9):
    rng = random.Random(eval_seed)
    records = []
    for _ in range(n_problems):
        prompt, target = make_example(rng)
        true_ans = int(target)
        greedy_out, greedy_ents = generate(model, prompt, greedy=True)
        greedy_val = parse_int(greedy_out)
        correct = int(greedy_val == true_ans)
        token_entropy = float(np.mean(greedy_ents)) if greedy_ents else 0.0

        samples = []
        for _ in range(K):
            s_out, _ = generate(model, prompt, greedy=False, temperature=temperature)
            samples.append(parse_int(s_out))
        vals = [v for v in samples if v is not None] or [None]
        from collections import Counter
        counts = Counter(vals)
        total = sum(counts.values())
        sem_ent = -sum((c/total) * math.log(c/total) for c in counts.values())
        self_consistency = sum(1 for v in samples if v == greedy_val) / K

        records.append(dict(prompt=prompt, target=target, greedy_out=greedy_out, correct=correct,
                             token_entropy=token_entropy, semantic_entropy=sem_ent,
                             self_consistency=self_consistency, digit_len=len(target)))
    return records

def auroc_signals(records):
    y = np.array([r["correct"] for r in records])
    out = {}
    for sig, sign in [("token_entropy", -1), ("semantic_entropy", -1), ("self_consistency", 1)]:
        x = np.array([r[sig] for r in records]) * sign
        out[sig] = roc_auc_score(y, x)
    return out

def bootstrap_ci(records, sig, sign, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    y = np.array([r["correct"] for r in records])
    x = np.array([r[sig] for r in records]) * sign
    n = len(records)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], x[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals

def bootstrap_diff_pvalue(records, sig_a, sign_a, sig_b, sign_b, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    y = np.array([r["correct"] for r in records])
    xa = np.array([r[sig_a] for r in records]) * sign_a
    xb = np.array([r[sig_b] for r in records]) * sign_b
    n = len(records)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], xa[idx]) - roc_auc_score(y[idx], xb[idx]))
    diffs = np.array(diffs)
    p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(p_two_sided)

def main():
    t0 = time.time()
    SEEDS = [0, 1, 2, 3, 4]
    all_seed_results = {}
    all_seed_records = {}
    main_records = None
    main_losses = None
    for seed in SEEDS:
        model, losses = train_model(seed=seed)
        records = run_eval(model, n_problems=400, K=10, eval_seed=123)
        acc = np.mean([r["correct"] for r in records])
        aurocs = auroc_signals(records)
        all_seed_results[seed] = dict(greedy_accuracy=acc, aurocs=aurocs, loss_start=losses[0], loss_end=losses[-1])
        all_seed_records[seed] = records
        print(f"seed={seed} acc={acc:.3f} aurocs={aurocs}")
        if seed == 0:
            main_records = records
            main_losses = losses

    # Bootstrap CIs on seed-0 run (this is the "headline" run reported in the paper)
    cis = {}
    for sig, sign in [("token_entropy", -1), ("semantic_entropy", -1), ("self_consistency", 1)]:
        lo, hi, _ = bootstrap_ci(main_records, sig, sign)
        cis[sig] = (lo, hi)

    diff_sem_vs_token = bootstrap_diff_pvalue(main_records, "semantic_entropy", -1, "token_entropy", -1)
    diff_sc_vs_token = bootstrap_diff_pvalue(main_records, "self_consistency", 1, "token_entropy", -1)
    diff_sem_vs_sc = bootstrap_diff_pvalue(main_records, "semantic_entropy", -1, "self_consistency", 1)

    # Per-seed bootstrap paired tests (addresses reviewer request: is significance seed-0-specific?)
    per_seed_tests = {}
    for seed in SEEDS:
        recs = all_seed_records[seed]
        if len(set(r["correct"] for r in recs)) < 2:
            continue
        d_sem = bootstrap_diff_pvalue(recs, "semantic_entropy", -1, "token_entropy", -1)
        d_sc = bootstrap_diff_pvalue(recs, "self_consistency", 1, "token_entropy", -1)
        per_seed_tests[seed] = dict(
            semantic_vs_token=dict(zip(["mean_diff", "lo", "hi", "p"], d_sem)),
            selfcons_vs_token=dict(zip(["mean_diff", "lo", "hi", "p"], d_sc)),
        )
        print(f"seed={seed} p(sem<token)={d_sem[3]:.3f} p(sc<token)={d_sc[3]:.3f}")

    # Sign test on "token entropy wins in every seed" claim
    from scipy.stats import binomtest
    wins = sum(1 for s in SEEDS if all_seed_results[s]["aurocs"]["token_entropy"] ==
               max(all_seed_results[s]["aurocs"].values()))
    sign_test_p = binomtest(wins, len(SEEDS), 0.5, alternative="greater").pvalue

    # Correlation between seed accuracy and token-entropy advantage magnitude
    accs = np.array([all_seed_results[s]["greedy_accuracy"] for s in SEEDS])
    advantages = np.array([
        all_seed_results[s]["aurocs"]["token_entropy"] -
        np.mean([all_seed_results[s]["aurocs"]["semantic_entropy"], all_seed_results[s]["aurocs"]["self_consistency"]])
        for s in SEEDS
    ])
    acc_advantage_corr = float(np.corrcoef(accs, advantages)[0, 1])

    # K sweep on seed-0 model (retrain to get fresh generate-capable model object is expensive;
    # reuse cached model by retraining seed 0 once more for K sweep with different K)
    model0, _ = train_model(seed=0)
    k_sweep = {}
    for K in [2, 5, 10, 20]:
        recs = run_eval(model0, n_problems=400, K=K, eval_seed=123)
        k_sweep[K] = auroc_signals(recs)
        print(f"K={K} aurocs={k_sweep[K]}")

    # Digit-position mechanism check: does token entropy underperform more on multi-digit answers?
    by_len = {}
    for L in sorted(set(r["digit_len"] for r in main_records)):
        subset = [r for r in main_records if r["digit_len"] == L]
        if len(subset) < 20 or len(set(r["correct"] for r in subset)) < 2:
            continue
        by_len[L] = dict(n=len(subset), acc=float(np.mean([r["correct"] for r in subset])),
                          aurocs=auroc_signals(subset))
        print(f"digit_len={L} n={len(subset)} aurocs={by_len[L]['aurocs']}")

    elapsed = time.time() - t0
    summary = dict(
        seed0_greedy_accuracy=all_seed_results[0]["greedy_accuracy"],
        seed0_aurocs=all_seed_results[0]["aurocs"],
        seed0_loss=(main_losses[0], main_losses[-1]),
        multi_seed=all_seed_results,
        bootstrap_ci=cis,
        bootstrap_diff_semantic_vs_token=dict(zip(["mean_diff", "lo", "hi", "p"], diff_sem_vs_token)),
        bootstrap_diff_selfcons_vs_token=dict(zip(["mean_diff", "lo", "hi", "p"], diff_sc_vs_token)),
        bootstrap_diff_semantic_vs_selfcons=dict(zip(["mean_diff", "lo", "hi", "p"], diff_sem_vs_sc)),
        k_sweep=k_sweep,
        by_digit_len=by_len,
        per_seed_bootstrap_tests=per_seed_tests,
        seed_win_sign_test=dict(wins=wins, n=len(SEEDS), p_greater=float(sign_test_p)),
        accuracy_vs_advantage_correlation=acc_advantage_corr,
        elapsed_seconds=elapsed,
    )
    with open("blind_review_1_results_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open("blind_review_1_raw_results_v2.json", "w") as f:
        json.dump(main_records, f, indent=2)
    print(f"Total elapsed: {elapsed:.1f}s")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
