import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, json, math
from sklearn.metrics import roc_auc_score

VOCAB = list("0123456789+= ") + ["<pad>", "<eos>"]
stoi = {c: i for i, c in enumerate(VOCAB)}
PAD, EOS = stoi["<pad>"], stoi["<eos>"]

def encode(s):
    return [stoi[c] for c in s]

def make_example(rng):
    a = rng.randint(1, 98)
    b = rng.randint(1, 98)
    q = f"{a}+{b}="
    ans = str(a + b)
    return q, ans, a + b

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=48, n_heads=4, n_layers=2, max_len=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)

def pad_batch(seqs, max_len):
    out = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out

def train_model(seed, steps=600, batch_size=64, lr=3e-3, max_len=16, device="cpu"):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyTransformer(len(VOCAB), max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        seqs = []
        for _ in range(batch_size):
            q, ans, _ = make_example(rng)
            seq = encode(q + ans) + [EOS]
            seqs.append(seq)
        x = pad_batch([s[:-1] for s in seqs], max_len - 1)
        y = pad_batch([s[1:] for s in seqs], max_len - 1)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def generate_greedy(model, prompt_ids, max_new=4, device="cpu"):
    seq = list(prompt_ids)
    logps = []
    ent1 = None
    for i in range(max_new):
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        probs = F.softmax(logits, dim=-1)
        if i == 0:
            ent1 = float(-(probs * torch.log(probs + 1e-12)).sum())
        nxt = int(torch.argmax(probs))
        lp = float(torch.log(probs[nxt] + 1e-12))
        logps.append(lp)
        if nxt == EOS:
            break
        seq.append(nxt)
    return seq[len(prompt_ids):], logps, ent1

@torch.no_grad()
def generate_sample(model, prompt_ids, temperature=1.0, max_new=4, device="cpu"):
    seq = list(prompt_ids)
    for i in range(max_new):
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        if nxt == EOS:
            break
        seq.append(nxt)
    return seq[len(prompt_ids):]

def decode_answer(ids):
    chars = []
    for i in ids:
        if i == EOS or i == PAD:
            break
        chars.append(VOCAB[i])
    return "".join(chars)

def parse_int(s):
    try:
        return int(s)
    except Exception:
        return None

def run_eval(model, n_test=400, K=8, eval_seed=42, device="cpu"):
    rng = random.Random(eval_seed)
    records = []
    for _ in range(n_test):
        q, ans, correct_val = make_example(rng)
        prompt_ids = encode(q)
        greedy_ids, logps, ent1 = generate_greedy(model, prompt_ids, device=device)
        greedy_str = decode_answer(greedy_ids)
        greedy_val = parse_int(greedy_str)
        is_correct = (greedy_val == correct_val)
        mean_logp = sum(logps) / max(len(logps), 1)
        samples = []
        for _ in range(K):
            s_ids = generate_sample(model, prompt_ids, device=device)
            s_str = decode_answer(s_ids)
            samples.append(parse_int(s_str))
        records.append(dict(
            q=q, correct_val=correct_val, greedy_val=greedy_val, is_correct=bool(is_correct),
            ent1=ent1, mean_logp=mean_logp, samples=samples,
        ))
    return records

def semantic_entropy(samples):
    vals = [s for s in samples if s is not None]
    if not vals:
        return 0.0
    from collections import Counter
    c = Counter(vals)
    n = len(samples)
    ent = 0.0
    for v, cnt in c.items():
        p = cnt / n
        ent -= p * math.log(p + 1e-12)
    return ent

def self_consistency(samples):
    from collections import Counter
    c = Counter(samples)
    if not c:
        return 0.0
    modal_count = c.most_common(1)[0][1]
    return modal_count / len(samples)

def compute_signals(records, K=8):
    out = []
    for r in records:
        samp = r["samples"][:K]
        se = semantic_entropy(samp)
        sc = self_consistency(samp)
        out.append(dict(
            is_correct=r["is_correct"],
            ent1=r["ent1"],
            mean_logp=r["mean_logp"],
            se=se,
            sc=sc,
        ))
    return out

def auroc_for_wrong(scores, is_correct, higher_means_wrong=True):
    y = np.array([0 if c else 1 for c in is_correct])  # 1 = wrong
    s = np.array(scores)
    if not higher_means_wrong:
        s = -s
    if len(set(y.tolist())) < 2:
        return float("nan")
    return roc_auc_score(y, s)

def bootstrap_auroc_diff(scores_a, scores_b, is_correct, higher_a_wrong, higher_b_wrong, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    y = np.array([0 if c else 1 for c in is_correct])
    a = np.array(scores_a) * (1 if higher_a_wrong else -1)
    b = np.array(scores_b) * (1 if higher_b_wrong else -1)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        auc_a = roc_auc_score(yy, a[idx])
        auc_b = roc_auc_score(yy, b[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le0 = float((diffs <= 0).mean())
    return float(lo), float(hi), p_le0, float(diffs.mean())

if __name__ == "__main__":
    import time, sys
    t0 = time.time()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    all_results = []
    for seed in range(n_seeds):
        model = train_model(seed=seed)
        records = run_eval(model, n_test=400, K=8, eval_seed=1000 + seed)
        sig = compute_signals(records, K=8)
        acc = np.mean([r["is_correct"] for r in records])
        ent1 = [s["ent1"] for s in sig]
        mlp = [s["mean_logp"] for s in sig]
        se = [s["se"] for s in sig]
        sc = [s["sc"] for s in sig]
        isc = [s["is_correct"] for s in sig]
        auc_ent1 = auroc_for_wrong(ent1, isc, higher_means_wrong=True)
        auc_mlp = auroc_for_wrong(mlp, isc, higher_means_wrong=False)
        auc_se = auroc_for_wrong(se, isc, higher_means_wrong=True)
        auc_sc = auroc_for_wrong(sc, isc, higher_means_wrong=False)
        lo1, hi1, p1, m1 = bootstrap_auroc_diff(mlp, se, isc, False, True, n_boot=1000, seed=seed)
        lo2, hi2, p2, m2 = bootstrap_auroc_diff(mlp, sc, isc, False, False, n_boot=1000, seed=seed)
        res = dict(seed=seed, acc=acc, auc_ent1=auc_ent1, auc_mlp=auc_mlp, auc_se=auc_se, auc_sc=auc_sc,
                   diff_mlp_se=dict(lo=lo1, hi=hi1, p_le0=p1, mean=m1),
                   diff_mlp_sc=dict(lo=lo2, hi=hi2, p_le0=p2, mean=m2))
        all_results.append(res)
        print(json.dumps(res))
    print("TOTAL_TIME", time.time() - t0)
    with open("multiseed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
