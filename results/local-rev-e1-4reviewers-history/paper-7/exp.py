import time, random, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

t0 = time.time()

VOCAB = list("0123456789+= ") + ["<eos>", "<pad>"]
stoi = {c: i for i, c in enumerate(VOCAB)}
PAD = stoi["<pad>"]
EOS = stoi["<eos>"]

def gen_examples(n, rng):
    exs = []
    for _ in range(n):
        da = rng.randint(1, 4)
        db = rng.randint(1, 4)
        a = rng.randint(10**(da-1) if da > 1 else 0, 10**da - 1)
        b = rng.randint(10**(db-1) if db > 1 else 0, 10**db - 1)
        s = a + b
        prompt = f"{a}+{b}="
        target = str(s)[::-1]
        n_carries = 0
        sa, sb = str(a)[::-1], str(b)[::-1]
        carry = 0
        for i in range(max(len(sa), len(sb))):
            da_ = int(sa[i]) if i < len(sa) else 0
            db_ = int(sb[i]) if i < len(sb) else 0
            tot = da_ + db_ + carry
            carry = 1 if tot >= 10 else 0
            if carry:
                n_carries += 1
        exs.append((prompt, target, a, b, s, max(da, db), n_carries))
    return exs

def encode(s):
    return [stoi[c] for c in s]

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x, return_hidden=False):
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

def build_batch(exs, max_len):
    seqs, loss_masks = [], []
    for prompt, target, *_ in exs:
        full = prompt + target
        ids = encode(full) + [EOS]
        lm = [0]*len(encode(prompt)) + [1]*(len(encode(target))+1)
        pad_n = max_len - len(ids)
        ids = ids + [PAD]*pad_n
        lm = lm + [0]*pad_n
        seqs.append(ids)
        loss_masks.append(lm)
    return torch.tensor(seqs), torch.tensor(loss_masks)

def greedy_decode(model, prompt, max_new=6):
    ids = encode(prompt)
    x = torch.tensor([ids])
    probs_seen, ents_seen = [], []
    for _ in range(max_new):
        logits = model(x)
        last = logits[0, -1]
        p = F.softmax(last, dim=-1)
        ent = -(p * (p.clamp_min(1e-12)).log()).sum().item()
        tok = p.argmax().item()
        probs_seen.append(p[tok].item())
        ents_seen.append(ent)
        if tok == EOS:
            break
        x = torch.cat([x, torch.tensor([[tok]])], dim=1)
    out_ids = x[0, len(ids):].tolist()
    digits = "".join(VOCAB[i] for i in out_ids if i < 10 or VOCAB[i] not in ("<eos>","<pad>"))
    digits = "".join(c for c in digits if c.isdigit())
    return digits, np.mean(probs_seen) if probs_seen else 0.0, -np.mean(ents_seen) if ents_seen else 0.0

def sample_decode(model, prompt, max_new=6, temp=1.0, rng=None):
    ids = encode(prompt)
    x = torch.tensor([ids])
    g = torch.Generator().manual_seed(rng.randint(0, 10**9))
    for _ in range(max_new):
        logits = model(x)
        last = logits[0, -1] / temp
        p = F.softmax(last, dim=-1)
        tok = torch.multinomial(p, 1, generator=g).item()
        if tok == EOS:
            break
        x = torch.cat([x, torch.tensor([[tok]])], dim=1)
    out_ids = x[0, len(ids):].tolist()
    digits = "".join(c for c in "".join(VOCAB[i] for i in out_ids) if c.isdigit())
    return digits

def get_hidden_repr(model, prompt):
    ids = encode(prompt)
    x = torch.tensor([ids])
    with torch.no_grad():
        _, h = model(x, return_hidden=True)
    return h[0, -1].numpy()

def run_seed(seed, n_train=20000, n_test=2000, epochs=6, k_samples=8):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_exs = gen_examples(n_train, rng)
    test_exs = gen_examples(n_test, rng)
    max_len = max(len(p)+len(t)+1 for p, t, *_ in train_exs+test_exs) + 1

    model = TinyTransformer(len(VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    bs = 128
    losses = []
    for ep in range(epochs):
        random.Random(seed*1000+ep).shuffle(train_exs)
        tot_loss, nb = 0.0, 0
        for i in range(0, len(train_exs), bs):
            batch = train_exs[i:i+bs]
            x, lm = build_batch(batch, max_len)
            logits = model(x)
            logits_shift = logits[:, :-1]
            targets = x[:, 1:]
            lm_shift = lm[:, 1:]
            loss = F.cross_entropy(logits_shift.reshape(-1, logits_shift.size(-1)),
                                    targets.reshape(-1), reduction="none")
            loss = (loss * lm_shift.reshape(-1)).sum() / lm_shift.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
        losses.append(tot_loss/nb)

    model.eval()
    recs = []
    with torch.no_grad():
        for prompt, target, a, b, s, nd, nc in test_exs:
            true_ans = str(s)[::-1]
            pred, mmp, nme = greedy_decode(model, prompt)
            correct = int(pred == true_ans)
            samples = [sample_decode(model, prompt, rng=rng) for _ in range(k_samples)]
            if samples:
                modal = max(set(samples), key=samples.count)
                sc = samples.count(modal) / len(samples)
            else:
                sc = 0.0
            hid = get_hidden_repr(model, prompt)
            recs.append(dict(correct=correct, mmp=mmp, nme=nme, sc=sc, nd=nd, nc=nc, hid=hid))

    y = np.array([r["correct"] for r in recs])
    mmp = np.array([r["mmp"] for r in recs])
    nme = np.array([r["nme"] for r in recs])
    sc = np.array([r["sc"] for r in recs])
    hid = np.stack([r["hid"] for r in recs])

    n = len(y)
    half = n // 2
    idx = np.arange(n)
    rs = np.random.RandomState(seed)
    rs.shuffle(idx)
    fit_idx, eval_idx = idx[:half], idx[half:]

    def auroc(x_):
        return roc_auc_score(y[eval_idx], x_[eval_idx])

    auroc_mmp = auroc(mmp)
    auroc_nme = auroc(nme)
    auroc_sc = auroc(sc)

    Xall = np.stack([mmp, nme, sc], axis=1)
    Xall = (Xall - Xall[fit_idx].mean(0)) / Xall[fit_idx].std(0).clip(1e-8)
    clf = LogisticRegression()
    clf.fit(Xall[fit_idx], y[fit_idx])
    combo_probs = clf.predict_proba(Xall)[:, 1]
    auroc_combo = roc_auc_score(y[eval_idx], combo_probs[eval_idx])

    # linear hidden-state probe (training-required signal, Semantic-Entropy-Probes-style)
    hid_std = (hid - hid[fit_idx].mean(0)) / hid[fit_idx].std(0).clip(1e-8)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(hid_std[fit_idx], y[fit_idx])
    probe_probs = probe.predict_proba(hid_std)[:, 1]
    auroc_probe = roc_auc_score(y[eval_idx], probe_probs[eval_idx])

    def ece(probs, ytrue, bins=10):
        edges = np.linspace(0, 1, bins+1)
        e = 0.0
        for i in range(bins):
            m = (probs >= edges[i]) & (probs < edges[i+1] if i < bins-1 else probs <= edges[i+1])
            if m.sum() == 0:
                continue
            conf = probs[m].mean()
            acc = ytrue[m].mean()
            e += (m.sum()/len(probs)) * abs(conf - acc)
        return e

    ece_sc_raw = ece(sc[eval_idx], y[eval_idx])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(sc[fit_idx], y[fit_idx])
    sc_cal = iso.predict(sc[eval_idx])
    ece_sc_iso = ece(sc_cal, y[eval_idx])
    ece_combo = ece(combo_probs[eval_idx], y[eval_idx])
    ece_probe = ece(probe_probs[eval_idx], y[eval_idx])

    acc = y.mean()

    return dict(seed=seed, acc=acc, auroc_mmp=auroc_mmp, auroc_nme=auroc_nme, auroc_sc=auroc_sc,
                auroc_combo=auroc_combo, auroc_probe=auroc_probe,
                ece_sc_raw=ece_sc_raw, ece_sc_iso=ece_sc_iso, ece_combo=ece_combo, ece_probe=ece_probe,
                final_loss=losses[-1])

if __name__ == "__main__":
    results = []
    for seed in [0, 1, 2]:
        r = run_seed(seed, n_train=20000, n_test=800, epochs=6, k_samples=8)
        results.append(r)
        print(seed, r)
    with open("multi_seed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("TOTAL TIME", time.time() - t0)
