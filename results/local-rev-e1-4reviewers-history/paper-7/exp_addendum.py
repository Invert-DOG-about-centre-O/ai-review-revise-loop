import time, random, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from exp import (gen_examples, TinyTransformer, build_batch, greedy_decode,
                  sample_decode, get_hidden_repr, VOCAB)

t0 = time.time()

def run_seed_addendum(seed, n_train=20000, n_test=800, epochs=6, k_samples=8):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_exs = gen_examples(n_train, rng)
    test_exs = gen_examples(n_test, rng)
    max_len = max(len(p) + len(t) + 1 for p, t, *_ in train_exs + test_exs) + 1

    model = TinyTransformer(len(VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    bs = 128
    for ep in range(epochs):
        random.Random(seed * 1000 + ep).shuffle(train_exs)
        for i in range(0, len(train_exs), bs):
            batch = train_exs[i:i + bs]
            x, lm = build_batch(batch, max_len)
            logits = model(x)
            logits_shift = logits[:, :-1]
            targets = x[:, 1:]
            lm_shift = lm[:, 1:]
            loss = F.cross_entropy(logits_shift.reshape(-1, logits_shift.size(-1)),
                                    targets.reshape(-1), reduction="none")
            loss = (loss * lm_shift.reshape(-1)).sum() / lm_shift.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    recs = []
    with torch.no_grad():
        for prompt, target, a, b, s, nd, nc in test_exs:
            true_ans = str(s)[::-1]
            pred, mmp, nme = greedy_decode(model, prompt)
            correct = int(pred == true_ans)
            samples = [sample_decode(model, prompt, rng=rng) for _ in range(k_samples)]
            modal = max(set(samples), key=samples.count)
            sc = samples.count(modal) / len(samples)
            hid = get_hidden_repr(model, prompt)
            recs.append(dict(correct=correct, mmp=mmp, nme=nme, sc=sc, nd=nd, nc=nc, hid=hid))

    y = np.array([r["correct"] for r in recs])
    mmp = np.array([r["mmp"] for r in recs])
    nd = np.array([r["nd"] for r in recs])
    nc = np.array([r["nc"] for r in recs])
    hid = np.stack([r["hid"] for r in recs])

    # difficulty gradient: accuracy and mean confidence by digit count and carry count
    diff = {}
    for d in sorted(set(nd.tolist())):
        m = nd == d
        if m.sum() >= 5:
            diff[f"digits_{d}"] = dict(n=int(m.sum()), acc=float(y[m].mean()), mean_mmp=float(mmp[m].mean()))
    carry_bucket = np.clip(nc, 0, 3)
    for c in sorted(set(carry_bucket.tolist())):
        m = carry_bucket == c
        if m.sum() >= 5:
            diff[f"carries_{c}"] = dict(n=int(m.sum()), acc=float(y[m].mean()), mean_mmp=float(mmp[m].mean()))

    # probe sample-size / regularization test via 2-fold CV (uses all 800 points as eval,
    # each point scored only when NOT in its fit fold)
    n = len(y)
    half = n // 2
    idx = np.arange(n)
    rs = np.random.RandomState(seed)
    rs.shuffle(idx)
    foldA, foldB = idx[:half], idx[half:]

    def cv_auroc(C):
        probs = np.zeros(n)
        for fit_idx, eval_idx in [(foldA, foldB), (foldB, foldA)]:
            hid_std = (hid - hid[fit_idx].mean(0)) / hid[fit_idx].std(0).clip(1e-8)
            clf = LogisticRegression(max_iter=1000, C=C)
            clf.fit(hid_std[fit_idx], y[fit_idx])
            probs[eval_idx] = clf.predict_proba(hid_std[eval_idx])[:, 1]
        return roc_auc_score(y, probs)

    probe_cv_default = cv_auroc(C=1.0)   # same effective fit size as original (~400), but both halves evaluated
    probe_cv_strongreg = cv_auroc(C=0.05)  # much stronger L2 regularization, same data

    return dict(seed=seed, difficulty=diff, probe_cv_default=probe_cv_default,
                probe_cv_strongreg=probe_cv_strongreg)


if __name__ == "__main__":
    results = []
    for seed in [0, 1, 2]:
        r = run_seed_addendum(seed)
        results.append(r)
        print(seed, r)
    with open("addendum_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("TOTAL TIME", time.time() - t0)
