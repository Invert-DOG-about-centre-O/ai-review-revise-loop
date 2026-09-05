import time, random, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from exp import (gen_examples, TinyTransformer, build_batch, greedy_decode,
                  sample_decode, get_hidden_repr, VOCAB)

t0 = time.time()

def run_seed_full(seed, n_train=20000, n_test=800, epochs=6, k_samples=8):
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
    nme = np.array([r["nme"] for r in recs])
    sc = np.array([r["sc"] for r in recs])
    nd = np.array([r["nd"] for r in recs])
    nc = np.array([r["nc"] for r in recs])
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

    hid_std = (hid - hid[fit_idx].mean(0)) / hid[fit_idx].std(0).clip(1e-8)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(hid_std[fit_idx], y[fit_idx])
    probe_probs = probe.predict_proba(hid_std)[:, 1]
    auroc_probe = roc_auc_score(y[eval_idx], probe_probs[eval_idx])

    # probe CV (2-fold, both regularization strengths) -- same protocol as exp_addendum.py
    foldA, foldB = idx[:half], idx[half:]
    def cv_auroc(C):
        probs = np.zeros(n)
        for fi, ei in [(foldA, foldB), (foldB, foldA)]:
            hs = (hid - hid[fi].mean(0)) / hid[fi].std(0).clip(1e-8)
            clf2 = LogisticRegression(max_iter=1000, C=C)
            clf2.fit(hs[fi], y[fi])
            probs[ei] = clf2.predict_proba(hs[ei])[:, 1]
        return roc_auc_score(y, probs)
    probe_cv_default = cv_auroc(C=1.0)
    probe_cv_strongreg = cv_auroc(C=0.05)

    def ece(probs, ytrue, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= edges[i + 1])
            if m.sum() == 0:
                continue
            conf = probs[m].mean()
            acc = ytrue[m].mean()
            e += (m.sum() / len(probs)) * abs(conf - acc)
        return e

    ece_sc_raw = ece(sc[eval_idx], y[eval_idx])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(sc[fit_idx], y[fit_idx])
    sc_cal = iso.predict(sc[eval_idx])
    ece_sc_iso = ece(sc_cal, y[eval_idx])
    ece_combo = ece(combo_probs[eval_idx], y[eval_idx])
    ece_probe = ece(probe_probs[eval_idx], y[eval_idx])

    # difficulty gradient incl. self-consistency and combo, not just mmp
    diff = {}
    for d in sorted(set(nd.tolist())):
        m = nd == d
        if m.sum() >= 5:
            diff[f"digits_{d}"] = dict(n=int(m.sum()), acc=float(y[m].mean()),
                                        mean_mmp=float(mmp[m].mean()), mean_sc=float(sc[m].mean()),
                                        mean_combo=float(combo_probs[m].mean()))
    cb = np.clip(nc, 0, 3)
    for c in sorted(set(cb.tolist())):
        m = cb == c
        if m.sum() >= 5:
            diff[f"carries_{c}"] = dict(n=int(m.sum()), acc=float(y[m].mean()),
                                         mean_mmp=float(mmp[m].mean()), mean_sc=float(sc[m].mean()),
                                         mean_combo=float(combo_probs[m].mean()))

    n_incorrect = int((1 - y).sum())
    return dict(seed=seed, acc=float(y.mean()), n_incorrect=n_incorrect,
                auroc_mmp=auroc_mmp, auroc_nme=auroc_nme, auroc_sc=auroc_sc,
                auroc_combo=auroc_combo, auroc_probe=auroc_probe,
                probe_cv_default=probe_cv_default, probe_cv_strongreg=probe_cv_strongreg,
                ece_sc_raw=ece_sc_raw, ece_sc_iso=ece_sc_iso, ece_combo=ece_combo, ece_probe=ece_probe,
                difficulty=diff)

if __name__ == "__main__":
    results = []
    for seed in [3, 4]:
        ts = time.time()
        r = run_seed_full(seed)
        print(seed, "time", time.time() - ts)
        print(r)
        results.append(r)
    with open("extra_seeds_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("TOTAL TIME", time.time() - t0)
