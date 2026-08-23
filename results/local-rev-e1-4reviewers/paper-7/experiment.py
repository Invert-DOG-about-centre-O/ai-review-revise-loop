"""
Cheap uncertainty signals for correctness prediction in a small autoregressive
transformer trained on integer addition.

We train a tiny char-level decoder-only transformer from scratch to compute
a+b=c (addition with carries), then compare several training-free confidence
signals for predicting whether a greedy decode is CORRECT:
  1. mean-max-prob   : mean of the max softmax probability at each generated digit
  2. neg-mean-entropy: negative mean predictive (token) entropy across generated digits
  3. self-consistency: agreement rate of the modal answer across k stochastic samples
  4. logistic combo  : logistic regression combining signals 1-3 (learned calibrator)

We evaluate: AUROC for correctness prediction, Expected Calibration Error (ECE)
of self-consistency as a probability estimate (before/after isotonic
recalibration), and a risk-coverage (selective prediction) curve for each
signal, on a held-out test set stratified by problem difficulty (# digits).

Everything runs on CPU in a few minutes with a synthetic dataset.
"""
import math
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Data: char-level addition problems "A+B=" -> "C" (reversed digit output,
# a well-known trick that makes addition much easier for small transformers
# since the least-significant digit / carry propagation comes first).
# ---------------------------------------------------------------------------
VOCAB = list("0123456789+=. ") + ["<pad>", "<eos>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
PAD = STOI["<pad>"]
EOS = STOI["<eos>"]

MAX_DIGITS = 4  # numbers up to 9999 -> difficulty knob


def make_example(n_digits):
    a = random.randint(0, 10 ** n_digits - 1)
    b = random.randint(0, 10 ** n_digits - 1)
    c = a + b
    prompt = f"{a}+{b}="
    answer = str(c)[::-1]  # reversed digits (LSB first)
    return prompt, answer, a, b, c


def encode(s):
    return [STOI[ch] for ch in s]


def n_carries(a, b):
    # number of carry operations needed when adding a+b in base 10
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        da, db = a % 10, b % 10
        s = da + db + carry
        carry = 1 if s >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def build_dataset(n, digit_range):
    data = []
    for _ in range(n):
        nd = random.choice(digit_range)
        prompt, answer, a, b, c = make_example(nd)
        data.append(dict(prompt=prompt, answer=answer, a=a, b=b, c=c,
                          n_digits=nd, n_carries=n_carries(a, b)))
    return data


# ---------------------------------------------------------------------------
# Model: tiny decoder-only transformer (causal LM over the concatenated
# "prompt+answer<eos>" sequence), trained with next-token cross-entropy loss
# masked to the answer region.
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.encoder(h, mask=mask)
        h = self.ln(h)
        return self.head(h)


def pad_batch(seqs, pad_id=PAD):
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def make_training_batch(examples):
    input_ids, target_ids, loss_mask = [], [], []
    for ex in examples:
        p_ids = encode(ex["prompt"])
        a_ids = encode(ex["answer"]) + [EOS]
        full = p_ids + a_ids
        inp = full[:-1]
        tgt = full[1:]
        mask = [0] * (len(p_ids) - 1) + [1] * len(a_ids)
        input_ids.append(inp)
        target_ids.append(tgt)
        loss_mask.append(mask)
    x = pad_batch(input_ids)
    y = pad_batch(target_ids)
    m = pad_batch(loss_mask, pad_id=0).float()
    return x, y, m


def train(model, train_data, epochs=6, batch_size=128, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(train_data)
    steps_per_epoch = n // batch_size
    for ep in range(epochs):
        random.shuffle(train_data)
        tot_loss = 0.0
        for step in range(steps_per_epoch):
            batch = train_data[step * batch_size:(step + 1) * batch_size]
            x, y, m = make_training_batch(batch)
            logits = model(x)
            loss_all = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            loss = (loss_all * m.reshape(-1)).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"epoch {ep+1}/{epochs} loss={tot_loss/steps_per_epoch:.4f}")
    return model


# ---------------------------------------------------------------------------
# Inference with uncertainty signals
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_stats(model, prompt, temperature=1.0, sample=False, max_new=6):
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    max_probs, entropies, out_chars = [], [], []
    for _ in range(max_new):
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        if sample:
            nxt = torch.multinomial(probs, 1).item()
        else:
            nxt = torch.argmax(probs).item()
        max_probs.append(probs[nxt].item())
        entropies.append(ent)
        if nxt == EOS:
            break
        out_chars.append(ITOS[nxt])
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    answer_str = "".join(out_chars)
    try:
        pred_val = int(answer_str[::-1]) if answer_str != "" else None
    except ValueError:
        pred_val = None
    return pred_val, float(np.mean(max_probs)) if max_probs else 0.0, float(np.mean(entropies)) if entropies else 0.0


@torch.no_grad()
def evaluate_example(model, ex, k_samples=8, sample_temp=1.0):
    greedy_val, mean_max_prob, mean_entropy = generate_with_stats(model, ex["prompt"], sample=False)
    correct = int(greedy_val == ex["c"])

    samples = []
    for _ in range(k_samples):
        v, _, _ = generate_with_stats(model, ex["prompt"], temperature=sample_temp, sample=True)
        samples.append(v)
    modal_val, modal_count = None, 0
    counts = {}
    for v in samples:
        counts[v] = counts.get(v, 0) + 1
    for v, c in counts.items():
        if c > modal_count:
            modal_val, modal_count = v, c
    self_consistency = modal_count / k_samples

    return dict(correct=correct, mean_max_prob=mean_max_prob, neg_mean_entropy=-mean_entropy,
                self_consistency=self_consistency, n_digits=ex["n_digits"], n_carries=ex["n_carries"],
                greedy_val=greedy_val, true_val=ex["c"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U statistic
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties by average rank
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs); labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        e += (mask.sum() / total) * abs(acc - conf)
    return e


def isotonic_fit_predict(train_x, train_y, test_x):
    # simple pool-adjacent-violators isotonic regression (monotone increasing)
    order = np.argsort(train_x)
    x_sorted = train_x[order]
    y_sorted = train_y[order]
    y_fit = y_sorted.astype(float).copy()
    weights = np.ones_like(y_fit)
    i = 0
    blocks = list(zip(y_fit, weights))
    # PAVA
    stack = []
    for val, w in zip(y_fit, weights):
        stack.append([val, w])
        while len(stack) > 1 and stack[-2][0] > stack[-1][0]:
            v2, w2 = stack.pop()
            v1, w1 = stack.pop()
            newv = (v1 * w1 + v2 * w2) / (w1 + w2)
            stack.append([newv, w1 + w2])
    # expand stack back to per-point values
    fitted = []
    for val, w in stack:
        fitted.extend([val] * int(w))
    fitted = np.array(fitted)
    # predict via step function interpolation on sorted x
    test_pred = np.interp(test_x, x_sorted, fitted, left=fitted[0], right=fitted[-1])
    return test_pred


class LogisticCalibrator:
    def __init__(self, n_features, lr=0.1, epochs=300):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y, dtype=float)
        n = len(y)
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = 1 / (1 + np.exp(-z))
            grad_w = X.T @ (p - y) / n
            grad_b = (p - y).mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-z))


def risk_coverage(scores, correct, n_points=20):
    order = np.argsort(-np.asarray(scores))  # high confidence first
    correct_sorted = np.asarray(correct)[order]
    n = len(correct_sorted)
    coverages, risks = [], []
    for frac in np.linspace(0.05, 1.0, n_points):
        k = max(1, int(round(frac * n)))
        subset = correct_sorted[:k]
        coverages.append(k / n)
        risks.append(1 - subset.mean())
    return coverages, risks


def standardize(train_x, *others):
    mu, sd = train_x.mean(), train_x.std() + 1e-8
    return [(train_x - mu) / sd] + [(o - mu) / sd for o in others]


def main():
    t0 = time.time()
    print("Building datasets...")
    train_data = build_dataset(20000, digit_range=[1, 2, 3, 4])
    val_data = build_dataset(1000, digit_range=[1, 2, 3, 4])
    test_data = build_dataset(2000, digit_range=[1, 2, 3, 4])

    model = TinyTransformer(len(VOCAB), d_model=64, n_heads=4, n_layers=3, max_len=32)
    print("Training model...")
    train(model, train_data, epochs=6, batch_size=128, lr=3e-3)
    print(f"Training done at t={time.time()-t0:.1f}s")

    model.eval()
    print("Evaluating on test set with uncertainty signals (this samples k times per example)...")
    results = []
    for i, ex in enumerate(test_data):
        r = evaluate_example(model, ex, k_samples=8, sample_temp=1.0)
        results.append(r)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test_data)} done, t={time.time()-t0:.1f}s")

    overall_acc = np.mean([r["correct"] for r in results])
    print(f"Overall greedy accuracy: {overall_acc:.4f}")

    labels = np.array([r["correct"] for r in results])
    mmp = np.array([r["mean_max_prob"] for r in results])
    nme = np.array([r["neg_mean_entropy"] for r in results])
    sc = np.array([r["self_consistency"] for r in results])
    n_digits = np.array([r["n_digits"] for r in results])
    n_carries = np.array([r["n_carries"] for r in results])

    # split test set in half: calibration-fit / final-eval (still all "test", but separates fitting from reporting)
    idx = np.arange(len(results))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    half = len(idx) // 2
    fit_idx, eval_idx = idx[:half], idx[half:]

    signals_raw = dict(mean_max_prob=mmp, neg_mean_entropy=nme, self_consistency=sc)
    auroc_results = {name: auroc(sig[eval_idx], labels[eval_idx]) for name, sig in signals_raw.items()}

    # logistic combo, fit on fit_idx, evaluate on eval_idx
    mmp_s, mmp_s_eval = standardize(mmp[fit_idx], mmp[eval_idx])
    nme_s, nme_s_eval = standardize(nme[fit_idx], nme[eval_idx])
    sc_s, sc_s_eval = standardize(sc[fit_idx], sc[eval_idx])
    X_fit = np.stack([mmp_s, nme_s, sc_s], axis=1)
    X_eval = np.stack([mmp_s_eval, nme_s_eval, sc_s_eval], axis=1)
    clf = LogisticCalibrator(n_features=3).fit(X_fit, labels[fit_idx])
    combo_scores_eval = clf.predict_proba(X_eval)
    auroc_results["logistic_combo"] = auroc(combo_scores_eval, labels[eval_idx])

    # ECE of self-consistency as a probability estimate, raw vs isotonic-recalibrated
    ece_raw = ece(sc[eval_idx], labels[eval_idx])
    iso_pred_eval = isotonic_fit_predict(sc[fit_idx], labels[fit_idx].astype(float), sc[eval_idx])
    ece_iso = ece(iso_pred_eval, labels[eval_idx])
    ece_logistic_combo = ece(combo_scores_eval, labels[eval_idx])

    # risk-coverage curves (selective prediction) on eval_idx for each raw signal + combo
    rc_curves = {}
    for name, sig in signals_raw.items():
        cov, risk = risk_coverage(sig[eval_idx], labels[eval_idx])
        rc_curves[name] = dict(coverage=cov, risk=risk)
    cov, risk = risk_coverage(combo_scores_eval, labels[eval_idx])
    rc_curves["logistic_combo"] = dict(coverage=cov, risk=risk)

    # Bootstrap CI (2000 resamples, paired across signals) on the single-seed
    # AUROC point estimates and pairwise gaps, reusing the eval_idx predictions
    # already computed above -- no retraining required.
    boot_rng = np.random.RandomState(0)
    n_eval = len(eval_idx)
    all_scores = dict(mean_max_prob=mmp[eval_idx], neg_mean_entropy=nme[eval_idx],
                       self_consistency=sc[eval_idx], logistic_combo=combo_scores_eval)
    y_eval = labels[eval_idx]
    n_boot = 2000
    boot_aurocs = {name: np.empty(n_boot) for name in all_scores}
    for b in range(n_boot):
        bidx = boot_rng.randint(0, n_eval, n_eval)
        yb = y_eval[bidx]
        for name, s in all_scores.items():
            boot_aurocs[name][b] = auroc(s[bidx], yb)
    boot_ci = {name: dict(mean=float(np.nanmean(v)),
                           ci_lo=float(np.nanpercentile(v, 2.5)),
                           ci_hi=float(np.nanpercentile(v, 97.5)))
               for name, v in boot_aurocs.items()}
    gap_mmp_minus_sc = boot_aurocs["mean_max_prob"] - boot_aurocs["self_consistency"]
    gap_mmp_minus_nme = boot_aurocs["mean_max_prob"] - boot_aurocs["neg_mean_entropy"]
    bootstrap_gaps = dict(
        mmp_minus_sc=dict(mean=float(np.nanmean(gap_mmp_minus_sc)),
                           ci_lo=float(np.nanpercentile(gap_mmp_minus_sc, 2.5)),
                           ci_hi=float(np.nanpercentile(gap_mmp_minus_sc, 97.5)),
                           excludes_zero=bool(np.nanpercentile(gap_mmp_minus_sc, 2.5) > 0)),
        mmp_minus_nme=dict(mean=float(np.nanmean(gap_mmp_minus_nme)),
                            ci_lo=float(np.nanpercentile(gap_mmp_minus_nme, 2.5)),
                            ci_hi=float(np.nanpercentile(gap_mmp_minus_nme, 97.5)),
                            excludes_zero=bool(np.nanpercentile(gap_mmp_minus_nme, 2.5) > 0)),
    )
    print("Bootstrap CIs:", json.dumps(boot_ci, indent=2))
    print("Bootstrap gaps:", json.dumps(bootstrap_gaps, indent=2))

    # AUROC of mean-max-prob broken down by problem difficulty (n_digits bucket),
    # reusing the eval_idx predictions -- answers whether discrimination is
    # uniform across the difficulty gradient or concentrated at some buckets.
    nd_eval = n_digits[eval_idx]
    auroc_by_digits = {}
    for d in sorted(set(nd_eval.tolist())):
        m = nd_eval == d
        if m.sum() >= 20 and len(set(y_eval[m].tolist())) == 2:
            auroc_by_digits[int(d)] = dict(n=int(m.sum()), auroc_mean_max_prob=float(auroc(mmp[eval_idx][m], y_eval[m])))
    print("AUROC by n_digits:", json.dumps(auroc_by_digits, indent=2))

    # accuracy by difficulty (n_digits, n_carries) -- sanity / discussion
    acc_by_digits = {}
    for d in sorted(set(n_digits.tolist())):
        mask = n_digits == d
        acc_by_digits[int(d)] = float(labels[mask].mean())
    acc_by_carries = {}
    for c in sorted(set(n_carries.tolist())):
        mask = n_carries == c
        if mask.sum() >= 5:
            acc_by_carries[int(c)] = dict(acc=float(labels[mask].mean()), n=int(mask.sum()))

    summary = dict(
        overall_accuracy=float(overall_acc),
        n_test=len(results),
        auroc=auroc_results,
        ece_self_consistency_raw=ece_raw,
        ece_self_consistency_isotonic=ece_iso,
        ece_logistic_combo=ece_logistic_combo,
        risk_coverage=rc_curves,
        accuracy_by_n_digits=acc_by_digits,
        accuracy_by_n_carries=acc_by_carries,
        logistic_weights=dict(w_mean_max_prob=float(clf.w[0]), w_neg_mean_entropy=float(clf.w[1]),
                               w_self_consistency=float(clf.w[2]), b=float(clf.b)),
        bootstrap_auroc_ci=boot_ci,
        bootstrap_auroc_gaps=bootstrap_gaps,
        auroc_by_n_digits=auroc_by_digits,
        wall_clock_sec=time.time() - t0,
    )

    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("risk_coverage",)}, indent=2))
    print(f"Total wall clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
