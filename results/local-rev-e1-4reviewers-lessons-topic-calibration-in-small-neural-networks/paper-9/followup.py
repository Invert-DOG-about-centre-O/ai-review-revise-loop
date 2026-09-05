"""
Follow-up checks for round-2 review:
1. Brier score alongside ECE (second calibration metric, bin-free) for existing digits results
   -- requires rerunning forward pass, but we can reconstruct from experiment.py's model training
   since raw probs weren't saved. We rerun training (identical seeds/hyperparams) and additionally
   compute Brier score, and also recompute ECE with 10 bins and 20 bins to check bin sensitivity.
2. Parameter count vs training-set size (interpolation threshold framing / double descent).
3. Optimization-budget confound: rerun width=4,8 (and 16 as control) with 4x epochs (600 vs 150)
   to see if the peak is a width effect or a width x fixed-budget interaction.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, ttest_rel
import json, time

t_start = time.time()

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, d_in, width, d_out, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), size=(d_in, width))
        self.b1 = np.zeros(width)
        self.W2 = rng.normal(0, np.sqrt(2.0 / max(width,1)), size=(width, d_out))
        self.b2 = np.zeros(d_out)
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2
    def logits(self, X):
        return self.forward(X)[2]
    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

def train_mlp(X, y, n_classes, width, seed, epochs=150, lr=0.03, l2=1e-4, batch_size=64):
    n, d_in = X.shape
    model = MLP(d_in, width, n_classes, seed)
    Y1h = np.eye(n_classes)[y]
    rng = np.random.default_rng(seed + 10000)
    mW1 = np.zeros_like(model.W1); vW1 = np.zeros_like(model.W1)
    mb1 = np.zeros_like(model.b1); vb1 = np.zeros_like(model.b1)
    mW2 = np.zeros_like(model.W2); vW2 = np.zeros_like(model.W2)
    mb2 = np.zeros_like(model.b2); vb2 = np.zeros_like(model.b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    idx_all = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch = idx_all[start:start+batch_size]
            Xb, Yb = X[batch], Y1h[batch]
            z1, a1, z2 = model.forward(Xb)
            p = softmax(z2)
            b = Xb.shape[0]
            dz2 = (p - Yb) / b
            dW2 = a1.T @ dz2 + l2 * model.W2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ model.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = Xb.T @ dz1 + l2 * model.W1
            db1 = dz1.sum(axis=0)
            t += 1
            for (param, grad, m, v) in [
                (model.W1, dW1, mW1, vW1), (model.b1, db1, mb1, vb1),
                (model.W2, dW2, mW2, vW2), (model.b2, db2, mb2, vb2)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * (grad ** 2)
                mhat = m / (1 - beta1 ** t)
                vhat = v / (1 - beta2 ** t)
                param -= lr * mhat / (np.sqrt(vhat) + eps)
    return model

def ece_score(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return ece

def brier_score(probs, labels, n_classes):
    Y1h = np.eye(n_classes)[labels]
    return ((probs - Y1h) ** 2).sum(axis=1).mean()

X, y = load_digits(return_X_y=True)
X = X / 16.0
n_classes = 10
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=y_trainval)
n_train = len(X_train)

print("=== Part 2: parameter count vs training set size (interpolation threshold) ===")
d_in = X.shape[1]
for width in [2,4,8,16,32,64,128,256,512]:
    n_params = d_in*width + width + width*n_classes + n_classes
    print(f"width={width:>4} n_params={n_params:>7} train_n={n_train} ratio(params/n)={n_params/n_train:.3f}")

print()
print("=== Part 1: Brier score + bin-sensitivity (widths 2,4,8,16,512), 10 seeds ===", flush=True)
brier_results = []
for width in [2,4,8,16,512]:
    for seed in range(10):
        model = train_mlp(X_train, y_train, n_classes, width, seed, epochs=150, lr=0.03, l2=1e-4, batch_size=64)
        test_probs = softmax(model.logits(X_test))
        test_acc = (test_probs.argmax(1) == y_test).mean()
        brier = brier_score(test_probs, y_test, n_classes)
        ece15 = ece_score(test_probs, y_test, n_bins=15)
        ece10 = ece_score(test_probs, y_test, n_bins=10)
        ece20 = ece_score(test_probs, y_test, n_bins=20)
        brier_results.append(dict(width=width, seed=seed, test_acc=float(test_acc),
                                   brier=float(brier), ece15=float(ece15), ece10=float(ece10), ece20=float(ece20)))
    print(f"width={width} done, elapsed={time.time()-t_start:.1f}s", flush=True)

with open("followup_brier.json", "w") as f:
    json.dump(brier_results, f, indent=2)

print()
for w in [2,4,8,16,512]:
    rows = [r for r in brier_results if r["width"]==w]
    b = np.array([r["brier"] for r in rows])
    e15 = np.array([r["ece15"] for r in rows])
    e10 = np.array([r["ece10"] for r in rows])
    e20 = np.array([r["ece20"] for r in rows])
    print(f"width={w:>4} brier={b.mean():.4f}+-{b.std():.4f} ece10={e10.mean():.4f} ece15={e15.mean():.4f} ece20={e20.mean():.4f}")

b48 = np.array([r["brier"] for r in brier_results if r["width"] in (4,8)])
bplateau = np.array([r["brier"] for r in brier_results if r["width"] in (16,512)])
t, p = ttest_ind(b48, bplateau, equal_var=False)
print(f"\nBrier: width 4+8 (n={len(b48)}) vs plateau 16+512 (n={len(bplateau)}): t={t:.3f} p={p:.3e}")

print()
print("=== Part 3: optimization-budget confound -- 4x epochs (600) at width=4,8,16 ===", flush=True)
budget_results = []
for width in [4, 8, 16]:
    for seed in range(10):
        model = train_mlp(X_train, y_train, n_classes, width, seed, epochs=600, lr=0.03, l2=1e-4, batch_size=64)
        test_probs = softmax(model.logits(X_test))
        test_acc = (test_probs.argmax(1) == y_test).mean()
        ece = ece_score(test_probs, y_test, n_bins=15)
        budget_results.append(dict(width=width, seed=seed, test_acc=float(test_acc), ece=float(ece)))
    print(f"width={width} (600 epochs) done, elapsed={time.time()-t_start:.1f}s", flush=True)

with open("followup_budget.json", "w") as f:
    json.dump(budget_results, f, indent=2)

print()
for w in [4,8,16]:
    rows = [r for r in budget_results if r["width"]==w]
    acc = np.array([r["test_acc"] for r in rows])
    ece = np.array([r["ece"] for r in rows])
    print(f"width={w:>4} (600ep) acc={acc.mean():.3f}+-{acc.std():.3f} ece={ece.mean():.4f}+-{ece.std():.4f}")

e48_budget = np.array([r["ece"] for r in budget_results if r["width"] in (4,8)])
e16_budget = np.array([r["ece"] for r in budget_results if r["width"]==16])
t2, p2 = ttest_ind(e48_budget, e16_budget, equal_var=False)
print(f"\n600-epoch ECE: width 4+8 (n={len(e48_budget)}) vs width 16 (n={len(e16_budget)}): t={t2:.3f} p={p2:.3e}")

print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
