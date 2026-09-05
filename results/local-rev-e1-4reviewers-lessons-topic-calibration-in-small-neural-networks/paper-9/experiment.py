"""
Calibration vs. width sweep for small MLPs on sklearn digits.
Pure-numpy MLP (no torch dependency) so it runs fast on CPU.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, spearmanr
import json, time

t_start = time.time()

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, d_in, width, d_out, seed):
        rng = np.random.default_rng(seed)
        # He init
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

def train_mlp(X, y, n_classes, width, seed, epochs=300, lr=0.05, l2=1e-4, batch_size=64):
    n, d_in = X.shape
    model = MLP(d_in, width, n_classes, seed)
    Y1h = np.eye(n_classes)[y]
    rng = np.random.default_rng(seed + 10000)
    # Adam state
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

def nll_score(probs, labels):
    eps = 1e-12
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return -np.log(p).mean()

def fit_temperature(logits, labels, n_classes, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 5.0, 200)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        probs = softmax(logits / T)
        nll = nll_score(probs, labels)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T

def run_all():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # scale pixel values to [0,1]
    n_classes = 10

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=y_trainval)

    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    n_seeds = 10

    results = []
    for width in widths:
        for seed in range(n_seeds):
            model = train_mlp(X_train, y_train, n_classes, width, seed,
                               epochs=150, lr=0.03, l2=1e-4, batch_size=64)

            train_logits = model.logits(X_train)
            train_probs = softmax(train_logits)
            train_acc = (train_probs.argmax(1) == y_train).mean()

            test_logits = model.logits(X_test)
            test_probs = softmax(test_logits)
            test_acc = (test_probs.argmax(1) == y_test).mean()
            test_ece = ece_score(test_probs, y_test)
            test_nll = nll_score(test_probs, y_test)

            val_logits = model.logits(X_val)
            T = fit_temperature(val_logits, y_val, n_classes)

            test_probs_ts = softmax(test_logits / T)
            test_ece_ts = ece_score(test_probs_ts, y_test)
            test_nll_ts = nll_score(test_probs_ts, y_test)

            avg_conf = test_probs.max(axis=1).mean()

            results.append(dict(
                width=width, seed=seed,
                train_acc=float(train_acc), test_acc=float(test_acc),
                test_ece=float(test_ece), test_nll=float(test_nll),
                temperature=float(T),
                test_ece_ts=float(test_ece_ts), test_nll_ts=float(test_nll_ts),
                avg_confidence=float(avg_conf),
                train_test_acc_gap=float(train_acc - test_acc),
            ))
        elapsed = time.time() - t_start
        print(f"width={width} done, elapsed={elapsed:.1f}s")

    with open("results_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    results = run_all()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
