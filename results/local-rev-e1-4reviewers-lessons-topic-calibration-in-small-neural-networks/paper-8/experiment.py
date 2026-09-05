"""
Calibration in small neural networks: capacity-dependent under/over-confidence.

Synthetic Gaussian-discriminant classification task with an analytically known
Bayes posterior. We sweep MLP hidden width (model capacity), train each with
plain cross-entropy, and measure:
  - test accuracy / NLL
  - empirical (binned) ECE
  - "Bayes-ECE": mean |predicted confidence - true Bayes posterior confidence|
  - optimal post-hoc temperature T* (fit on a held-out val set)

Everything is run on CPU with a synthetic dataset so it finishes in minutes.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Synthetic data: 3-class Gaussian discriminant analysis (QDA) with diagonal,
# per-class covariance so the Bayes-optimal boundary is quadratic (nontrivial
# for small nets) while the true posterior is still analytically tractable.
# ---------------------------------------------------------------------------
D = 8
NUM_CLASSES = 3
RNG_SEED = 0

def make_gda_params(rng):
    means = rng.normal(scale=2.2, size=(NUM_CLASSES, D))
    variances = rng.uniform(0.6, 2.5, size=(NUM_CLASSES, D))
    priors = np.array([1.0, 1.0, 1.0]) / 3.0
    return means, variances, priors

def sample_gda(n, means, variances, priors, rng):
    y = rng.choice(NUM_CLASSES, size=n, p=priors)
    x = rng.normal(size=(n, D)) * np.sqrt(variances[y]) + means[y]
    return x.astype(np.float32), y.astype(np.int64)

def bayes_posterior(x, means, variances, priors):
    # log N(x; mu_k, diag(var_k)) for each class k
    logp = np.zeros((x.shape[0], NUM_CLASSES))
    for k in range(NUM_CLASSES):
        diff2 = (x - means[k]) ** 2 / variances[k]
        logp[:, k] = -0.5 * diff2.sum(axis=1) - 0.5 * np.log(variances[k]).sum()
    logp += np.log(priors)[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    return p

rng = np.random.default_rng(RNG_SEED)
means, variances, priors = make_gda_params(rng)

N_TRAIN, N_VAL, N_TEST = 2000, 800, 2000
x_train, y_train = sample_gda(N_TRAIN, means, variances, priors, rng)
x_val, y_val = sample_gda(N_VAL, means, variances, priors, rng)
x_test, y_test = sample_gda(N_TEST, means, variances, priors, rng)

bayes_post_test = bayes_posterior(x_test, means, variances, priors)
bayes_acc = (bayes_post_test.argmax(axis=1) == y_test).mean()

xt_train = torch.tensor(x_train)
yt_train = torch.tensor(y_train)
xt_val = torch.tensor(x_val)
yt_val = torch.tensor(y_val)
xt_test = torch.tensor(x_test)
yt_test = torch.tensor(y_test)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, width, depth=2, label_smoothing_input_dim=D):
        super().__init__()
        layers = []
        in_dim = label_smoothing_input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        layers += [nn.Linear(in_dim, NUM_CLASSES)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(width, label_smoothing=0.0, epochs=80, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    model = MLP(width)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    n = xt_train.shape[0]
    batch_size = 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(xt_train[idx])
            loss = loss_fn(logits, yt_train[idx])
            loss.backward()
            opt.step()
    return model

# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def ece(confidences, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece_val += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece_val

def fit_temperature(logits, labels, lo=0.05, hi=10.0, steps=200):
    """Grid + refine search for temperature minimizing NLL (avoids autograd fuss)."""
    logits = logits.detach()
    labels = labels.detach()
    grid = np.geomspace(lo, hi, steps)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        scaled = logits / T
        nll = F.cross_entropy(scaled, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T), best_nll

def evaluate(model, label="model"):
    with torch.no_grad():
        logits_test = model(xt_test)
        probs_test = F.softmax(logits_test, dim=1).numpy()
        preds_test = probs_test.argmax(axis=1)
        conf_test = probs_test.max(axis=1)
        correct_test = (preds_test == y_test).astype(np.float32)

        logits_val = model(xt_val)

    acc = correct_test.mean()
    nll = F.cross_entropy(logits_test, yt_test).item()
    ece_val = ece(conf_test, correct_test)

    # Bayes-ECE: compare predicted confidence to the *true* posterior prob of
    # the predicted class (ground truth available only because data is synthetic)
    bayes_conf_of_pred = bayes_post_test[np.arange(len(preds_test)), preds_test]
    bayes_ece_val = np.mean(np.abs(conf_test - bayes_conf_of_pred))

    T_star, _ = fit_temperature(logits_val, yt_val)

    # post-temperature-scaling metrics on test set
    with torch.no_grad():
        probs_test_ts = F.softmax(logits_test / T_star, dim=1).numpy()
    conf_test_ts = probs_test_ts.max(axis=1)
    ece_ts = ece(conf_test_ts, correct_test)

    return dict(
        label=label,
        acc=float(acc),
        nll=float(nll),
        ece=float(ece_val),
        bayes_ece=float(bayes_ece_val),
        T_star=float(T_star),
        ece_after_ts=float(ece_ts),
        mean_confidence=float(conf_test.mean()),
    )

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    widths = [2, 4, 16, 64, 256]
    seeds = [0, 1]

    results = []
    for width in widths:
        for seed in seeds:
            model = train_model(width, label_smoothing=0.0, seed=seed)
            r = evaluate(model, label=f"w{width}_s{seed}")
            r["width"] = width
            r["seed"] = seed
            r["label_smoothing"] = 0.0
            results.append(r)
            print(f"[{time.time()-t0:6.1f}s] width={width:4d} seed={seed} "
                  f"acc={r['acc']:.3f} ece={r['ece']:.4f} bayes_ece={r['bayes_ece']:.4f} "
                  f"T*={r['T_star']:.3f} ece_ts={r['ece_after_ts']:.4f}")

    # label smoothing ablation on smallest & largest width
    for width in [2, 256]:
        for seed in seeds:
            model = train_model(width, label_smoothing=0.1, seed=seed)
            r = evaluate(model, label=f"w{width}_s{seed}_ls0.1")
            r["width"] = width
            r["seed"] = seed
            r["label_smoothing"] = 0.1
            results.append(r)
            print(f"[{time.time()-t0:6.1f}s] LS width={width:4d} seed={seed} "
                  f"acc={r['acc']:.3f} ece={r['ece']:.4f} bayes_ece={r['bayes_ece']:.4f} "
                  f"T*={r['T_star']:.3f} ece_ts={r['ece_after_ts']:.4f}")

    out = dict(
        bayes_acc=float(bayes_acc),
        d=D,
        num_classes=NUM_CLASSES,
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
        widths=widths, seeds=seeds,
        results=results,
        total_time_s=time.time() - t0,
    )
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s. Bayes accuracy: {bayes_acc:.3f}")
