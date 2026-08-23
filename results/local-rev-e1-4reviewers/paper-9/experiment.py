"""
Probabilistic methods in (surrogate) LLM classifiers: comparing raw softmax,
temperature scaling, deep ensembles (self-consistency surrogate), and their
combination, under split conformal prediction.

No internet access is available in this sandbox (HF Hub downloads fail with
SSL errors), so we cannot pull a pretrained LLM. Instead we construct a
controlled synthetic surrogate: a small MLP classifier trained on a noisy,
overlapping-class synthetic dataset, tuned to reproduce the well-documented
LLM failure mode of *overconfident* softmax outputs (Guo et al. 2017 showed
modern deep nets, and later work showed LLMs, are systematically
overconfident). This lets us study, under ground truth we control, whether
temperature scaling and ensembling (a proxy for self-consistency sampling)
actually fix calibration, and whether split conformal prediction gives valid
coverage regardless.
"""
import time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset: 6-class Gaussian-mixture-in-feature-space task with
#    label noise, mimicking ambiguous natural-language classification (e.g.
#    intent/topic classification) where an LLM's next-token distribution
#    over answer options is genuinely uncertain for some inputs.
# ---------------------------------------------------------------------------
N_CLASSES = 6
DIM = 20
N_TRAIN, N_CAL, N_TEST = 4000, 1500, 1500
LABEL_NOISE = 0.12  # fraction of training labels flipped -> induces overconfidence when memorized

def make_data(n, class_means, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, N_CLASSES, size=n)
    X = class_means[y] + rng.randn(n, DIM) * 1.8
    if noise > 0:
        flip = rng.rand(n) < noise
        y_noisy = y.copy()
        y_noisy[flip] = rng.randint(0, N_CLASSES, size=flip.sum())
        return X.astype(np.float32), y.astype(np.int64), y_noisy.astype(np.int64)
    return X.astype(np.float32), y.astype(np.int64), y.astype(np.int64)

rng0 = np.random.RandomState(42)
class_means = rng0.randn(N_CLASSES, DIM) * 1.6  # overlapping clusters -> real ambiguity

X_train, y_train_clean, y_train = make_data(N_TRAIN, class_means, noise=LABEL_NOISE, seed=1)
X_cal, y_cal, _ = make_data(N_CAL, class_means, noise=0.0, seed=2)
X_test, y_test, _ = make_data(N_TEST, class_means, noise=0.0, seed=3)

Xtr = torch.tensor(X_train); ytr = torch.tensor(y_train)
Xcal = torch.tensor(X_cal); ycal = torch.tensor(y_cal)
Xte = torch.tensor(X_test); yte = torch.tensor(y_test)

# ---------------------------------------------------------------------------
# 2. Model: small MLP, trained long enough to memorize noisy labels (this is
#    what produces realistic overconfidence, as in Guo et al. 2017).
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim=DIM, hidden=64, nclass=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, nclass),
        )
    def forward(self, x):
        return self.net(x)

def train_model(seed, epochs=60, bootstrap=False):
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if bootstrap:
        idx = np.random.RandomState(seed).choice(N_TRAIN, N_TRAIN, replace=True)
        xb, yb = Xtr[idx], ytr[idx]
    else:
        xb, yb = Xtr, ytr
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
    return model

print("Training base model...")
base_model = train_model(seed=0, epochs=80, bootstrap=False)

K_ENSEMBLE = 5
print(f"Training ensemble of {K_ENSEMBLE} models (bootstrap + different init, surrogate for self-consistency)...")
ensemble = [train_model(seed=100 + k, epochs=80, bootstrap=True) for k in range(K_ENSEMBLE)]
print(f"Training done at t={time.time()-t_start:.1f}s")

# ---------------------------------------------------------------------------
# 3. Confidence estimators
# ---------------------------------------------------------------------------
@torch.no_grad()
def softmax_probs(model, X, T=1.0):
    logits = model(X) / T
    return F.softmax(logits, dim=1).numpy()

@torch.no_grad()
def ensemble_probs(models, X, T=1.0):
    ps = [F.softmax(m(X) / T, dim=1).numpy() for m in models]
    return np.mean(ps, axis=0)

def fit_temperature(logit_fn, X, y, n_iter=200):
    """Fit scalar temperature by minimizing NLL on calibration data (Guo et al. 2017)."""
    logT = torch.zeros(1, requires_grad=True)
    with torch.no_grad():
        logits = logit_fn(X)
    y_t = torch.tensor(y)
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=n_iter)
    def closure():
        opt.zero_grad()
        T = torch.exp(logT)
        loss = F.cross_entropy(logits / T, y_t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(logT).item())

@torch.no_grad()
def base_logits(X):
    return base_model(X)

@torch.no_grad()
def ensemble_logits(X):
    # average of member logits, used only for fitting a single ensemble temperature
    return torch.mean(torch.stack([m(X) for m in ensemble]), dim=0)

T_base = fit_temperature(base_logits, Xcal, y_cal)
T_ens = fit_temperature(ensemble_logits, Xcal, y_cal)
print(f"Fitted temperatures: base T={T_base:.3f}, ensemble T={T_ens:.3f}")

methods = {
    "raw_softmax":        lambda X: softmax_probs(base_model, X, T=1.0),
    "temp_scaled":         lambda X: softmax_probs(base_model, X, T=T_base),
    "ensemble":             lambda X: ensemble_probs(ensemble, X, T=1.0),
    "ensemble_temp_scaled": lambda X: ensemble_probs(ensemble, X, T=T_ens),
}

# ---------------------------------------------------------------------------
# 4. Metrics: accuracy, NLL, Brier score, Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------
def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        e += (mask.sum() / total) * abs(acc_bin - conf_bin)
    return e

def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return -np.log(p).mean()

def brier(probs, y):
    onehot = np.eye(N_CLASSES)[y]
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))

def accuracy(probs, y):
    return (probs.argmax(axis=1) == y).mean()

results = {}
for name, fn in methods.items():
    p_test = fn(Xte)
    results[name] = dict(
        accuracy=float(accuracy(p_test, y_test)),
        nll=float(nll(p_test, y_test)),
        brier=float(brier(p_test, y_test)),
        ece=float(ece(p_test, y_test)),
        mean_confidence=float(p_test.max(axis=1).mean()),
    )
    print(name, results[name])

# ---------------------------------------------------------------------------
# 5. Split conformal prediction (LAC score: s(x,y) = 1 - p_model(y|x)) on top
#    of each confidence method, target coverage 1-alpha = 0.90.
# ---------------------------------------------------------------------------
def conformal_eval(prob_fn, alpha=0.10):
    p_cal = prob_fn(Xcal)
    scores_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(y_cal)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")

    p_test = prob_fn(Xte)
    pred_sets = p_test >= (1 - qhat)
    covered = pred_sets[np.arange(len(y_test)), y_test]
    coverage = covered.mean()
    avg_size = pred_sets.sum(axis=1).mean()
    return dict(qhat=float(qhat), coverage=float(coverage), avg_set_size=float(avg_size))

conformal_results = {}
for name, fn in methods.items():
    conformal_results[name] = conformal_eval(fn, alpha=0.10)
    print("conformal(LAC)", name, conformal_results[name])

# ---------------------------------------------------------------------------
# 5b. APS (Adaptive Prediction Sets, Romano et al. 2020): nonconformity score
#     = cumulative sum of sorted probs up to and including the true class.
#     Designed to avoid the empty/degenerate sets that LAC can produce.
# ---------------------------------------------------------------------------
def aps_score(probs, y):
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])
    return cumsum[np.arange(len(y)), rank]

def aps_eval(prob_fn, alpha=0.10):
    p_cal = prob_fn(Xcal)
    scores_cal = aps_score(p_cal, y_cal)
    n = len(y_cal)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")

    p_test = prob_fn(Xte)
    order = np.argsort(-p_test, axis=1)
    sorted_probs = np.take_along_axis(p_test, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    include = cumsum <= qhat
    # always include the top class (standard APS convention)
    include[:, 0] = True
    pred_sets = np.zeros_like(include)
    np.put_along_axis(pred_sets, order, include, axis=1)

    covered = pred_sets[np.arange(len(y_test)), y_test]
    coverage = covered.mean()
    avg_size = pred_sets.sum(axis=1).mean()
    empty_frac = float((pred_sets.sum(axis=1) == 0).mean())
    return dict(qhat=float(qhat), coverage=float(coverage), avg_set_size=float(avg_size), empty_frac=empty_frac)

aps_results = {}
for name, fn in methods.items():
    aps_results[name] = aps_eval(fn, alpha=0.10)
    print("conformal(APS)", name, aps_results[name])

lac_empty_frac = {}
for name, fn in methods.items():
    p_cal = fn(Xcal)
    scores_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(y_cal)
    q_level = min(1.0, math.ceil((n + 1) * 0.90) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")
    p_test = fn(Xte)
    pred_sets = p_test >= (1 - qhat)
    lac_empty_frac[name] = float((pred_sets.sum(axis=1) == 0).mean())
    conformal_results[name]["empty_frac"] = lac_empty_frac[name]

elapsed = time.time() - t_start
print(f"Total experiment time: {elapsed:.1f}s")

out = dict(
    label_noise=LABEL_NOISE,
    n_train=N_TRAIN, n_cal=N_CAL, n_test=N_TEST, n_classes=N_CLASSES,
    T_base=T_base, T_ens=T_ens,
    calibration_metrics=results,
    conformal_results_LAC=conformal_results,
    conformal_results_APS=aps_results,
    elapsed_seconds=elapsed,
)
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Wrote results.json")
