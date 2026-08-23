"""
RAPS hyperparameter robustness check, added per round-3 review (all four reviewers
independently flagged that the RAPS calibration-effect SIGN FLIP in raps_experiment.py
was reported at a single untuned (k_reg=1, lambda=0.02) setting with no sensitivity
check). This script:
  (a) sweeps a small (k_reg, lambda) grid at default settings (sigma=1.8, noise=0.12)
      to check whether the sign of the calibration effect (temp-scaled - raw) is
      stable, using 5 seeds per cell (reduced from 10 to fit the time budget --
      flagged as a limitation);
  (b) checks whether the sign holds at the low-overlap (sigma=1.2) and high-noise
      (noise=0.24) regimes from the sec 3.3 sweep, at the paper's original
      (k_reg=1, lambda=0.02), 5 seeds each.
"""
import time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

t_start = time.time()
N_CLASSES = 6
DIM = 20
N_TRAIN, N_CAL, N_TEST = 4000, 1500, 1500
N_SEEDS = 5

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

def make_data(n, class_means, noise, sigma, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, N_CLASSES, size=n)
    X = class_means[y] + rng.randn(n, DIM) * sigma
    if noise > 0:
        flip = rng.rand(n) < noise
        y_noisy = y.copy()
        y_noisy[flip] = rng.randint(0, N_CLASSES, size=flip.sum())
        return X.astype(np.float32), y.astype(np.int64), y_noisy.astype(np.int64)
    return X.astype(np.float32), y.astype(np.int64), y.astype(np.int64)

def train_model(Xtr, ytr, seed, epochs=80):
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def softmax_probs(model, X, T=1.0):
    return F.softmax(model(X) / T, dim=1).numpy()

def fit_temperature(model, X, y, n_iter=200):
    logT = torch.zeros(1, requires_grad=True)
    with torch.no_grad():
        logits = model(X)
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

def raps_eval(prob_fn, Xcal, ycal, Xte, yte, alpha=0.10, k_reg=1, lam=0.02):
    p_cal = prob_fn(Xcal)
    order = np.argsort(-p_cal, axis=1)
    sorted_probs = np.take_along_axis(p_cal, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank = np.array([np.where(order[i] == ycal[i])[0][0] for i in range(len(ycal))])
    reg = lam * np.maximum(rank + 1 - k_reg, 0)
    scores_cal = cumsum[np.arange(len(ycal)), rank] + reg
    n = len(ycal)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")

    p_test = prob_fn(Xte)
    order_t = np.argsort(-p_test, axis=1)
    sorted_t = np.take_along_axis(p_test, order_t, axis=1)
    cumsum_t = np.cumsum(sorted_t, axis=1)
    ranks_t = np.arange(N_CLASSES)[None, :]
    reg_t = lam * np.maximum(ranks_t + 1 - k_reg, 0)
    scores_t = cumsum_t + reg_t
    include = scores_t <= qhat
    include[:, 0] = True
    pred_sets = np.zeros_like(include)
    np.put_along_axis(pred_sets, order_t, include, axis=1)
    covered = pred_sets[np.arange(len(yte)), yte]
    return dict(coverage=float(covered.mean()), avg_set_size=float(pred_sets.sum(axis=1).mean()))

def run_condition(k_reg, lam, sigma, noise, n_seeds=N_SEEDS):
    raw_sizes, ts_sizes = [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        rng0 = np.random.RandomState(42 + seed)
        class_means = rng0.randn(N_CLASSES, DIM) * 1.6
        X_train, _, y_train = make_data(N_TRAIN, class_means, noise, sigma, seed=1000 + seed)
        X_cal, y_cal, _ = make_data(N_CAL, class_means, 0.0, sigma, seed=2000 + seed)
        X_test, y_test, _ = make_data(N_TEST, class_means, 0.0, sigma, seed=3000 + seed)
        Xtr = torch.tensor(X_train); ytr = torch.tensor(y_train)
        Xcal = torch.tensor(X_cal); Xte = torch.tensor(X_test)
        model = train_model(Xtr, ytr, seed=seed)
        T = fit_temperature(model, Xcal, y_cal)
        raw_fn = lambda X: softmax_probs(model, X, T=1.0)
        ts_fn = lambda X: softmax_probs(model, X, T=T)
        r_raw = raps_eval(raw_fn, Xcal, y_cal, Xte, y_test, k_reg=k_reg, lam=lam)
        r_ts = raps_eval(ts_fn, Xcal, y_cal, Xte, y_test, k_reg=k_reg, lam=lam)
        raw_sizes.append(r_raw["avg_set_size"]); ts_sizes.append(r_ts["avg_set_size"])
    raw_arr, ts_arr = np.array(raw_sizes), np.array(ts_sizes)
    t_calib, p_calib = stats.ttest_rel(ts_arr, raw_arr)
    return dict(
        raw_mean=float(raw_arr.mean()), ts_mean=float(ts_arr.mean()),
        calib_effect=float(ts_arr.mean() - raw_arr.mean()),
        t=float(t_calib), p=float(p_calib),
    )

# (a) grid over (k_reg, lambda) at default sigma=1.8, noise=0.12
grid_results = []
for k_reg in [1, 2, 3]:
    for lam in [0.0, 0.01, 0.02, 0.05, 0.1]:
        r = run_condition(k_reg, lam, sigma=1.8, noise=0.12)
        r.update(k_reg=k_reg, lam=lam)
        grid_results.append(r)
        print(f"k_reg={k_reg} lam={lam}: calib_effect={r['calib_effect']:.3f} p={r['p']:.2e}")

# (b) sign check at low-overlap and high-noise regimes, default (k_reg=1, lam=0.02)
regime_results = {}
regime_results["sigma_1.2"] = run_condition(1, 0.02, sigma=1.2, noise=0.12)
regime_results["noise_0.24"] = run_condition(1, 0.02, sigma=1.8, noise=0.24)
regime_results["default_sigma1.8_noise0.12"] = run_condition(1, 0.02, sigma=1.8, noise=0.12)
for k, v in regime_results.items():
    print(f"{k}: calib_effect={v['calib_effect']:.3f} p={v['p']:.2e}")

n_negative = sum(1 for r in grid_results if r["calib_effect"] < 0)
n_positive = sum(1 for r in grid_results if r["calib_effect"] > 0)

out = dict(
    grid=grid_results,
    regimes=regime_results,
    n_grid_cells=len(grid_results),
    n_cells_negative_effect=n_negative,
    n_cells_positive_effect=n_positive,
    elapsed_seconds=time.time() - t_start,
)
print(json.dumps(out, indent=2))
with open("results_raps_grid.json", "w") as f:
    json.dump(out, f, indent=2)
