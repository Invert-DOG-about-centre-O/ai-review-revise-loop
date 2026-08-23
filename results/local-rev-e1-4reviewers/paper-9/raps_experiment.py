"""
RAPS (Regularized Adaptive Prediction Sets, Angelopoulos et al. 2020) comparison,
added per reviewer request (3 of 4 round-2 reviewers flagged RAPS as the standard
mitigation for APS's large sets, absent from v2). Reuses the same 10-seed default
setting (sigma=1.8, noise=0.12) as multiseed_experiment.py, raw_softmax and
temp_scaled only (the two methods that bracket the calibration effect on APS).
Fixed regularization (k_reg=1, lambda=0.02), not tuned per Angelopoulos et al.'s
two-stage procedure -- flagged as a limitation, not a full RAPS replication.
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
LABEL_NOISE = 0.12
N_SEEDS = 10
K_REG, LAM = 1, 0.02

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

def make_data(n, class_means, noise, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, N_CLASSES, size=n)
    X = class_means[y] + rng.randn(n, DIM) * 1.8
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

def raps_eval(prob_fn, Xcal, ycal, Xte, yte, alpha=0.10, k_reg=K_REG, lam=LAM):
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

raw_sizes, ts_sizes = [], []
for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    rng0 = np.random.RandomState(42 + seed)
    class_means = rng0.randn(N_CLASSES, DIM) * 1.6
    X_train, _, y_train = make_data(N_TRAIN, class_means, LABEL_NOISE, seed=1000 + seed)
    X_cal, y_cal, _ = make_data(N_CAL, class_means, 0.0, seed=2000 + seed)
    X_test, y_test, _ = make_data(N_TEST, class_means, 0.0, seed=3000 + seed)
    Xtr = torch.tensor(X_train); ytr = torch.tensor(y_train)
    Xcal = torch.tensor(X_cal); Xte = torch.tensor(X_test)
    model = train_model(Xtr, ytr, seed=seed)
    T = fit_temperature(model, Xcal, y_cal)
    raw_fn = lambda X: softmax_probs(model, X, T=1.0)
    ts_fn = lambda X: softmax_probs(model, X, T=T)
    r_raw = raps_eval(raw_fn, Xcal, y_cal, Xte, y_test)
    r_ts = raps_eval(ts_fn, Xcal, y_cal, Xte, y_test)
    raw_sizes.append(r_raw["avg_set_size"]); ts_sizes.append(r_ts["avg_set_size"])
    print(f"seed {seed}: raps_raw={r_raw}, raps_ts={r_ts}")

raw_arr, ts_arr = np.array(raw_sizes), np.array(ts_sizes)
t_calib, p_calib = stats.ttest_rel(ts_arr, raw_arr)

# LAC set size (raw_softmax) at default settings for the RAPS-vs-LAC comparison,
# reusing results_multiseed.json so we don't retrain a second time.
with open("results_multiseed.json") as f:
    mm = json.load(f)
lac_raw_mean = mm["conformal_LAC"]["raw_softmax"]["avg_set_size"]["mean"]
aps_raw_mean = mm["conformal_APS"]["raw_softmax"]["avg_set_size"]["mean"]

out = dict(
    k_reg=K_REG, lam=LAM, n_seeds=N_SEEDS,
    raps_raw_mean=float(raw_arr.mean()), raps_raw_std=float(raw_arr.std()),
    raps_ts_mean=float(ts_arr.mean()), raps_ts_std=float(ts_arr.std()),
    raps_calib_effect=float(abs(ts_arr.mean() - raw_arr.mean())),
    raps_calib_ttest=dict(t=float(t_calib), p=float(p_calib)),
    lac_raw_mean_ref=lac_raw_mean, aps_raw_mean_ref=aps_raw_mean,
    raps_vs_lac_gap=float(raw_arr.mean() - lac_raw_mean),
    raps_vs_aps_gap=float(raw_arr.mean() - aps_raw_mean),
    elapsed_seconds=time.time() - t_start,
)
print(json.dumps(out, indent=2))
with open("results_raps.json", "w") as f:
    json.dump(out, f, indent=2)
