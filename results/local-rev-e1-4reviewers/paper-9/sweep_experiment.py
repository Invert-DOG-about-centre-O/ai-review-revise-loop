"""
Sensitivity sweep: does the 'LAC vs APS set-size gap dominates calibration
effects' conclusion hold as we vary label noise and class overlap (sigma)?
10 seeds per config (bumped from 3 in the previous revision, per reviewer
concern that 3 seeds is too thin to trust the sigma=1.2 reversal), with
per-seed effect tracking so we can report std and a paired significance test.
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
N_SEEDS = 10

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

def make_data(n, class_means, sigma, noise, seed):
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

def conformal_eval_lac(prob_fn, Xcal, ycal, Xte, yte, alpha=0.10):
    p_cal = prob_fn(Xcal)
    scores_cal = 1.0 - p_cal[np.arange(len(ycal)), ycal]
    n = len(ycal)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")
    p_test = prob_fn(Xte)
    pred_sets = p_test >= (1 - qhat)
    return float(pred_sets.sum(axis=1).mean())

def aps_score(probs, y):
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])
    return cumsum[np.arange(len(y)), rank]

def conformal_eval_aps(prob_fn, Xcal, ycal, Xte, yte, alpha=0.10):
    p_cal = prob_fn(Xcal)
    scores_cal = aps_score(p_cal, ycal)
    n = len(ycal)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")
    p_test = prob_fn(Xte)
    order = np.argsort(-p_test, axis=1)
    sorted_probs = np.take_along_axis(p_test, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    include = cumsum <= qhat
    include[:, 0] = True
    pred_sets = np.zeros_like(include)
    np.put_along_axis(pred_sets, order, include, axis=1)
    return float(pred_sets.sum(axis=1).mean())

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

def run_config(sigma, noise, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng0 = np.random.RandomState(42 + seed)
    class_means = rng0.randn(N_CLASSES, DIM) * 1.6
    X_train, _, y_train = make_data(N_TRAIN, class_means, sigma, noise, seed=1000 + seed)
    X_cal, y_cal, _ = make_data(N_CAL, class_means, sigma, 0.0, seed=2000 + seed)
    X_test, y_test, _ = make_data(N_TEST, class_means, sigma, 0.0, seed=3000 + seed)
    Xtr = torch.tensor(X_train); ytr = torch.tensor(y_train)
    Xcal = torch.tensor(X_cal); Xte = torch.tensor(X_test)
    model = train_model(Xtr, ytr, seed=seed)
    T = fit_temperature(model, Xcal, y_cal)
    raw_fn = lambda X: softmax_probs(model, X, T=1.0)
    ts_fn = lambda X: softmax_probs(model, X, T=T)
    lac_raw = conformal_eval_lac(raw_fn, Xcal, y_cal, Xte, y_test)
    lac_ts = conformal_eval_lac(ts_fn, Xcal, y_cal, Xte, y_test)
    aps_raw = conformal_eval_aps(raw_fn, Xcal, y_cal, Xte, y_test)
    aps_ts = conformal_eval_aps(ts_fn, Xcal, y_cal, Xte, y_test)
    return dict(T=T, lac_raw=lac_raw, lac_ts=lac_ts, aps_raw=aps_raw, aps_ts=aps_ts)

configs = [
    ("noise_0.00", 1.8, 0.00), ("noise_0.12", 1.8, 0.12), ("noise_0.24", 1.8, 0.24),
    ("sigma_1.2", 1.2, 0.12), ("sigma_1.8", 1.8, 0.12), ("sigma_2.4", 2.4, 0.12),
]
summary = {}
for label, sigma, noise in configs:
    runs = [run_config(sigma, noise, seed) for seed in range(N_SEEDS)]
    agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    std = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    # per-seed effect arrays for paired significance testing
    per_seed_calib_lac = np.array([r["lac_ts"] - r["lac_raw"] for r in runs])
    per_seed_calib_aps = np.array([r["aps_ts"] - r["aps_raw"] for r in runs])
    per_seed_score = np.array([r["aps_raw"] - r["lac_raw"] for r in runs])
    calib_effect_lac = abs(agg["lac_ts"] - agg["lac_raw"])
    calib_effect_aps = abs(agg["aps_ts"] - agg["aps_raw"])
    score_effect = abs(agg["aps_raw"] - agg["lac_raw"])
    # paired t-test: is calib_effect_APS significantly different from score_effect?
    t_calib_vs_score, p_calib_vs_score = stats.ttest_rel(np.abs(per_seed_calib_aps), np.abs(per_seed_score))
    t_calib_lac_vs_0, p_calib_lac_vs_0 = stats.ttest_1samp(per_seed_calib_lac, 0.0)
    summary[label] = dict(**agg,
                           calib_effect_LAC=calib_effect_lac, calib_effect_LAC_std=std["lac_ts"],
                           calib_effect_APS=calib_effect_aps, calib_effect_APS_std=std["aps_ts"],
                           score_effect=score_effect, score_effect_std=std["aps_raw"],
                           p_calib_APS_vs_score=float(p_calib_vs_score),
                           p_calib_LAC_vs_zero=float(p_calib_lac_vs_0))
    print(label, summary[label])

print(f"Total sweep time: {time.time()-t_start:.1f}s")
with open("results_sweep.json", "w") as f:
    json.dump(summary, f, indent=2)
