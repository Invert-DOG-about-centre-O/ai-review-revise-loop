"""
Multi-seed replication of experiment.py to answer reviewer concerns about
single-run noise. Repeats the full pipeline (data draw, training, calibration,
conformal eval) across N_SEEDS independent seeds (class means, label noise,
train/cal/test draws, and model init all vary together) and reports
mean +/- std for every headline number in the paper.
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
K_ENSEMBLE = 5
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

def train_model(Xtr, ytr, seed, epochs=80, bootstrap=False):
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if bootstrap:
        idx = np.random.RandomState(seed).choice(len(Xtr), len(Xtr), replace=True)
        xb, yb = Xtr[idx], ytr[idx]
    else:
        xb, yb = Xtr, ytr
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def softmax_probs(model, X, T=1.0):
    return F.softmax(model(X) / T, dim=1).numpy()

@torch.no_grad()
def ensemble_probs(models, X, T=1.0):
    ps = [F.softmax(m(X) / T, dim=1).numpy() for m in models]
    return np.mean(ps, axis=0)

def fit_temperature(logit_fn, X, y, n_iter=200):
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
        e += (mask.sum() / total) * abs(correct[mask].mean() - conf[mask].mean())
    return e

def conformal_eval(prob_fn, Xcal, ycal, Xte, yte, alpha=0.10):
    p_cal = prob_fn(Xcal)
    scores_cal = 1.0 - p_cal[np.arange(len(ycal)), ycal]
    n = len(ycal)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores_cal, q_level, method="higher")
    p_test = prob_fn(Xte)
    pred_sets = p_test >= (1 - qhat)
    covered = pred_sets[np.arange(len(yte)), yte]
    return dict(coverage=float(covered.mean()), avg_set_size=float(pred_sets.sum(axis=1).mean()),
                empty_frac=float((pred_sets.sum(axis=1) == 0).mean()))

def aps_score(probs, y):
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])
    return cumsum[np.arange(len(y)), rank]

def aps_eval(prob_fn, Xcal, ycal, Xte, yte, alpha=0.10):
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
    covered = pred_sets[np.arange(len(yte)), yte]
    return dict(coverage=float(covered.mean()), avg_set_size=float(pred_sets.sum(axis=1).mean()),
                empty_frac=float((pred_sets.sum(axis=1) == 0).mean()))

method_names = ["raw_softmax", "temp_scaled", "ensemble", "ensemble_temp_scaled"]
agg = {m: {"ece": [], "accuracy": [], "T": []} for m in method_names}
agg_lac = {m: {"coverage": [], "avg_set_size": [], "empty_frac": []} for m in method_names}
agg_aps = {m: {"coverage": [], "avg_set_size": [], "empty_frac": []} for m in method_names}

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng0 = np.random.RandomState(42 + seed)
    class_means = rng0.randn(N_CLASSES, DIM) * 1.6

    X_train, _, y_train = make_data(N_TRAIN, class_means, LABEL_NOISE, seed=1000 + seed)
    X_cal, y_cal, _ = make_data(N_CAL, class_means, 0.0, seed=2000 + seed)
    X_test, y_test, _ = make_data(N_TEST, class_means, 0.0, seed=3000 + seed)

    Xtr = torch.tensor(X_train); ytr = torch.tensor(y_train)
    Xcal = torch.tensor(X_cal); Xte = torch.tensor(X_test)

    base_model = train_model(Xtr, ytr, seed=seed, epochs=80, bootstrap=False)
    ensemble = [train_model(Xtr, ytr, seed=100 * (seed + 1) + k, epochs=80, bootstrap=True) for k in range(K_ENSEMBLE)]

    T_base = fit_temperature(lambda X: base_model(X), Xcal, y_cal)
    T_ens = fit_temperature(lambda X: torch.mean(torch.stack([m(X) for m in ensemble]), dim=0), Xcal, y_cal)

    methods = {
        "raw_softmax":          lambda X: softmax_probs(base_model, X, T=1.0),
        "temp_scaled":          lambda X: softmax_probs(base_model, X, T=T_base),
        "ensemble":              lambda X: ensemble_probs(ensemble, X, T=1.0),
        "ensemble_temp_scaled":  lambda X: ensemble_probs(ensemble, X, T=T_ens),
    }
    Tvals = {"raw_softmax": 1.0, "temp_scaled": T_base, "ensemble": 1.0, "ensemble_temp_scaled": T_ens}

    for name, fn in methods.items():
        p_test = fn(Xte)
        agg[name]["ece"].append(ece(p_test, y_test))
        agg[name]["accuracy"].append((p_test.argmax(axis=1) == y_test).mean())
        agg[name]["T"].append(Tvals[name])
        lac = conformal_eval(fn, Xcal, y_cal, Xte, y_test, alpha=0.10)
        for k, v in lac.items():
            agg_lac[name][k].append(v)
        aps = aps_eval(fn, Xcal, y_cal, Xte, y_test, alpha=0.10)
        for k, v in aps.items():
            agg_aps[name][k].append(v)
    print(f"seed {seed} done at t={time.time()-t_start:.1f}s")

def summarize(d):
    return {m: {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in metrics.items()} for m, metrics in d.items()}

# Paired significance tests across the 10 matched seeds (same data draw/init per seed),
# addressing reviewer request for formal tests rather than eyeballing effect vs. std.
lac_raw_arr = np.array(agg_lac["raw_softmax"]["avg_set_size"])
lac_ts_arr = np.array(agg_lac["temp_scaled"]["avg_set_size"])
aps_raw_arr = np.array(agg_aps["raw_softmax"]["avg_set_size"])
aps_ts_arr = np.array(agg_aps["temp_scaled"]["avg_set_size"])
ece_raw_arr = np.array(agg["raw_softmax"]["ece"])
ece_ens_arr = np.array(agg["ensemble"]["ece"])

t_lac_calib, p_lac_calib = stats.ttest_rel(lac_ts_arr, lac_raw_arr)
t_aps_calib, p_aps_calib = stats.ttest_rel(aps_ts_arr, aps_raw_arr)
score_switch_per_seed = aps_raw_arr - lac_raw_arr
aps_calib_per_seed = aps_ts_arr - aps_raw_arr
t_score_vs_apscalib, p_score_vs_apscalib = stats.ttest_rel(np.abs(score_switch_per_seed), np.abs(aps_calib_per_seed))
t_ens_ece, p_ens_ece = stats.ttest_rel(ece_ens_arr, ece_raw_arr)

significance = dict(
    lac_calib_effect_ttest=dict(t=float(t_lac_calib), p=float(p_lac_calib)),
    aps_calib_effect_ttest=dict(t=float(t_aps_calib), p=float(p_aps_calib)),
    score_switch_vs_aps_calib_ttest=dict(t=float(t_score_vs_apscalib), p=float(p_score_vs_apscalib)),
    ensemble_worsens_ece_ttest=dict(t=float(t_ens_ece), p=float(p_ens_ece)),
)

out = dict(
    n_seeds=N_SEEDS,
    calibration=summarize(agg),
    conformal_LAC=summarize(agg_lac),
    conformal_APS=summarize(agg_aps),
    significance=significance,
    elapsed_seconds=time.time() - t_start,
)
with open("results_multiseed.json", "w") as f:
    json.dump(out, f, indent=2)
print("Wrote results_multiseed.json")
print(json.dumps(out, indent=2))
