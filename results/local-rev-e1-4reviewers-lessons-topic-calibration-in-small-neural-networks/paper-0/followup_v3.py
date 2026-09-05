"""
Follow-up experiments for v3, addressing points raised independently by all
four round-2 reviewers:
  (a) Does a bounded temperature search fix the degenerate L-BFGS fit at
      hidden=2 + label smoothing, or is the pathology intrinsic to a
      near-random classifier regardless of optimizer?
  (b) A formal statistical check (binomial test) on the crossover mode
      claim (8 in 7/10 seeds), plus a bootstrap over seeds to see how much
      the mode/range would move with a different sample of 10 seeds.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DEVICE = "cpu"
N_BINS = 15
SEEDS = list(range(10))


def load_data(seed):
    X, y = load_digits(return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    to_t = lambda a, dt: torch.tensor(a, dtype=dt)
    return (
        to_t(X_train, torch.float32), to_t(y_train, torch.long),
        to_t(X_val, torch.float32), to_t(y_val, torch.long),
        to_t(X_test, torch.float32), to_t(y_test, torch.long),
    )


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def train_model(hidden, X_train, y_train, label_smoothing=0.0):
    model = MLP(X_train.shape[1], hidden, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.train()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
    return model


def ece(probs, labels, n_bins=N_BINS):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        prop = in_bin.float().mean().item()
        bin_counts.append(int(in_bin.sum().item()))
        if prop > 0:
            acc_bin = accuracies[in_bin].float().mean().item()
            conf_bin = confidences[in_bin].mean().item()
            ece_val += abs(acc_bin - conf_bin) * prop
    return ece_val, bin_counts


def fit_temperature_bounded(logits, labels):
    """Bounded scalar search (Brent's method within [0.05, 20]) instead of
    unconstrained L-BFGS, directly answering reviewers' question of whether
    a bounded optimizer avoids the degenerate near-zero fit."""
    def nll(T):
        T = max(T, 1e-3)
        return F.cross_entropy(logits / T, labels).item()
    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                           options={"xatol": 1e-4})
    return res.x


def evaluate(model, X, y, T=1.0):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits / T, dim=1)
        preds = probs.argmax(dim=1)
        acc = preds.eq(y).float().mean().item()
        avg_conf = probs.max(dim=1).values.mean().item()
        e, bin_counts = ece(probs, y)
    return acc, avg_conf, e, bin_counts


def experiment_a():
    """Bounded-T fix attempt at hidden=2 + label smoothing across 10 seeds."""
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)
        model = train_model(2, X_train, y_train, label_smoothing=0.1)
        acc, conf, e_pre, _ = evaluate(model, X_test, y_test)
        with torch.no_grad():
            val_logits = model(X_val)
        T_bounded = fit_temperature_bounded(val_logits, y_val)
        acc_post, conf_post, e_post, bin_counts = evaluate(model, X_test, y_test, T=T_bounded)
        rows.append(dict(seed=seed, test_acc=acc, ece_pre=e_pre,
                          T_bounded=T_bounded, ece_post_bounded=e_post,
                          min_test_bin_count=min(bin_counts)))
    return rows


def experiment_b():
    """Statistical check on the crossover-mode claim + a seed-resampling
    bootstrap to see how much the mode/range would move under resampling."""
    crossover_widths = [16, 8, 8, 8, 4, 4, 8, 8, 8, 8]  # from Table 2, v2
    n = len(crossover_widths)
    k = sum(1 for w in crossover_widths if w == 8)
    # Binomial test: is 7/10 "mode 8" surprising under a null of uniform
    # choice among the 3 observed values {4, 8, 16}?
    bt = binomtest(k, n, p=1/3, alternative="greater")
    # Bootstrap: resample 10 seeds with replacement, 5000 times, record mode
    rng = np.random.default_rng(0)
    widths = np.array(crossover_widths)
    modes = []
    for _ in range(5000):
        sample = rng.choice(widths, size=n, replace=True)
        vals, counts = np.unique(sample, return_counts=True)
        modes.append(vals[np.argmax(counts)])
    modes = np.array(modes)
    mode_dist = {int(v): int((modes == v).sum()) for v in np.unique(modes)}
    return dict(k=k, n=n, binomial_p_greater_than_uniform=bt.pvalue,
                bootstrap_mode_distribution_over_5000=mode_dist,
                bootstrap_frac_mode_is_8=float((modes == 8).mean()))


if __name__ == "__main__":
    t0 = time.time()
    a = experiment_a()
    b = experiment_b()
    out = dict(bounded_T_hidden2_smoothed=a, crossover_stats=b, elapsed_sec=time.time() - t0)
    print(json.dumps(out, indent=2))
    with open("followup_v3_results.json", "w") as f:
        json.dump(out, f, indent=2)
