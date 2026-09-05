"""
Follow-up experiments for v4, addressing points raised independently across
round-3 reviews:
  (a) The binomial null for the crossover mode was restricted to the three
      *observed* widths {4,8,16} (reviewers 2,3 flagged this as post-hoc).
      We add a second test against a null uniform over the full 9-width grid.
  (b) The Limitations section speculated that ECE *binning* noise (sparse
      bins on a ~360-example test set) might drive crossover jitter. But the
      direction call (over/under-confident) is a raw confidence-vs-accuracy
      comparison with NO binning involved -- so binning cannot be the
      mechanism. We instead directly quantify test-set *sampling* noise: for
      each trained seed model at widths 4/8/16, bootstrap-resample the test
      set itself (holding the model fixed) and see how often the
      confidence-vs-accuracy sign flips.
  (c) Does the bounded-T fix generalize beyond hidden=2? We apply it at
      hidden=4,8,16 with label smoothing across all 10 seeds and compare to
      the unconstrained fit already reported in Table 3.
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
FULL_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512]


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
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        prop = in_bin.float().mean().item()
        if prop > 0:
            acc_bin = accuracies[in_bin].float().mean().item()
            conf_bin = confidences[in_bin].mean().item()
            ece_val += abs(acc_bin - conf_bin) * prop
    return ece_val


def fit_temperature_unconstrained(logits, labels):
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.clamp(min=1e-3).item()


def fit_temperature_bounded(logits, labels):
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
        e = ece(probs, y)
    return acc, avg_conf, e


def experiment_wider_null():
    """Binomial test against a null uniform over the FULL 9-width grid,
    not just the 3 widths that happened to be observed as crossovers."""
    crossover_widths = [16, 8, 8, 8, 4, 4, 8, 8, 8, 8]
    n = len(crossover_widths)
    k = sum(1 for w in crossover_widths if w == 8)
    bt_narrow = binomtest(k, n, p=1/3, alternative="greater")
    bt_wide = binomtest(k, n, p=1/len(FULL_GRID), alternative="greater")
    return dict(k=k, n=n,
                p_narrow_null_3widths=bt_narrow.pvalue,
                p_wide_null_9widths=bt_wide.pvalue)


def experiment_testset_noise():
    """For widths {4,8,16}, smoothing=0: bootstrap-resample the FIXED test
    set (2000 resamples) per seed's trained model, and measure how often the
    confidence-vs-accuracy sign flips relative to the full-test-set call.
    This isolates test-set sampling noise in the direction statistic from
    training/seed noise, and shows binning is not the mechanism (direction
    uses no bins)."""
    rng = np.random.default_rng(0)
    out = {}
    for hidden in [4, 8, 16]:
        flip_fracs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)
            model = train_model(hidden, X_train, y_train, label_smoothing=0.0)
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(X_test), dim=1)
                conf_all = probs.max(dim=1).values.numpy()
                preds = probs.argmax(dim=1).numpy()
                correct = (preds == y_test.numpy()).astype(float)
            full_sign = conf_all.mean() - correct.mean()  # >0 => overconfident
            n_test = len(correct)
            flips = 0
            for _ in range(2000):
                idx = rng.integers(0, n_test, n_test)
                resample_sign = conf_all[idx].mean() - correct[idx].mean()
                if (resample_sign > 0) != (full_sign > 0):
                    flips += 1
            flip_fracs.append(flips / 2000)
        out[hidden] = dict(mean_flip_frac=float(np.mean(flip_fracs)),
                            per_seed_flip_frac=[float(x) for x in flip_fracs])
    return out


def experiment_bounded_generalization():
    """Does the bounded-T fix generalize past hidden=2? Apply it at
    hidden=4,8,16 with label smoothing=0.1 across 10 seeds, compare to
    unconstrained fit on the identical validation logits."""
    out = {}
    for hidden in [4, 8, 16]:
        rows = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)
            model = train_model(hidden, X_train, y_train, label_smoothing=0.1)
            with torch.no_grad():
                val_logits = model(X_val)
            T_unc = fit_temperature_unconstrained(val_logits, y_val)
            T_bnd = fit_temperature_bounded(val_logits, y_val)
            _, _, e_unc = evaluate(model, X_test, y_test, T=T_unc)
            _, _, e_bnd = evaluate(model, X_test, y_test, T=T_bnd)
            rows.append(dict(seed=seed, T_unconstrained=T_unc, T_bounded=T_bnd,
                              ece_unconstrained=e_unc, ece_bounded=e_bnd))
        out[hidden] = dict(
            rows=rows,
            mean_T_unconstrained=float(np.mean([r["T_unconstrained"] for r in rows])),
            mean_T_bounded=float(np.mean([r["T_bounded"] for r in rows])),
            mean_ece_unconstrained=float(np.mean([r["ece_unconstrained"] for r in rows])),
            mean_ece_bounded=float(np.mean([r["ece_bounded"] for r in rows])),
            max_abs_T_diff=float(max(abs(r["T_unconstrained"] - r["T_bounded"]) for r in rows)),
        )
    return out


if __name__ == "__main__":
    t0 = time.time()
    wider_null = experiment_wider_null()
    testset_noise = experiment_testset_noise()
    bounded_gen = experiment_bounded_generalization()
    out = dict(wider_null=wider_null, testset_noise=testset_noise,
               bounded_generalization=bounded_gen, elapsed_sec=time.time() - t0)
    print(json.dumps(out, indent=2))
    with open("followup_v4_results.json", "w") as f:
        json.dump(out, f, indent=2)
