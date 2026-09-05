"""
Calibration in small neural networks: does label smoothing add anything
beyond post-hoc temperature scaling, across a width sweep?

Trains small single-hidden-layer MLPs of varying width on two datasets
(sklearn digits, synthetic blobs-with-label-noise), with and without label
smoothing, over many seeds. Evaluates ECE/NLL/accuracy before and after
post-hoc temperature scaling.

All seeds are derived by fixed integer arithmetic on explicit indices
(no hash()) so reruns are bit-identical.
"""
import json
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
CONDITIONS = ["baseline", "label_smooth"]
N_SEEDS = 15
EPOCHS = 120
LR = 0.05
N_BINS = 15
LS_EPS = 0.1

DATASET_INDEX = {"digits": 0, "blobs": 1}
CONDITION_INDEX = {"baseline": 0, "label_smooth": 1}


def make_seed(dataset, condition, width, seed_idx):
    # explicit fixed-integer mapping, no hash()
    w_idx = WIDTHS.index(width)
    return (
        1_000_000
        + DATASET_INDEX[dataset] * 100_000
        + CONDITION_INDEX[condition] * 10_000
        + w_idx * 100
        + seed_idx
    )


def load_dataset(name, seed):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
        X = X / 16.0
    elif name == "blobs":
        X, y = make_classification(
            n_samples=1500,
            n_features=12,
            n_informative=6,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=2,
            class_sep=1.0,
            flip_y=0.05,
            random_state=seed,
        )
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    else:
        raise ValueError(name)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    n_classes = len(np.unique(y))
    return (X_train, y_train, X_val, y_val, X_test, y_test, n_classes)


class MLP(nn.Module):
    def __init__(self, in_dim, width, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def smoothed_ce_loss(logits, targets, n_classes, eps):
    logp = torch.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.full_like(logp, eps / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - eps)
    return -(true_dist * logp).sum(dim=1).mean()


def ece(probs, labels, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        e += (mask.sum() / total) * abs(acc_bin - conf_bin)
    return float(e)


def nll(probs, labels):
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(val_logits, val_labels):
    T = torch.ones(1, requires_grad=True)
    labels_t = torch.tensor(val_labels, dtype=torch.long)
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    optimizer = torch.optim.LBFGS([T], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits_t / T.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.clamp(min=1e-3).item())


def train_one(dataset, condition, width, seed_idx, warnings_log):
    seed = make_seed(dataset, condition, width, seed_idx)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test, n_classes = load_dataset(
        dataset, seed
    )
    in_dim = X_train.shape[1]

    model = MLP(in_dim, width, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.02)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for epoch in range(EPOCHS):
            opt.zero_grad()
            logits = model(Xt)
            if condition == "baseline":
                loss = nn.functional.cross_entropy(logits, yt)
            else:
                loss = smoothed_ce_loss(logits, yt, n_classes, LS_EPS)
            loss.backward()
            opt.step()
        for w in wlist:
            warnings_log.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "width": width,
                    "seed_idx": seed_idx,
                    "message": str(w.message),
                }
            )

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        test_logits = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    T = fit_temperature(val_logits, y_val)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    probs_raw = softmax(test_logits)
    probs_ts = softmax(test_logits / T)

    acc = float((probs_raw.argmax(axis=1) == y_test).mean())

    return {
        "dataset": dataset,
        "condition": condition,
        "width": width,
        "seed_idx": seed_idx,
        "seed": seed,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "acc_test": acc,
        "ece_raw": ece(probs_raw, y_test),
        "ece_ts": ece(probs_ts, y_test),
        "nll_raw": nll(probs_raw, y_test),
        "nll_ts": nll(probs_ts, y_test),
        "temperature": T,
    }


def main():
    t0 = time.time()
    results = []
    warnings_log = []
    total = len(DATASET_INDEX) * len(CONDITIONS) * len(WIDTHS) * N_SEEDS
    count = 0
    for dataset in DATASET_INDEX:
        for condition in CONDITIONS:
            for width in WIDTHS:
                for seed_idx in range(N_SEEDS):
                    r = train_one(dataset, condition, width, seed_idx, warnings_log)
                    results.append(r)
                    count += 1
                    if count % 50 == 0:
                        print(
                            f"{count}/{total} done, elapsed={time.time()-t0:.1f}s",
                            file=sys.stderr,
                        )
    elapsed = time.time() - t0
    print(f"TOTAL elapsed: {elapsed:.1f}s for {total} runs", file=sys.stderr)

    with open("raw_results.json", "w") as f:
        json.dump(
            {
                "config": {
                    "widths": WIDTHS,
                    "conditions": CONDITIONS,
                    "n_seeds": N_SEEDS,
                    "epochs": EPOCHS,
                    "lr": LR * 0.02,
                    "n_bins": N_BINS,
                    "ls_eps": LS_EPS,
                },
                "results": results,
                "warnings": warnings_log,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=1,
        )
    print(f"Saved {len(results)} results, {len(warnings_log)} warnings")


if __name__ == "__main__":
    main()
