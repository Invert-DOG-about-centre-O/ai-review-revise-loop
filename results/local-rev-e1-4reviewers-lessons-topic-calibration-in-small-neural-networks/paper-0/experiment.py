"""
Capacity-dependent calibration study for small neural networks.

We sweep MLP hidden width over a wide range (2 -> 512 units) on a real
multi-class digit classification task (sklearn `digits`, 10 classes, 8x8
images). For each width we:
  1. Train a 1-hidden-layer MLP (with and without label smoothing).
  2. Measure pre-calibration ECE / average confidence / accuracy on a held
     out test set.
  3. Fit post-hoc temperature scaling on a validation split (minimizing
     NLL) and re-measure ECE with the learned temperature T.
  4. Record whether the network is over- or under-confident, and whether
     the fitted T is >1 (softens, appropriate for overconfidence) or <1
     (sharpens, appropriate for underconfidence).

This lets us see, empirically, whether small (underparameterized) networks
sit on the *other side* of the over/under-confidence boundary from large
networks, and whether a single global calibration recipe (temperature
scaling tuned the "standard" direction) is even the right tool once you
get small enough.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cpu"
N_BINS = 15
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
EPOCHS = 200
LR = 1e-2
WEIGHT_DECAY = 0.0


def load_data():
    X, y = load_digits(return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
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
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_model(hidden, X_train, y_train, label_smoothing=0.0):
    model = MLP(X_train.shape[1], hidden, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
    return model


def ece(probs, labels, n_bins=N_BINS):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    total = len(labels)
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


def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.clamp(min=1e-3).item()


def evaluate(model, X, y, T=1.0):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits / T, dim=1)
        preds = probs.argmax(dim=1)
        acc = preds.eq(y).float().mean().item()
        avg_conf = probs.max(dim=1).values.mean().item()
        e = ece(probs, y)
    return acc, avg_conf, e, logits


def run():
    t0 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    results = []
    for smoothing in [0.0, 0.1]:
        for hidden in WIDTHS:
            model = train_model(hidden, X_train, y_train, label_smoothing=smoothing)

            train_acc, train_conf, train_ece, _ = evaluate(model, X_train, y_train)
            acc, conf, e_pre, test_logits = evaluate(model, X_test, y_test)

            # Fit temperature on validation set
            with torch.no_grad():
                val_logits = model(X_val)
            T = fit_temperature(val_logits, y_val)

            acc_post, conf_post, e_post, _ = evaluate(model, X_test, y_test, T=T)

            n_params = sum(p.numel() for p in model.parameters())
            direction = "underconfident" if conf < acc else ("overconfident" if conf > acc else "calibrated")

            row = dict(
                smoothing=smoothing, hidden=hidden, n_params=n_params,
                train_acc=train_acc, train_ece=train_ece,
                test_acc=acc, test_conf_pre=conf, test_ece_pre=e_pre,
                direction=direction, fitted_T=T,
                test_conf_post=conf_post, test_ece_post=e_post,
            )
            results.append(row)
            print(f"smoothing={smoothing} hidden={hidden:4d} params={n_params:6d} "
                  f"test_acc={acc:.3f} conf={conf:.3f} ECE_pre={e_pre:.4f} "
                  f"T={T:.3f} ECE_post={e_post:.4f} [{direction}]")

    elapsed = time.time() - t0
    print(f"\nTotal experiment time: {elapsed:.1f}s")

    with open("results.json", "w") as f:
        json.dump({"results": results, "elapsed_sec": elapsed,
                    "widths": WIDTHS, "epochs": EPOCHS, "n_bins": N_BINS}, f, indent=2)


if __name__ == "__main__":
    run()
