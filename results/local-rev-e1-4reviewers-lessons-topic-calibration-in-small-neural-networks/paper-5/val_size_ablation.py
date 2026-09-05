"""
Targeted ablation requested by all four reviewers of v1: does enlarging the
validation set (holding training data fixed) stabilize the post-hoc
temperature fit at near-chance accuracy (digits, width=2), and does the same
manipulation matter at a width where accuracy is already good (width=16)?

Train set fraction is held fixed at 60% of the digits dataset; the remaining
40% is split between validation and test at four different ratios, so only
the amount of data used to *fit the temperature* changes.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from run_experiment import MLP, ece, fit_temperature

VAL_FRACS_OF_REMAINDER = [0.125, 0.25, 0.5, 0.75]  # val size as frac of the 40% held out
WIDTHS = [2, 16]
N_SEEDS = 20
EPOCHS = 120
LR = 0.05 * 0.02


def load_split(seed, val_frac_of_remainder):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=1 - val_frac_of_remainder, random_state=seed, stratify=y_rem
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_one(width, val_frac, seed_idx):
    seed = 5_000_000 + int(val_frac * 1000) * 1000 + width * 10 + seed_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(seed, val_frac)
    model = MLP(X_train.shape[1], width, 10)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        test_logits = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    T = fit_temperature(val_logits, y_val)
    probs_raw = softmax(test_logits)
    probs_ts = softmax(test_logits / T)
    acc = float((probs_raw.argmax(axis=1) == y_test).mean())
    return {
        "width": width,
        "val_frac_of_remainder": val_frac,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "seed_idx": seed_idx,
        "acc_test": acc,
        "temperature": T,
        "ece_raw": ece(probs_raw, y_test),
        "ece_ts": ece(probs_ts, y_test),
    }


def main():
    t0 = time.time()
    results = []
    for width in WIDTHS:
        for val_frac in VAL_FRACS_OF_REMAINDER:
            for seed_idx in range(N_SEEDS):
                results.append(run_one(width, val_frac, seed_idx))
    print(f"elapsed={time.time()-t0:.1f}s, n={len(results)}")

    summary = {}
    for width in WIDTHS:
        for val_frac in VAL_FRACS_OF_REMAINDER:
            rows = [r for r in results if r["width"] == width and r["val_frac_of_remainder"] == val_frac]
            T = np.array([r["temperature"] for r in rows])
            ece_ts = np.array([r["ece_ts"] for r in rows])
            acc = np.array([r["acc_test"] for r in rows])
            key = f"width{width}_valfrac{val_frac}"
            summary[key] = {
                "n_val": rows[0]["n_val"],
                "n_test": rows[0]["n_test"],
                "acc_mean": float(acc.mean()),
                "temperature_mean": float(T.mean()),
                "temperature_sd": float(T.std(ddof=1)),
                "ece_ts_mean": float(ece_ts.mean()),
                "ece_ts_sd": float(ece_ts.std(ddof=1)),
            }
    with open("val_size_ablation_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=1)
    for k, v in summary.items():
        print(f"{k}: n_val={v['n_val']} acc={v['acc_mean']:.3f} T_sd={v['temperature_sd']:.3f} ece_ts_sd={v['ece_ts_sd']:.3f}")


if __name__ == "__main__":
    main()
