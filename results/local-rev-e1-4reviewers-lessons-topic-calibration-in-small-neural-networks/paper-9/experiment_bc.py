"""
Second-dataset robustness check: same width sweep and MLP training code,
applied to sklearn breast_cancer (30-dim, 2-class, 569 samples) instead of
digits (64-dim, 10-class, 1797 samples), to test whether the width 4-8
miscalibration peak is digits-specific or a more general small-width pattern.
Reduced grid (drop 256/512) and same epoch count, to fit within time budget.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import json, time

from experiment import softmax, train_mlp, ece_score, nll_score, fit_temperature

t_start = time.time()

def run_all():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    n_classes = 2

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=y_trainval)

    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    widths = [2, 4, 8, 16, 32, 64, 128]
    n_seeds = 10

    results = []
    for width in widths:
        for seed in range(n_seeds):
            model = train_mlp(X_train, y_train, n_classes, width, seed,
                               epochs=150, lr=0.03, l2=1e-4, batch_size=64)

            test_logits = model.logits(X_test)
            test_probs = softmax(test_logits)
            test_acc = (test_probs.argmax(1) == y_test).mean()
            test_ece = ece_score(test_probs, y_test)
            test_nll = nll_score(test_probs, y_test)

            train_logits = model.logits(X_train)
            train_probs = softmax(train_logits)
            train_acc = (train_probs.argmax(1) == y_train).mean()

            results.append(dict(
                width=width, seed=seed,
                train_acc=float(train_acc), test_acc=float(test_acc),
                test_ece=float(test_ece), test_nll=float(test_nll),
                train_test_acc_gap=float(train_acc - test_acc),
            ))
        elapsed = time.time() - t_start
        print(f"width={width} done, elapsed={elapsed:.1f}s")

    with open("results_bc.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    results = run_all()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
