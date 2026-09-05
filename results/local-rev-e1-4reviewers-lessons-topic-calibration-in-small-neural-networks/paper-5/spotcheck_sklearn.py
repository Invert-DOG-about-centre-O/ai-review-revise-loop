"""
Cheap cross-library spot check: does the "temperature scaling reduces ECE"
descriptive finding replicate outside our custom PyTorch training loop?

Uses sklearn's MLPClassifier (a different implementation/solver stack:
lbfgs-backed Adam, different weight init and numeric backend than the
hand-rolled PyTorch loop used for the main study) on the digits dataset at
three widths. Since sklearn does not expose pre-softmax logits directly,
we recover logits up to an additive-per-sample constant via log(predict_proba),
which is sufficient for temperature scaling because softmax is invariant to
a uniform additive shift of its inputs.
"""
import json

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from run_experiment import ece, nll

WIDTHS = [2, 32, 256]
N_SEEDS = 15


def fit_temperature_np(val_logits, val_labels):
    def nll_at_T(T):
        z = val_logits / max(T, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return nll(p, val_labels)

    res = minimize_scalar(nll_at_T, bounds=(0.05, 10.0), method="bounded")
    return res.x


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


X, y = load_digits(return_X_y=True)
X = X / 16.0

results = []
conv_warnings = []
for width in WIDTHS:
    for seed_idx in range(N_SEEDS):
        seed = 2_000_000 + WIDTHS.index(width) * 100 + seed_idx
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )
        clf = MLPClassifier(
            hidden_layer_sizes=(width,),
            activation="relu",
            solver="adam",
            max_iter=3000,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            clf.fit(X_train, y_train)
            for w in wlist:
                conv_warnings.append(
                    {"width": width, "seed_idx": seed_idx, "message": str(w.message)}
                )

        val_probs = clf.predict_proba(X_val)
        test_probs = clf.predict_proba(X_test)
        val_logits = np.log(np.clip(val_probs, 1e-12, 1.0))
        test_logits = np.log(np.clip(test_probs, 1e-12, 1.0))

        T = fit_temperature_np(val_logits, y_val)
        test_probs_ts = softmax(test_logits / T)

        results.append(
            {
                "width": width,
                "seed_idx": seed_idx,
                "acc": float((test_probs.argmax(1) == y_test).mean()),
                "ece_raw": ece(test_probs, y_test),
                "ece_ts": ece(test_probs_ts, y_test),
                "temperature": float(T),
            }
        )

raw = np.array([r["ece_raw"] for r in results])
ts = np.array([r["ece_ts"] for r in results])
from scipy import stats

diff = raw - ts
w_stat, p = stats.wilcoxon(diff)

out = {
    "library": "sklearn.neural_network.MLPClassifier (solver=adam)",
    "dataset": "digits",
    "widths": WIDTHS,
    "n_seeds": N_SEEDS,
    "results": results,
    "mean_ece_raw": float(raw.mean()),
    "mean_ece_ts": float(ts.mean()),
    "mean_reduction": float(diff.mean()),
    "wilcoxon_p": float(p),
    "n_convergence_warnings": len(conv_warnings),
    "convergence_warnings": conv_warnings,
}
with open("spotcheck_results.json", "w") as f:
    json.dump(out, f, indent=1)

print(
    f"sklearn spot check: raw ECE={raw.mean():.4f} -> ts ECE={ts.mean():.4f} "
    f"(reduction={diff.mean():.4f}, wilcoxon p={p:.4g}, n={len(results)})"
)
