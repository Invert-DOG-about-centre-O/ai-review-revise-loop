"""
Third dataset check (exploratory, run after seeing the main result): sklearn's
breast_cancer (real, 2-class, 569 samples), same pipeline as experiment.py,
to test whether the pre-ECE-predicts-benefit finding and the width-T*
relationship replicate on another real dataset beyond digits/synthetic.
"""
import json
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, wilcoxon

T0 = time.time()
WIDTHS = [4, 8, 16, 32, 64, 128, 256]
N_SEEDS = 20
BASE_SEED = 20260827
DATASET_OFFSET = 20_000_000
WIDTH_OFFSET = {w: i * 100_000 for i, w in enumerate(WIDTHS)}
N_BINS = 15
EPS = 1e-12


def make_seed(width, seed_idx):
    return BASE_SEED + DATASET_OFFSET + WIDTH_OFFSET[width] + seed_idx


def ece(probs, labels, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def probs_to_pseudologits(p):
    return np.log(np.clip(p, EPS, 1.0))


def softmax_with_temp(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_probs, val_labels):
    logits = probs_to_pseudologits(val_probs)

    def nll(T):
        p = softmax_with_temp(logits, T)
        p_true = p[np.arange(len(val_labels)), val_labels]
        return -np.log(np.clip(p_true, EPS, 1.0)).mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                           options={"xatol": 1e-4})
    return float(res.x)


def run_one(X, y, width, seed_idx):
    seed = make_seed(width, seed_idx)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd

    clf = MLPClassifier(hidden_layer_sizes=(width,), activation="relu",
                         solver="adam", alpha=1e-4, max_iter=300,
                         random_state=seed)
    clf.fit(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)
    pre_ece = ece(test_probs, y_test)
    T_star = fit_temperature(val_probs, y_val)
    test_logits = probs_to_pseudologits(test_probs)
    post_ece = ece(softmax_with_temp(test_logits, T_star), y_test)
    return dict(width=width, seed_idx=seed_idx, seed=seed, test_acc=test_acc,
                pre_ece=pre_ece, post_ece=post_ece, T_star=T_star)


def main():
    d = load_breast_cancer()
    X, y = d.data.astype(np.float64), d.target
    results = []
    for width in WIDTHS:
        for seed_idx in range(N_SEEDS):
            results.append(run_one(X, y, width, seed_idx))
    with open("results_dataset3_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    widths = np.array([r["width"] for r in results], dtype=float)
    T = np.array([r["T_star"] for r in results])
    pre = np.array([r["pre_ece"] for r in results])
    post = np.array([r["post_ece"] for r in results])
    delta = pre - post

    rho, p_rho = spearmanr(np.log2(widths), T, alternative="greater")
    wstat, p_w = wilcoxon(pre, post, alternative="greater")
    rho_pre_delta, p_pre_delta = spearmanr(pre, delta)
    mask = widths > 4
    rho_excl4, p_excl4 = spearmanr(np.log2(widths[mask]), T[mask], alternative="greater")

    summary = dict(
        n_runs=len(results),
        spearman_rho_width_Tstar=float(rho), spearman_p=float(p_rho),
        spearman_rho_width_Tstar_excl_width4=float(rho_excl4), p_excl_width4=float(p_excl4),
        wilcoxon_p_pre_gt_post=float(p_w),
        mean_pre_ece=float(pre.mean()), mean_post_ece=float(post.mean()),
        spearman_rho_preece_delta=float(rho_pre_delta), p_preece_delta=float(p_pre_delta),
    )
    with open("results_dataset3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"TOTAL ELAPSED: {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
