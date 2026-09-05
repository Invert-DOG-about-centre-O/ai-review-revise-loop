"""
Calibration vs. width in small neural networks.

For two datasets (digits: real, and a synthetic 3-class nonlinear problem),
train 1-hidden-layer MLPs of varying (small) width, fit post-hoc temperature
scaling on a held-out calibration split, and record pre-/post-calibration ECE
and the fitted optimal temperature T*, across many fixed-seed repeats.

Seeding: every (dataset, width, seed_idx) run uses an explicit integer seed
built from a fixed dict-based offset table -- never hash(). Each run reseeds
immediately before its own train/val/test split and model fit, so results
are per-run reproducible in isolation (see lessons.md).
"""
import json
import time
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, wilcoxon

T0 = time.time()

WIDTHS = [4, 8, 16, 32, 64, 128, 256]
N_SEEDS = 20
BASE_SEED = 20260827  # fixed, not hash()-derived
N_BINS = 15
EPS = 1e-12

DATASET_OFFSET = {"digits": 0, "synthetic": 10_000_000}
WIDTH_OFFSET = {w: i * 100_000 for i, w in enumerate(WIDTHS)}


def make_seed(dataset, width, seed_idx):
    return BASE_SEED + DATASET_OFFSET[dataset] + WIDTH_OFFSET[width] + seed_idx


def load_dataset(name, seed):
    if name == "digits":
        d = load_digits()
        X, y = d.data.astype(np.float64), d.target
    elif name == "synthetic":
        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_informative=8,
            n_redundant=4,
            n_classes=3,
            n_clusters_per_class=2,
            class_sep=1.0,
            flip_y=0.03,
            random_state=seed,
        )
    else:
        raise ValueError(name)
    # standardize features using train stats only, done after split below
    return X, y


def ece(probs, labels, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        total += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(total)


def probs_to_pseudologits(probs):
    return np.log(np.clip(probs, EPS, 1.0))


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


def run_one(dataset, width, seed_idx):
    seed = make_seed(dataset, width, seed_idx)
    rng = np.random.RandomState(seed)
    X, y = load_dataset(dataset, seed)

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

    clf = MLPClassifier(
        hidden_layer_sizes=(width,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=300,
        random_state=seed,
        early_stopping=False,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    pre_ece = ece(test_probs, y_test)
    T_star = fit_temperature(val_probs, y_val)
    test_logits = probs_to_pseudologits(test_probs)
    calibrated_probs = softmax_with_temp(test_logits, T_star)
    post_ece = ece(calibrated_probs, y_test)

    return dict(
        dataset=dataset, width=width, seed_idx=seed_idx, seed=seed,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        train_acc=train_acc, test_acc=test_acc,
        pre_ece=pre_ece, post_ece=post_ece, T_star=T_star,
    )


def main():
    results = []
    for dataset in ["digits", "synthetic"]:
        for width in WIDTHS:
            for seed_idx in range(N_SEEDS):
                r = run_one(dataset, width, seed_idx)
                results.append(r)
        elapsed = time.time() - T0
        print(f"finished dataset={dataset} at t={elapsed:.1f}s, "
              f"n_results={len(results)}")

    with open("results_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Pre-registered confirmatory analyses ----
    summary = {}
    for dataset in ["digits", "synthetic"]:
        rows = [r for r in results if r["dataset"] == dataset]
        widths = np.array([r["width"] for r in rows], dtype=np.float64)
        log2w = np.log2(widths)
        T_star = np.array([r["T_star"] for r in rows])
        pre_ece = np.array([r["pre_ece"] for r in rows])
        post_ece = np.array([r["post_ece"] for r in rows])

        rho, p_rho = spearmanr(log2w, T_star, alternative="greater")

        try:
            wstat, p_wilcoxon = wilcoxon(pre_ece, post_ece,
                                          alternative="greater")
        except ValueError:
            wstat, p_wilcoxon = float("nan"), float("nan")

        per_width = {}
        for w in WIDTHS:
            m = widths == w
            per_width[w] = dict(
                mean_T_star=float(T_star[m].mean()),
                sd_T_star=float(T_star[m].std(ddof=1)),
                mean_pre_ece=float(pre_ece[m].mean()),
                mean_post_ece=float(post_ece[m].mean()),
                mean_test_acc=float(np.array([r["test_acc"] for r in rows])[m].mean()),
                n=int(m.sum()),
            )

        summary[dataset] = dict(
            n_runs=len(rows),
            spearman_rho_width_Tstar=float(rho),
            spearman_p_one_sided=float(p_rho),
            wilcoxon_stat_pre_gt_post_ece=float(wstat),
            wilcoxon_p_one_sided=float(p_wilcoxon),
            mean_pre_ece=float(pre_ece.mean()),
            mean_post_ece=float(post_ece.mean()),
            mean_ece_reduction=float((pre_ece - post_ece).mean()),
            per_width=per_width,
        )

    # Bonferroni correction across the 2 datasets for the confirmatory
    # width-vs-T* correlation test (pre-registered alpha=0.05 family-wise).
    alpha_corrected = 0.05 / 2
    for dataset in summary:
        summary[dataset]["alpha_corrected"] = alpha_corrected
        summary[dataset]["significant_after_correction"] = bool(
            summary[dataset]["spearman_p_one_sided"] < alpha_corrected
        )

    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"TOTAL ELAPSED: {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
