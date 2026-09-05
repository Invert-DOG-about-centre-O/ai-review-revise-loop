"""
Follow-up analysis addressing reviewer concerns on v1:
1. Quantify convergence (n_iter_ vs max_iter) by width/dataset.
2. Report training-set accuracy/ECE by width to test the underfitting-relief mechanism.
3. Partial (rank) correlation of ECE_pre with log2(width) controlling for test accuracy,
   to check whether the width effect survives once accuracy is accounted for
   (raised specifically because circles shows a 31-point accuracy swing across widths).
Uses the SAME fixed integer seeds as experiment.py so results are directly comparable;
does not touch or overwrite raw_results.json / analysis.json from the original run.
"""
import json
import numpy as np
import warnings
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WIDTHS = [2, 4, 8, 16, 32, 64, 128]
N_SEEDS = 10
DATASETS = ["moons", "circles", "breast_cancer"]
EPS = 1e-6
N_BINS = 15


def get_dataset(name, seed):
    if name == "moons":
        X, y = make_moons(n_samples=600, noise=0.30, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=600, noise=0.15, factor=0.5, random_state=seed)
    else:
        data = load_breast_cancer()
        X, y = data.data, data.target
    return X, y


def ece_binary(y_true, p1, n_bins=N_BINS):
    conf = np.maximum(p1, 1 - p1)
    pred = (p1 >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def run_one(dataset, width, seed):
    X, y = get_dataset(dataset, seed)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed + 1, stratify=y_rest
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(width,), activation="relu", solver="adam",
        alpha=1e-4, max_iter=2000, random_state=seed, early_stopping=False,
    )
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        clf.fit(X_train_s, y_train)
        converged = not any("Maximum iterations" in str(w.message) for w in wlist)

    p1_train = clf.predict_proba(X_train_s)[:, 1]
    p1_test = clf.predict_proba(X_test_s)[:, 1]
    acc_train = float(((p1_train >= 0.5).astype(int) == y_train).mean())
    acc_test = float(((p1_test >= 0.5).astype(int) == y_test).mean())
    ece_train = ece_binary(y_train, p1_train)
    ece_test = ece_binary(y_test, p1_test)

    return dict(
        dataset=dataset, width=width, seed=seed, n_iter=int(clf.n_iter_),
        converged=bool(converged), acc_train=acc_train, acc_test=acc_test,
        ece_train=ece_train, ece_test=ece_test,
    )


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x,y controlling for z, via rank residuals."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    bx = np.polyfit(rz, rx, 1)
    by = np.polyfit(rz, ry, 1)
    res_x = rx - np.polyval(bx, rz)
    res_y = ry - np.polyval(by, rz)
    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def main():
    results = []
    for di, dataset in enumerate(DATASETS):
        for wi, width in enumerate(WIDTHS):
            for ri in range(N_SEEDS):
                seed = di * 100000 + wi * 1000 + ri
                results.append(run_one(dataset, width, seed))

    with open("followup_results.json", "w") as f:
        json.dump(results, f, indent=2)

    out = {}
    for dataset in DATASETS:
        rows = [r for r in results if r["dataset"] == dataset]
        log2w = np.array([np.log2(r["width"]) for r in rows])
        acc_test = np.array([r["acc_test"] for r in rows])
        ece_test = np.array([r["ece_test"] for r in rows])
        acc_train = np.array([r["acc_train"] for r in rows])
        ece_train = np.array([r["ece_train"] for r in rows])
        n_converged = sum(r["converged"] for r in rows)

        acc_by_width = {w: float(np.mean([r["acc_test"] for r in rows if r["width"] == w])) for w in WIDTHS}
        train_acc_by_width = {w: float(np.mean([r["acc_train"] for r in rows if r["width"] == w])) for w in WIDTHS}
        train_ece_by_width = {w: float(np.mean([r["ece_train"] for r in rows if r["width"] == w])) for w in WIDTHS}
        nonconverged_by_width = {w: int(sum(1 for r in rows if r["width"] == w and not r["converged"])) for w in WIDTHS}

        pr, pp = partial_spearman(log2w, ece_test, acc_test)
        pr_train, pp_train = partial_spearman(log2w, ece_test, acc_train)
        acc_width_rho, acc_width_p = stats.spearmanr(log2w, acc_test)

        out[dataset] = dict(
            n_converged=n_converged, n_total=len(rows),
            nonconverged_by_width=nonconverged_by_width,
            acc_test_by_width=acc_by_width,
            train_acc_by_width=train_acc_by_width,
            train_ece_by_width=train_ece_by_width,
            spearman_log2width_vs_acc_test=float(acc_width_rho),
            spearman_log2width_vs_acc_test_p=float(acc_width_p),
            partial_spearman_log2width_vs_ece_pre_given_acc_test=pr,
            partial_spearman_p_given_acc_test=pp,
            partial_spearman_log2width_vs_ece_pre_given_acc_train=pr_train,
            partial_spearman_p_given_acc_train=pp_train,
        )

    with open("followup_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
