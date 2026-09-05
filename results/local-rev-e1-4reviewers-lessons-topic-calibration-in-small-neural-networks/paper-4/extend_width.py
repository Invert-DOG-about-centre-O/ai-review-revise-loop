"""
Extended-width follow-up addressing round-2 reviewer questions about a
possible turning point between the underfitting-relief regime (this paper)
and the large-model overconfidence regime (Guo et al. 2017). Trains the
SAME architecture/pipeline at widths up to 1024 on all three datasets,
10 seeds each, using the same seed formula extended to new width indices.
"""
import json
import time
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WIDTHS_EXT = [128, 256, 512, 1024]  # 128 repeated for continuity with main study
N_SEEDS = 10
N_BINS = 15
EPS = 1e-6
DATASETS = ["moons", "circles", "breast_cancer"]


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
    clf.fit(X_train_s, y_train)

    p1_test = clf.predict_proba(X_test_s)[:, 1]
    acc_test = float(((p1_test >= 0.5).astype(int) == y_test).mean())
    ece_test = ece_binary(y_test, p1_test)
    return dict(dataset=dataset, width=width, seed=seed, acc_test=acc_test, ece_test=ece_test)


def main():
    t0 = time.time()
    results = []
    # use width_idx 7..10 (continuing the original 0..6 grid) for fresh seeds at new widths,
    # plus width_idx 6 (=128) recomputed with SAME seed formula as main study for continuity
    width_idx_map = {128: 6, 256: 7, 512: 8, 1024: 9}
    for di, dataset in enumerate(DATASETS):
        for width in WIDTHS_EXT:
            wi = width_idx_map[width]
            for ri in range(N_SEEDS):
                seed = di * 100000 + wi * 1000 + ri
                results.append(run_one(dataset, width, seed))
    elapsed = time.time() - t0
    print(f"Extended-width runs: {len(results)} in {elapsed:.1f}s")

    with open("extend_width_results.json", "w") as f:
        json.dump(results, f, indent=2)

    out = {"elapsed_seconds": elapsed}
    for dataset in DATASETS:
        rows = [r for r in results if r["dataset"] == dataset]
        log2w = np.array([np.log2(r["width"]) for r in rows])
        ece = np.array([r["ece_test"] for r in rows])
        acc = np.array([r["acc_test"] for r in rows])
        rho, p = stats.spearmanr(log2w, ece)
        ece_by_width = {w: float(np.mean([r["ece_test"] for r in rows if r["width"] == w])) for w in WIDTHS_EXT}
        acc_by_width = {w: float(np.mean([r["acc_test"] for r in rows if r["width"] == w])) for w in WIDTHS_EXT}
        out[dataset] = dict(
            spearman_128_to_1024=float(rho), p=float(p),
            ece_by_width=ece_by_width, acc_by_width=acc_by_width,
        )

    with open("extend_width_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
