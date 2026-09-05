"""
Round-3 reviewers (all four, independently) flagged that the "2-class,
2-feature geometry suppresses overconfidence" claim is inferred from a single
difficulty manipulation and was never actually tested by varying dimensionality
directly (Limitations #5 in v3, "the natural next experiment given how cheap
these sweeps are"). This script runs that experiment: hold class_sep=1.0 and
flip_y=0.05 fixed at the ORIGINAL (easy) synthetic values, but raise
n_features from 2 to 10 (n_informative=10, n_clusters_per_class=1, since
sklearn requires 2**n_informative >= n_classes*n_clusters_per_class and we
want to isolate dimensionality, not clusters-per-class). Same 15 seeds, same
8 widths. If "low-dimensional/2-feature" geometry is the suppressor, going to
10-D at the SAME noise/margin as the easy original should restore more
crossovers. If not, the 2-class count itself (not dimensionality) is
implicated.

Not pre-registered -- exploratory, reported as such.
"""
import json, time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.stats import fisher_exact

t0 = time.time()
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
SEEDS = list(range(15))
N_BINS = 10


def ece_and_bias(probs, y_true, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return ece, conf.mean() - correct.mean()


def get_highdim_synthetic(seed):
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=10, n_redundant=0,
        n_clusters_per_class=1, class_sep=1.0, flip_y=0.05,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def run_cell(width, seed):
    X_train, X_test, y_train, y_test = get_highdim_synthetic(seed)
    clf = MLPClassifier(hidden_layer_sizes=(width,), activation="relu", solver="adam",
                         max_iter=500, random_state=seed, early_stopping=False, alpha=1e-4)
    clf.fit(X_train, y_train)
    train_probs = clf.predict_proba(X_train)
    test_probs = clf.predict_proba(X_test)
    train_acc = (train_probs.argmax(axis=1) == y_train).mean()
    test_acc = (test_probs.argmax(axis=1) == y_test).mean()
    ece, bias = ece_and_bias(test_probs, y_test)
    return dict(width=width, seed=seed, train_acc=float(train_acc), test_acc=float(test_acc),
                ece=float(ece), bias=float(bias))


def main():
    results = []
    for seed in SEEDS:
        for width in WIDTHS:
            cell = run_cell(width, seed)
            results.append(cell)
            print(f"highdim_synthetic seed={seed} width={width} test_acc={cell['test_acc']:.3f} "
                  f"bias={cell['bias']:+.3f} ece={cell['ece']:.3f}", flush=True)
    elapsed = time.time() - t0

    crossovers, never_touch, touch_revert = {}, 0, 0
    for seed in SEEDS:
        rows = sorted([r for r in results if r["seed"] == seed], key=lambda r: r["width"])
        biases = [r["bias"] for r in rows]
        cross = None
        for i, w in enumerate(WIDTHS):
            if all(b >= 0 for b in biases[i:]):
                cross = w
                break
        crossovers[seed] = cross
        if cross is None:
            if any(b >= 0 for b in biases):
                touch_revert += 1
            else:
                never_touch += 1

    defined = [v for v in crossovers.values() if v is not None]
    none_count = sum(1 for v in crossovers.values() if v is None)

    # compare existence rate to original (2-D) synthetic (4 defined / 11 undefined, n=15)
    table = [[len(defined), none_count], [4, 11]]
    odds_ratio, p = fisher_exact(table)

    bias_by_width = {}
    for w in WIDTHS:
        vals = [r["bias"] for r in results if r["width"] == w]
        bias_by_width[w] = {"mean": float(np.mean(vals)), "n": len(vals)}

    train_acc_256 = np.mean([r["train_acc"] for r in results if r["width"] == 256])
    test_acc_256 = np.mean([r["test_acc"] for r in results if r["width"] == 256])

    out = {
        "config": {"n_features": 10, "n_informative": 10, "n_clusters_per_class": 1,
                    "class_sep": 1.0, "flip_y": 0.05, "n_samples": 2000},
        "widths": WIDTHS, "seeds": SEEDS,
        "raw_results": results,
        "crossover_widths": crossovers,
        "crossover_defined_count": len(defined),
        "crossover_none_count": none_count,
        "never_touch_count": never_touch,
        "touch_revert_count": touch_revert,
        "crossover_summary": {
            "n": len(defined),
            "mean": float(np.mean(defined)) if defined else None,
            "median": float(np.median(defined)) if defined else None,
        },
        "bias_by_width": bias_by_width,
        "mean_train_acc_width256": float(train_acc_256),
        "mean_test_acc_width256": float(test_acc_256),
        "fisher_exact_vs_original_2d_synthetic": {
            "table": table, "odds_ratio": float(odds_ratio), "p_value": float(p),
            "note": "rows: [highdim(10-D) defined/undefined], [original 2-D synthetic 4/11]",
        },
        "elapsed_seconds": elapsed,
    }
    with open("geometry_ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved geometry_ablation_results.json")
    print(f"Defined crossovers: {len(defined)}/15, never_touch={never_touch}, touch_revert={touch_revert}")
    print(f"Fisher exact vs original 2-D synthetic: OR={odds_ratio:.3f} p={p:.4f}")
    print(f"Mean train/test acc at width 256: {train_acc_256:.4f}/{test_acc_256:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
