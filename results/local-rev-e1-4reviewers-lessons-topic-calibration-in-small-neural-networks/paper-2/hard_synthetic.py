"""
Revision-round follow-up experiment, prompted by all four round-2 reviewers
independently asking the same question: is the synthetic task's near-total
absence of a crossover (11/15 'never touches overconfidence') a property of
the *phenomenon*, or just of this one easy/low-noise `make_classification`
configuration? Given the original 240-run sweep only took 267s, this is a
cheap, decisive follow-up: rerun the synthetic side with a harder
configuration (lower class_sep, higher flip_y) at the same 15 seeds x 8
widths, and compare crossover existence rates.

Not pre-registered -- explicitly exploratory, reported as such in the paper.
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


def get_hard_synthetic(seed):
    X, y = make_classification(
        n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=2, class_sep=0.5, flip_y=0.15,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def run_cell(width, seed):
    X_train, X_test, y_train, y_test = get_hard_synthetic(seed)
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
            print(f"hard_synthetic seed={seed} width={width} test_acc={cell['test_acc']:.3f} "
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

    # compare existence rate to original synthetic (4 defined / 11 undefined, n=15)
    table = [[len(defined), none_count], [4, 11]]
    odds_ratio, p = fisher_exact(table)

    bias_by_width = {}
    for w in WIDTHS:
        vals = [r["bias"] for r in results if r["width"] == w]
        bias_by_width[w] = {"mean": float(np.mean(vals)), "n": len(vals)}

    out = {
        "config": {"class_sep": 0.5, "flip_y": 0.15, "n_samples": 2000},
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
        "fisher_exact_vs_original_synthetic": {
            "table": table, "odds_ratio": float(odds_ratio), "p_value": float(p),
            "note": "rows: [hard_synthetic defined/undefined], [original_synthetic 4/11]",
        },
        "elapsed_seconds": elapsed,
    }
    with open("hard_synthetic_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved hard_synthetic_results.json")
    print(f"Defined crossovers: {len(defined)}/15, never_touch={never_touch}, touch_revert={touch_revert}")
    print(f"Fisher exact vs original synthetic: OR={odds_ratio:.3f} p={p:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
