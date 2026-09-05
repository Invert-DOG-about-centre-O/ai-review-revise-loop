"""
Calibration vs. width in small MLPs: cross-dataset generalization check.

Pre-registered protocol (decided BEFORE any results were inspected):
  - Two datasets: sklearn `digits` (8x8 images, 10 classes, real data) and a
    synthetic 2D `make_classification` task (2000 samples, 2 classes).
  - Widths (single hidden layer MLP): [2,4,8,16,32,64,128,256].
  - Seeds: fixed integer seeds 0..14 (15 seeds), used deterministically for
    BOTH the train/test split and the MLP initialization for that
    (dataset, width, seed) cell -- each cell is an independently reseeded run
    (no shared RNG stream across widths, per project lessons).
  - Metrics per cell: test accuracy, train accuracy, 10-bin ECE, and
    confidence bias = mean(max softmax prob on test) - test accuracy.
    Positive bias = overconfident, negative = underconfident.
  - Per-seed "crossover width" = smallest width at which confidence bias
    becomes positive (>=0) and stays non-negative for all larger widths
    tested for that seed; NaN if no such width exists in the grid.
  - CONFIRMATORY TEST (pre-registered, run exactly once, no peeking):
    Mann-Whitney U test comparing the per-seed crossover-width distributions
    of the two datasets. Alpha = 0.05. This is the only significance test
    on the primary endpoint; no sample-size extension is permitted after
    seeing results.
  - MECHANISM CHECK: test whether crossover aligns with train-accuracy
    saturation (>=99% train accuracy) by directly comparing, per seed, the
    smallest width reaching >=99% train acc vs. the crossover width.

All numbers are saved to results.json for the paper to cite.
"""
import json, time, sys
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.stats import mannwhitneyu

t0 = time.time()

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
SEEDS = list(range(15))  # fixed, pre-registered, no hash()
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
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    bias = conf.mean() - correct.mean()
    return ece, bias


def get_dataset(name, seed):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
        X = X / 16.0
    elif name == "synthetic":
        X, y = make_classification(
            n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
            n_clusters_per_class=2, class_sep=1.0, flip_y=0.05,
            random_state=seed,
        )
    else:
        raise ValueError(name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def run_cell(dataset_name, width, seed):
    X_train, X_test, y_train, y_test = get_dataset(dataset_name, seed)
    clf = MLPClassifier(
        hidden_layer_sizes=(width,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed,
        early_stopping=False,
        alpha=1e-4,
    )
    clf.fit(X_train, y_train)
    train_probs = clf.predict_proba(X_train)
    test_probs = clf.predict_proba(X_test)
    train_acc = (train_probs.argmax(axis=1) == y_train).mean()
    test_acc = (test_probs.argmax(axis=1) == y_test).mean()
    ece, bias = ece_and_bias(test_probs, y_test)
    return dict(
        dataset=dataset_name, width=width, seed=seed,
        train_acc=float(train_acc), test_acc=float(test_acc),
        ece=float(ece), bias=float(bias),
    )


def main():
    results = []
    for dataset_name in ["digits", "synthetic"]:
        for seed in SEEDS:
            for width in WIDTHS:
                cell = run_cell(dataset_name, width, seed)
                results.append(cell)
                print(f"{dataset_name} seed={seed} width={width} "
                      f"test_acc={cell['test_acc']:.3f} bias={cell['bias']:+.3f} "
                      f"ece={cell['ece']:.3f}", flush=True)
    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f}s", flush=True)

    # --- crossover width per (dataset, seed) ---
    crossovers = {"digits": {}, "synthetic": {}}
    train_sat_width = {"digits": {}, "synthetic": {}}
    for dataset_name in ["digits", "synthetic"]:
        for seed in SEEDS:
            rows = sorted(
                [r for r in results if r["dataset"] == dataset_name and r["seed"] == seed],
                key=lambda r: r["width"],
            )
            biases = [r["bias"] for r in rows]
            cross = None
            for i, w in enumerate(WIDTHS):
                if all(b >= 0 for b in biases[i:]):
                    cross = w
                    break
            crossovers[dataset_name][seed] = cross

            sat = None
            for r in rows:
                if r["train_acc"] >= 0.99:
                    sat = r["width"]
                    break
            train_sat_width[dataset_name][seed] = sat

    # --- confirmatory test: Mann-Whitney U on crossover widths, digits vs synthetic ---
    digits_cross = [v for v in crossovers["digits"].values() if v is not None]
    synth_cross = [v for v in crossovers["synthetic"].values() if v is not None]
    mw_result = None
    if len(digits_cross) >= 2 and len(synth_cross) >= 2:
        stat, p = mannwhitneyu(digits_cross, synth_cross, alternative="two-sided")
        mw_result = {"statistic": float(stat), "p_value": float(p),
                      "n_digits": len(digits_cross), "n_synthetic": len(synth_cross)}

    # --- mechanism check: does crossover width == train-saturation width? ---
    mech_agree = {"digits": 0, "synthetic": 0}
    mech_total = {"digits": 0, "synthetic": 0}
    mech_pairs = {"digits": [], "synthetic": []}
    for dataset_name in ["digits", "synthetic"]:
        for seed in SEEDS:
            c = crossovers[dataset_name][seed]
            s = train_sat_width[dataset_name][seed]
            if c is not None and s is not None:
                mech_total[dataset_name] += 1
                mech_pairs[dataset_name].append({"seed": seed, "crossover": c, "train_sat": s})
                if c == s:
                    mech_agree[dataset_name] += 1

    # effect size: Cohen's d style via rank-biserial from Mann-Whitney, plus simple median/IQR summary
    def summary(vals):
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return {"n": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "min": float(arr.min()), "max": float(arr.max()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}

    out = {
        "widths": WIDTHS,
        "seeds": SEEDS,
        "n_bins_ece": N_BINS,
        "raw_results": results,
        "crossover_widths": crossovers,
        "crossover_summary": {
            "digits": summary(digits_cross),
            "synthetic": summary(synth_cross),
        },
        "crossover_none_count": {
            "digits": sum(1 for v in crossovers["digits"].values() if v is None),
            "synthetic": sum(1 for v in crossovers["synthetic"].values() if v is None),
        },
        "mannwhitney_crossover_digits_vs_synthetic": mw_result,
        "mechanism_check_train_saturation": {
            "agree_count": mech_agree,
            "total_paired": mech_total,
            "pairs": mech_pairs,
        },
        "elapsed_seconds": elapsed,
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results.json")
    print("Crossover summary:", json.dumps(out["crossover_summary"], indent=2))
    print("Mann-Whitney:", mw_result)
    print("Mechanism agreement:", mech_agree, "/", mech_total)


if __name__ == "__main__":
    main()
