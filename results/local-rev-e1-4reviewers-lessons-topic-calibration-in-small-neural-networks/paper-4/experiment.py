"""
Calibration vs. width in small MLPs, with post-hoc temperature scaling,
across three binary-classification datasets (two synthetic, one real).

Pre-registered analysis plan (decided BEFORE looking at results):
  H1: ECE_pre (before temperature scaling) is positively correlated with
      log2(width), tested per-dataset with Spearman rank correlation.
      Null: permutation null that shuffles the width label among the
      (width, ECE_pre) pairs actually observed (fixed grid of widths used
      in this study), 20000 permutations, two-sided.
  H2: Temperature scaling reduces ECE (paired, same trained model,
      same test set): Wilcoxon signed-rank test on (ECE_pre, ECE_post)
      pairs, pooled within each dataset (n = n_widths * n_seeds).
  H3 (secondary): ECE_post (NOT the pre-post delta, to avoid the
      mechanical pre/delta correlation trap) is still correlated with
      log2(width) -- i.e. does temperature scaling equalize calibration
      across capacities or just shift it down uniformly? Same permutation
      null design as H1.

Generalization is only claimed if H1 (or H3) hold in at least 2 of 3
datasets in the same direction (pre-registered bar, see lessons.md).

Seeding: every individually reported trained model gets an explicit
integer seed computed as dataset_idx*100000 + width_idx*1000 + rep_idx,
fixed at data-generation and model-init time. No hash()-based seeding.
"""
import json
import time
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WIDTHS = [2, 4, 8, 16, 32, 64, 128]
N_SEEDS = 10
N_BINS = 15
EPS = 1e-6
N_PERM = 20000

DATASETS = ["moons", "circles", "breast_cancer"]


def get_dataset(name, seed):
    if name == "moons":
        X, y = make_moons(n_samples=600, noise=0.30, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=600, noise=0.15, factor=0.5, random_state=seed)
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError(name)
    return X, y


def ece_binary(y_true, p1, n_bins=N_BINS):
    """Expected calibration error for binary probs p1 = P(y=1)."""
    conf = np.maximum(p1, 1 - p1)
    pred = (p1 >= 0.5).astype(int)
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
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def fit_temperature(y_val, p1_val, n_grid=400):
    """Fit scalar temperature T minimizing NLL on validation logits via 1D grid search
    over log(T), refined with a local scan. p1_val are pre-scaling P(y=1)."""
    p1c = np.clip(p1_val, EPS, 1 - EPS)
    z = np.log(p1c / (1 - p1c))  # binary logit
    logT_grid = np.linspace(np.log(0.05), np.log(20.0), n_grid)

    def nll(logT):
        T = np.exp(logT)
        zt = z / T
        p = 1 / (1 + np.exp(-zt))
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

    vals = [nll(lt) for lt in logT_grid]
    best_idx = int(np.argmin(vals))
    best_logT = logT_grid[best_idx]
    # local refine
    lo = logT_grid[max(0, best_idx - 1)]
    hi = logT_grid[min(len(logT_grid) - 1, best_idx + 1)]
    fine_grid = np.linspace(lo, hi, 200)
    fine_vals = [nll(lt) for lt in fine_grid]
    best_logT = fine_grid[int(np.argmin(fine_vals))]
    return float(np.exp(best_logT))


def apply_temperature(p1, T):
    p1c = np.clip(p1, EPS, 1 - EPS)
    z = np.log(p1c / (1 - p1c))
    zt = z / T
    p = 1 / (1 + np.exp(-zt))
    return np.clip(p, EPS, 1 - EPS)


def run_one(dataset, width, seed):
    X, y = get_dataset(dataset, seed)
    # split: train / val (for temperature fitting) / test
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed + 1, stratify=y_rest
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(width,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=2000,
        random_state=seed,
        early_stopping=False,
    )
    clf.fit(X_train, y_train)

    p1_val = clf.predict_proba(X_val)[:, 1]
    p1_test = clf.predict_proba(X_test)[:, 1]

    acc_test = float(((p1_test >= 0.5).astype(int) == y_test).mean())
    ece_pre = ece_binary(y_test, p1_test)

    T = fit_temperature(y_val, p1_val)
    p1_test_scaled = apply_temperature(p1_test, T)
    ece_post = ece_binary(y_test, p1_test_scaled)

    return dict(
        dataset=dataset, width=width, seed=seed, T=T,
        acc_test=acc_test, ece_pre=ece_pre, ece_post=ece_post,
        n_test=len(y_test),
    )


def permutation_test_corr(x, y, n_perm=N_PERM, rng=None):
    """Two-sided permutation test for Spearman correlation, shuffling y."""
    if rng is None:
        rng = np.random.default_rng(12345)
    obs = stats.spearmanr(x, y).correlation
    n = len(x)
    perm_corrs = np.empty(n_perm)
    y_arr = np.array(y)
    for i in range(n_perm):
        y_perm = rng.permutation(y_arr)
        perm_corrs[i] = stats.spearmanr(x, y_perm).correlation
    p = float(np.mean(np.abs(perm_corrs) >= abs(obs)))
    return float(obs), p, perm_corrs


def main():
    t0 = time.time()
    results = []
    for di, dataset in enumerate(DATASETS):
        for wi, width in enumerate(WIDTHS):
            for ri in range(N_SEEDS):
                seed = di * 100000 + wi * 1000 + ri
                res = run_one(dataset, width, seed)
                res["seed_used"] = seed
                results.append(res)
    elapsed = time.time() - t0
    print(f"Total training time: {elapsed:.1f}s for {len(results)} runs")

    with open("raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Analysis ----
    analysis = {"elapsed_seconds": elapsed, "n_runs": len(results)}
    per_dataset = {}
    for dataset in DATASETS:
        rows = [r for r in results if r["dataset"] == dataset]
        log2w = np.array([np.log2(r["width"]) for r in rows])
        ece_pre = np.array([r["ece_pre"] for r in rows])
        ece_post = np.array([r["ece_post"] for r in rows])
        T_vals = np.array([r["T"] for r in rows])
        acc = np.array([r["acc_test"] for r in rows])

        rng = np.random.default_rng(777 + DATASETS.index(dataset))
        h1_corr, h1_p, _ = permutation_test_corr(log2w, ece_pre, rng=rng)
        rng2 = np.random.default_rng(888 + DATASETS.index(dataset))
        h3_corr, h3_p, _ = permutation_test_corr(log2w, ece_post, rng=rng2)

        wstat, wp = stats.wilcoxon(ece_pre, ece_post, alternative="greater")
        # effect size: matched-pairs rank-biserial
        diffs = ece_pre - ece_post
        n_nonzero = np.sum(diffs != 0)
        pos = np.sum(diffs > 0)
        neg = np.sum(diffs < 0)
        rank_biserial = (pos - neg) / n_nonzero if n_nonzero > 0 else float("nan")

        per_dataset[dataset] = dict(
            n=len(rows),
            mean_acc=float(acc.mean()),
            mean_ece_pre=float(ece_pre.mean()),
            mean_ece_post=float(ece_post.mean()),
            mean_T=float(T_vals.mean()),
            H1_spearman_log2width_vs_ece_pre=h1_corr,
            H1_permutation_p=h1_p,
            H3_spearman_log2width_vs_ece_post=h3_corr,
            H3_permutation_p=h3_p,
            H2_wilcoxon_stat=float(wstat),
            H2_wilcoxon_p_greater=float(wp),
            H2_rank_biserial=float(rank_biserial),
            ece_pre_by_width={
                w: float(np.mean([r["ece_pre"] for r in rows if r["width"] == w]))
                for w in WIDTHS
            },
            ece_post_by_width={
                w: float(np.mean([r["ece_post"] for r in rows if r["width"] == w]))
                for w in WIDTHS
            },
        )

    analysis["per_dataset"] = per_dataset

    # Generalization bar: H1 direction consistent (positive corr, p<0.05) in >=2/3 datasets
    h1_sig_positive = [
        d for d in DATASETS
        if per_dataset[d]["H1_spearman_log2width_vs_ece_pre"] > 0
        and per_dataset[d]["H1_permutation_p"] < 0.05
    ]
    h3_sig_positive = [
        d for d in DATASETS
        if per_dataset[d]["H3_spearman_log2width_vs_ece_post"] > 0
        and per_dataset[d]["H3_permutation_p"] < 0.05
    ]
    analysis["H1_generalizes"] = len(h1_sig_positive) >= 2
    analysis["H1_sig_datasets"] = h1_sig_positive
    analysis["H3_generalizes"] = len(h3_sig_positive) >= 2
    analysis["H3_sig_datasets"] = h3_sig_positive

    # Power sanity check: minimum detectable Spearman rho at n=70, alpha=0.05, power=0.8 (approx via Fisher z)
    n_per_dataset = len(WIDTHS) * N_SEEDS
    analysis["n_per_dataset"] = n_per_dataset

    with open("analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
