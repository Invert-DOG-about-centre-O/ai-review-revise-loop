"""
Equal-parameter-budget comparison: ensembles of small MLPs vs a single
larger MLP, for calibration (ECE/NLL/Brier), before and after temperature
scaling, on two datasets (sklearn digits; synthetic Gaussian blobs).

Pre-registered primary hypothesis (H1): at matched total hidden-unit /
parameter budget, a K-member ensemble of small MLPs has LOWER test ECE
than a single MLP of the same total width, evaluated AFTER temperature
scaling is applied to both (temperature fit on a held-out validation split).

Pre-registered null: the paired per-seed difference
  d = ECE_single_calibrated - ECE_ensemble_calibrated
is symmetric about 0 (Wilcoxon signed-rank test, two-sided, alpha=0.05).
Primary family (pre-specified, Bonferroni-corrected at alpha=0.05/2=0.025):
  H1a: digits dataset,  H1b: synthetic-blobs dataset.
Everything else (uncalibrated deltas, NLL/Brier deltas, boundary-hit
diagnostics, mechanism check) is secondary/exploratory and uncorrected,
labelled as such in the paper.

All seeds are fixed integers derived from an explicit index (never hash()).
"""
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split

DEVICE = "cpu"
torch.set_num_threads(1)

N_SEEDS = 20
BASE_WIDTH = 32          # single-model hidden width (the "budget")
ENSEMBLE_K = 4            # ensemble size
ENSEMBLE_WIDTH = BASE_WIDTH // ENSEMBLE_K  # 8, matched total hidden units
N_BINS = 15
EPOCHS = 150
LR = 0.05
T_GRID = np.linspace(0.05, 10.0, 400)  # bounded temperature search grid
T_BOUNDARY_TOL = 0.02 * (T_GRID[-1] - T_GRID[0]) / len(T_GRID)  # tol for "at boundary"


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_mlp(X_train, y_train, in_dim, hidden, n_classes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(in_dim, hidden, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.05)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return model


def get_logits(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, y_true, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        e += (mask.sum() / total) * abs(acc_bin - conf_bin)
    return float(e)


def nll(probs, y_true):
    eps = 1e-12
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.log(p).mean())


def brier(probs, y_true, n_classes):
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(val_logits, val_y):
    """Grid search T minimizing NLL on validation logits. Returns (T, hit_boundary)."""
    best_T, best_nll = 1.0, np.inf
    for T in T_GRID:
        p = softmax(val_logits, T)
        n = nll(p, val_y)
        if n < best_nll:
            best_nll, best_T = n, T
    hit_lo = abs(best_T - T_GRID[0]) <= T_BOUNDARY_TOL
    hit_hi = abs(best_T - T_GRID[-1]) <= T_BOUNDARY_TOL
    return float(best_T), bool(hit_lo or hit_hi)


def ensemble_probs(models, X, T=None):
    all_p = []
    for m in models:
        logits = get_logits(m, X)
        t = T if T is not None else 1.0
        all_p.append(softmax(logits, t))
    return np.mean(all_p, axis=0)


def run_dataset(name, X, y, n_classes, seed_offset):
    in_dim = X.shape[1]
    results = []
    single_params = None
    ens_params = None
    for i in range(N_SEEDS):
        seed = seed_offset + i  # explicit fixed integer mapping, no hash()
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
        X_train_n = (X_train - mu) / sd
        X_val_n = (X_val - mu) / sd
        X_test_n = (X_test - mu) / sd

        # single model
        single_seed = seed * 1000 + 1  # fixed explicit offset, not hash()
        single = train_mlp(X_train_n, y_train, in_dim, BASE_WIDTH, n_classes, single_seed)
        single_params = count_params(single)
        single_val_logits = get_logits(single, X_val_n)
        single_test_logits = get_logits(single, X_test_n)

        T_single, hit_single = fit_temperature(single_val_logits, y_val)
        p_single_raw = softmax(single_test_logits, 1.0)
        p_single_cal = softmax(single_test_logits, T_single)

        # ensemble of K small models, each with its own explicit sub-seed
        ens_models = []
        for k in range(ENSEMBLE_K):
            sub_seed = seed * 1000 + 100 + k  # fixed explicit offset per member
            m = train_mlp(X_train_n, y_train, in_dim, ENSEMBLE_WIDTH, n_classes, sub_seed)
            ens_models.append(m)
        ens_params = ENSEMBLE_K * count_params(ens_models[0])

        # fit a single shared temperature for the ensemble's averaged logits proxy:
        # use averaged (uncalibrated) probs -> to calibrate an averaged-probability
        # ensemble we scale each member's logits by shared T before averaging.
        def ens_val_nll(T):
            p = ensemble_probs(ens_models, X_val_n, T)
            return nll(p, y_val)

        best_T_e, best_nll_e = 1.0, np.inf
        for T in T_GRID:
            n = ens_val_nll(T)
            if n < best_nll_e:
                best_nll_e, best_T_e = n, T
        hit_ens = bool(
            abs(best_T_e - T_GRID[0]) <= T_BOUNDARY_TOL
            or abs(best_T_e - T_GRID[-1]) <= T_BOUNDARY_TOL
        )

        p_ens_raw = ensemble_probs(ens_models, X_test_n, 1.0)
        p_ens_cal = ensemble_probs(ens_models, X_test_n, best_T_e)

        # Mechanism check (intervention): does a SEPARATE temperature per
        # ensemble member (fit on val, then averaged) close the gap versus a
        # single SHARED temperature for the whole ensemble? Tests whether the
        # shared-T bottleneck (not underfitting per se) explains the deficit.
        member_Ts = []
        for m in ens_models:
            v_logits = get_logits(m, X_val_n)
            t_m, _ = fit_temperature(v_logits, y_val)
            member_Ts.append(t_m)
        p_ens_percal_list = [
            softmax(get_logits(m, X_test_n), t_m) for m, t_m in zip(ens_models, member_Ts)
        ]
        p_ens_percal = np.mean(p_ens_percal_list, axis=0)

        row = dict(
            dataset=name,
            seed=seed,
            acc_single=float((p_single_raw.argmax(1) == y_test).mean()),
            acc_ensemble=float((p_ens_raw.argmax(1) == y_test).mean()),
            ece_single_raw=ece(p_single_raw, y_test),
            ece_single_cal=ece(p_single_cal, y_test),
            ece_ensemble_raw=ece(p_ens_raw, y_test),
            ece_ensemble_cal=ece(p_ens_cal, y_test),
            ece_ensemble_percal=ece(p_ens_percal, y_test),
            nll_single_raw=nll(p_single_raw, y_test),
            nll_single_cal=nll(p_single_cal, y_test),
            nll_ensemble_raw=nll(p_ens_raw, y_test),
            nll_ensemble_cal=nll(p_ens_cal, y_test),
            nll_ensemble_percal=nll(p_ens_percal, y_test),
            brier_single_raw=brier(p_single_raw, y_test, n_classes),
            brier_single_cal=brier(p_single_cal, y_test, n_classes),
            brier_ensemble_raw=brier(p_ens_raw, y_test, n_classes),
            brier_ensemble_cal=brier(p_ens_cal, y_test, n_classes),
            T_single=T_single,
            T_ensemble=best_T_e,
            T_single_hit_boundary=hit_single,
            T_ensemble_hit_boundary=hit_ens,
            train_nll_single=nll(softmax(get_logits(single, X_train_n)), y_train),
            train_nll_ensemble=nll(ensemble_probs(ens_models, X_train_n, 1.0), y_train),
        )
        results.append(row)
        print(f"[{name}] seed {seed} ({i+1}/{N_SEEDS}) "
              f"ECE_single_cal={row['ece_single_cal']:.4f} "
              f"ECE_ens_cal={row['ece_ensemble_cal']:.4f} "
              f"T_single={T_single:.2f} T_ens={best_T_e:.2f}")
    return results, single_params, ens_params


def permutation_test_paired(diffs, n_perm=20000, seed=12345):
    """Sign-flip permutation null for a paired difference (robustness check for Wilcoxon)."""
    rng = np.random.RandomState(seed)
    obs = np.mean(diffs)
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        stat = np.mean(diffs * signs)
        if abs(stat) >= abs(obs):
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))


def power_simulation(n=N_SEEDS, effect_sd_ratio=0.5, n_sim=3000, alpha=0.025, seed=777):
    """Simulate detectable power of paired Wilcoxon at n=N_SEEDS for a given
    standardized effect size (mean diff / sd of diffs), two-sided, alpha per
    the Bonferroni-corrected primary family (0.05/2)."""
    rng = np.random.RandomState(seed)
    rejections = 0
    for _ in range(n_sim):
        d = rng.normal(loc=effect_sd_ratio, scale=1.0, size=n)
        try:
            _, p = wilcoxon(d)
        except ValueError:
            p = 1.0
        if p < alpha:
            rejections += 1
    return rejections / n_sim


def main():
    t0 = time.time()
    all_results = []

    # Dataset 1: sklearn digits (real, small, 10-class, 64-dim)
    digits = load_digits()
    Xd, yd = digits.data.astype(np.float64), digits.target.astype(np.int64)
    res_d, sp_d, ep_d = run_dataset("digits", Xd, yd, 10, seed_offset=1000)
    all_results += res_d

    # Dataset 2: synthetic Gaussian blobs (controlled, 4-class, 20-dim, noisy)
    Xb, yb = make_blobs(
        n_samples=1800, centers=4, n_features=20, cluster_std=6.0, random_state=42
    )
    yb = yb.astype(np.int64)
    res_b, sp_b, ep_b = run_dataset("blobs", Xb, yb, 4, seed_offset=2000)
    all_results += res_b

    print(f"\nParam counts: digits single={sp_d} ensemble_total={ep_d}; "
          f"blobs single={sp_b} ensemble_total={ep_b}")

    # ---- Primary statistical tests (pre-registered, Bonferroni alpha=0.025) ----
    stats = {}
    for name in ["digits", "blobs"]:
        rows = [r for r in all_results if r["dataset"] == name]
        d_cal = np.array([r["ece_single_cal"] - r["ece_ensemble_cal"] for r in rows])
        d_raw = np.array([r["ece_single_raw"] - r["ece_ensemble_raw"] for r in rows])
        wstat, wp = wilcoxon(d_cal)
        perm_obs, perm_p = permutation_test_paired(d_cal)
        wstat_raw, wp_raw = wilcoxon(d_raw)
        n_boundary_single = sum(r["T_single_hit_boundary"] for r in rows)
        n_boundary_ens = sum(r["T_ensemble_hit_boundary"] for r in rows)
        rank_biserial = 1 - (2 * wstat) / (len(d_cal) * (len(d_cal) + 1) / 2)

        # Secondary/exploratory mechanism check: shared-T vs per-member-T ensemble
        d_percal = np.array([r["ece_single_cal"] - r["ece_ensemble_percal"] for r in rows])
        d_shared_vs_per = np.array([r["ece_ensemble_cal"] - r["ece_ensemble_percal"] for r in rows])
        try:
            wstat_pc, wp_pc = wilcoxon(d_percal)
        except ValueError:
            wstat_pc, wp_pc = float("nan"), float("nan")
        try:
            wstat_svp, wp_svp = wilcoxon(d_shared_vs_per)
        except ValueError:
            wstat_svp, wp_svp = float("nan"), float("nan")

        stats[name] = dict(
            n=len(rows),
            mean_diff_cal=float(d_cal.mean()),
            median_diff_cal=float(np.median(d_cal)),
            sd_diff_cal=float(d_cal.std(ddof=1)),
            wilcoxon_stat=float(wstat),
            wilcoxon_p=float(wp),
            rank_biserial=float(rank_biserial),
            perm_p=perm_p,
            mean_diff_raw=float(d_raw.mean()),
            wilcoxon_p_raw=float(wp_raw),
            frac_boundary_single=n_boundary_single / len(rows),
            frac_boundary_ensemble=n_boundary_ens / len(rows),
            mean_train_nll_single=float(np.mean([r["train_nll_single"] for r in rows])),
            mean_train_nll_ensemble=float(np.mean([r["train_nll_ensemble"] for r in rows])),
            mean_diff_percal=float(d_percal.mean()),
            wilcoxon_p_percal=float(wp_pc),
            mean_diff_shared_vs_per=float(d_shared_vs_per.mean()),
            wilcoxon_p_shared_vs_per=float(wp_svp),
        )
        print(f"\n[{name}] PRIMARY calibrated-ECE test: mean diff (single-ens) = "
              f"{stats[name]['mean_diff_cal']:.4f}, Wilcoxon p={wp:.5f}, "
              f"perm p={perm_p:.5f}, rank-biserial={rank_biserial:.3f}")
        print(f"[{name}] boundary-hit rate: single={stats[name]['frac_boundary_single']:.2f} "
              f"ensemble={stats[name]['frac_boundary_ensemble']:.2f}")
        print(f"[{name}] MECHANISM CHECK (secondary): single-vs-per-member-calibrated-ens "
              f"diff={d_percal.mean():.4f} p={wp_pc:.5f}; "
              f"shared-T-vs-per-member-T-ens diff={d_shared_vs_per.mean():.4f} p={wp_svp:.5f}")

    # ---- Power simulation for the primary test design (n=N_SEEDS, alpha=0.025) ----
    power_curve = {}
    for eff in [0.3, 0.5, 0.8, 1.0]:
        power_curve[eff] = power_simulation(effect_sd_ratio=eff)
    print(f"\nPower simulation (n={N_SEEDS}, alpha=0.025, standardized effect->power): {power_curve}")

    elapsed = time.time() - t0
    print(f"\nTotal experiment wall time: {elapsed:.1f}s")

    out = dict(
        config=dict(
            n_seeds=N_SEEDS, base_width=BASE_WIDTH, ensemble_k=ENSEMBLE_K,
            ensemble_width=ENSEMBLE_WIDTH, epochs=EPOCHS, n_bins=N_BINS,
            t_grid_min=float(T_GRID[0]), t_grid_max=float(T_GRID[-1]),
            single_params_digits=sp_d, ensemble_params_digits=ep_d,
            single_params_blobs=sp_b, ensemble_params_blobs=ep_b,
        ),
        per_seed_results=all_results,
        primary_stats=stats,
        power_curve={str(k): v for k, v in power_curve.items()},
        wall_time_seconds=elapsed,
    )
    with open("round1_review_1_results_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results.json")


if __name__ == "__main__":
    main()
