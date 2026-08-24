"""
Simulation of algorithmic monoculture and outcome homogenization in
repeated-application decision settings (e.g., hiring), plus a mitigation
via randomized-lottery decorrelation.

All randomness is seeded via numpy's Generator (PCG64) with integer seeds,
which is process-stable (no reliance on Python's randomized hash()).
"""
import numpy as np
import json
import time
from scipy import stats

N = 3000          # applicant pool size
DEFAULT_K = 8      # number of institutions
DEFAULT_P = 0.20   # fraction accepted per institution
DEFAULT_SIGMA = 1.0
N_SEEDS_MAIN = 30
N_SEEDS_ABLATION = 30


def draw_noise(rng, size, dist="gaussian"):
    if dist == "gaussian":
        return rng.standard_normal(size)
    elif dist == "student_t":
        # Student-t df=3, heavy-tailed, variance-normalized to 1
        x = rng.standard_t(df=3, size=size)
        return x / np.sqrt(3 / (3 - 2))  # scale to unit variance
    else:
        raise ValueError(dist)


def run_trial(seed, K=DEFAULT_K, rho=0.5, p=DEFAULT_P, sigma=DEFAULT_SIGMA,
              dist="gaussian", eta=0.0):
    """
    One trial: N applicants, K institutions.
    score_k = skill + sigma*( sqrt(rho)*shared_noise + sqrt(1-rho)*idio_noise_k )
    Institution accepts top-p fraction by its own score.
    With probability eta, an individual's decision at a given institution is
    instead replaced by an independent Bernoulli(p) lottery draw (decorrelation
    mitigation).
    Returns dict of per-trial arrays needed for metrics.
    """
    rng = np.random.default_rng(seed)
    skill = rng.standard_normal(N)
    shared_noise = draw_noise(rng, N, dist)

    decisions = np.zeros((K, N), dtype=bool)
    n_accept = max(1, int(round(p * N)))
    for k in range(K):
        idio = draw_noise(rng, N, dist)
        score = skill + sigma * (np.sqrt(rho) * shared_noise + np.sqrt(1 - rho) * idio)
        threshold_idx = np.argsort(-score)[:n_accept]
        accept = np.zeros(N, dtype=bool)
        accept[threshold_idx] = True
        if eta > 0:
            lottery_mask = rng.random(N) < eta
            lottery_draw = rng.random(N) < p
            accept = np.where(lottery_mask, lottery_draw, accept)
        decisions[k] = accept

    return skill, decisions


def homogenization_corr(decisions):
    """Average pairwise Pearson (phi) correlation of binary decision vectors
    across all K choose 2 institution pairs."""
    K = decisions.shape[0]
    corrs = []
    for i in range(K):
        for j in range(i + 1, K):
            a = decisions[i].astype(float)
            b = decisions[j].astype(float)
            if a.std() == 0 or b.std() == 0:
                continue
            corrs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(corrs))


def systemic_rejection_excess(decisions, p):
    """Excess systemic rejection: observed fraction rejected by ALL K
    institutions minus the independence-baseline prediction (1-p)^K."""
    K, n = decisions.shape
    reject_all = np.mean(np.all(~decisions, axis=0))
    baseline = (1 - p) ** K
    return float(reject_all - baseline), float(reject_all), float(baseline)


def accuracy_recall_at_p(skill, decisions, p):
    """Mean over institutions of recall@p: fraction of true top-p-skill
    individuals accepted by that institution."""
    n = len(skill)
    n_top = max(1, int(round(p * n)))
    true_top = np.argsort(-skill)[:n_top]
    true_top_mask = np.zeros(n, dtype=bool)
    true_top_mask[true_top] = True
    recalls = [decisions[k][true_top_mask].mean() for k in range(decisions.shape[0])]
    return float(np.mean(recalls))


def sweep_rho(rhos, K, p, dist, n_seeds, eta=0.0, seed_base=1000):
    rows = []
    for rho in rhos:
        Hs, Ss, Accs = [], [], []
        for s in range(n_seeds):
            seed = seed_base + hash_seed(rho, K, p, dist, eta, s)
            skill, decisions = run_trial(seed, K=K, rho=rho, p=p, dist=dist, eta=eta)
            Hs.append(homogenization_corr(decisions))
            excess, obs, base = systemic_rejection_excess(decisions, p)
            Ss.append(excess)
            Accs.append(accuracy_recall_at_p(skill, decisions, p))
        rows.append(dict(rho=rho, K=K, p=p, dist=dist, eta=eta,
                          H_mean=float(np.mean(Hs)), H_std=float(np.std(Hs)),
                          S_mean=float(np.mean(Ss)), S_std=float(np.std(Ss)),
                          Acc_mean=float(np.mean(Accs)), Acc_std=float(np.std(Accs)),
                          n_seeds=n_seeds))
    return rows


def hash_seed(*args):
    import hashlib
    s = "|".join(str(a) for a in args)
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def main():
    t0 = time.time()
    results = {}

    # --- Main sweep: rho in [0,1], Gaussian noise, K=8, p=0.2 ---
    rhos = [round(x, 2) for x in np.linspace(0, 1, 11)]
    results["main_gaussian"] = sweep_rho(rhos, DEFAULT_K, DEFAULT_P, "gaussian", N_SEEDS_MAIN)

    # --- Robustness: same sweep with Student-t heavy-tailed noise ---
    results["main_student_t"] = sweep_rho(rhos, DEFAULT_K, DEFAULT_P, "student_t", N_SEEDS_ABLATION)

    # --- Ablation: K (number of institutions) at rho=0.5, Gaussian ---
    results["ablation_K"] = {}
    for K in [4, 8, 16]:
        results["ablation_K"][K] = sweep_rho([0.5], K, DEFAULT_P, "gaussian", N_SEEDS_ABLATION)[0]

    # --- Ablation: selection stringency p at rho=0.5, Gaussian, K=8 ---
    results["ablation_p"] = {}
    for p in [0.1, 0.2, 0.4]:
        results["ablation_p"][p] = sweep_rho([0.5], DEFAULT_K, p, "gaussian", N_SEEDS_ABLATION)[0]

    # --- Mitigation sweep: lottery eta at rho=0.8 (high monoculture), Gaussian, K=8, p=0.2 ---
    etas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6]
    rows = []
    for eta in etas:
        rows.extend(sweep_rho([0.8], DEFAULT_K, DEFAULT_P, "gaussian", N_SEEDS_MAIN, eta=eta))
    results["mitigation_eta"] = rows

    # Also run mitigation sweep with Student-t noise for robustness of the tradeoff
    rows_t = []
    for eta in etas:
        rows_t.extend(sweep_rho([0.8], DEFAULT_K, DEFAULT_P, "student_t", N_SEEDS_ABLATION, eta=eta))
    results["mitigation_eta_student_t"] = rows_t

    # --- Statistical tests ---
    # Paired comparison H(rho=0) vs H(rho=1), Gaussian, using per-seed values (not just summary)
    def paired_seed_values(rho, K, p, dist, n_seeds, eta=0.0, seed_base=1000):
        Hs = []
        for s in range(n_seeds):
            seed = seed_base + hash_seed(rho, K, p, dist, eta, s)
            skill, decisions = run_trial(seed, K=K, rho=rho, p=p, dist=dist, eta=eta)
            Hs.append(homogenization_corr(decisions))
        return np.array(Hs)

    H0 = paired_seed_values(0.0, DEFAULT_K, DEFAULT_P, "gaussian", N_SEEDS_MAIN)
    H1 = paired_seed_values(1.0, DEFAULT_K, DEFAULT_P, "gaussian", N_SEEDS_MAIN)
    t_stat, p_val = stats.ttest_rel(H1, H0)

    # Regression / correlation of H_mean vs rho across the main sweep (Gaussian)
    rho_arr = np.array([r["rho"] for r in results["main_gaussian"]])
    H_arr = np.array([r["H_mean"] for r in results["main_gaussian"]])
    pearson_r, pearson_p = stats.pearsonr(rho_arr, H_arr)

    rho_arr_t = np.array([r["rho"] for r in results["main_student_t"]])
    H_arr_t = np.array([r["H_mean"] for r in results["main_student_t"]])
    pearson_r_t, pearson_p_t = stats.pearsonr(rho_arr_t, H_arr_t)

    # Bonferroni correction context: how many total per-cell significance tests
    # would be run if we tested each sweep point individually against rho=0 baseline
    n_total_cells = len(rhos) * 2 + 3 + 3 + len(etas) * 2  # main x2 dists + K abl + p abl + mitigation x2 dists
    bonferroni_alpha = 0.05 / n_total_cells

    results["stats"] = dict(
        paired_t_H0_vs_H1=dict(t=float(t_stat), p=float(p_val), n=N_SEEDS_MAIN,
                                mean_H0=float(H0.mean()), mean_H1=float(H1.mean())),
        pearson_rho_vs_H_gaussian=dict(r=float(pearson_r), p=float(pearson_p)),
        pearson_rho_vs_H_student_t=dict(r=float(pearson_r_t), p=float(pearson_p_t)),
        n_total_sensitivity_cells=n_total_cells,
        bonferroni_alpha=bonferroni_alpha,
    )

    # --- Empirical anchor check ---
    # Bommasani et al. (2026), "Algorithmic Monocultures in Hiring": among applicants
    # submitting 4 applications through the SAME algorithmic system, 10% were rejected
    # across all 4 -- vs. an independence baseline that accurately predicts non-algorithmic
    # (Fortune 500, non-monoculture) hiring data. We replicate their K=4 comparison point
    # and report which rho in our simulation reproduces a comparable excess-rejection ratio.
    p_anchor = 0.30  # roughly matches typical funnel accept rates; see paper text for caveat
    K_anchor = 4
    anchor_rows = sweep_rho([round(x, 2) for x in np.linspace(0, 1, 11)], K_anchor, p_anchor,
                             "gaussian", N_SEEDS_ABLATION, seed_base=5000)
    results["anchor_K4"] = anchor_rows

    results["meta"] = dict(N=N, DEFAULT_K=DEFAULT_K, DEFAULT_P=DEFAULT_P,
                            N_SEEDS_MAIN=N_SEEDS_MAIN, N_SEEDS_ABLATION=N_SEEDS_ABLATION,
                            runtime_sec=time.time() - t0)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("DONE. Runtime: %.1fs" % (time.time() - t0))
    print(json.dumps(results["stats"], indent=2))


if __name__ == "__main__":
    main()
