"""
Agent-based simulation of human-AI reliance dynamics.

Models a population of N simulated "human" decision-makers who each face a
sequence of T binary decisions, receive advice from a shared AI advisor of
fixed reliability, and adapt a scalar "trust" state using one of three
trust-update heuristics. We measure team accuracy, over-/under-reliance,
and complementarity (team accuracy vs. the better of human-alone / AI-alone)
across a factorial grid of AI reliability x advice transparency x heuristic,
plus a secondary ablation on human population heterogeneity.

Deterministic given a seed; pure numpy, no external data or network access.
"""
import numpy as np
import pandas as pd
from scipy import stats
import json
import time
import hashlib

RNG_GLOBAL_SEED = 12345

def stable_hash(*parts):
    """Deterministic replacement for Python's hash(), which is randomized
    per-process for strings (PYTHONHASHSEED) and therefore NOT reproducible
    across runs despite fixing RNG_GLOBAL_SEED."""
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest(), 16) % 10_000_000

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def run_one(seed, N, T, p_ai, transparent, heuristic, hetero, burn_in=None,
            p_h_mean=0.65, p_h_sd=0.12, eta_up=0.08, eta_down=0.24,
            conf_beta=1.5):
    """Run one simulated population trajectory. Returns dict of steady-state metrics."""
    rng = np.random.default_rng(seed)
    if burn_in is None:
        burn_in = T // 2  # steady-state window = second half

    # human ability per agent
    if hetero:
        # Beta shaped to have mean p_h_mean, sd p_h_sd
        m, s2 = p_h_mean, p_h_sd ** 2
        a = m * (m * (1 - m) / s2 - 1)
        b = (1 - m) * (m * (1 - m) / s2 - 1)
        p_h = rng.beta(a, b, size=N)
        p_h = np.clip(p_h, 0.05, 0.95)
    else:
        p_h = np.full(N, p_h_mean)

    trust = np.full(N, 0.5)
    a_cnt = np.full(N, 2.0)  # Bayesian prior pseudo-counts
    b_cnt = np.full(N, 2.0)

    correct_hist = np.zeros((T, N), dtype=bool)
    comply_hist = np.zeros((T, N), dtype=bool)
    ai_correct_hist = np.zeros((T, N), dtype=bool)
    human_correct_hist = np.zeros((T, N), dtype=bool)

    for t in range(T):
        y = rng.integers(0, 2, size=N)
        h_flip = rng.random(N) >= p_h
        h_signal = np.where(h_flip, 1 - y, y)
        ai_flip = rng.random(N) >= p_ai
        ai_signal = np.where(ai_flip, 1 - y, y)
        ai_correct = (ai_signal == y)
        h_correct = (h_signal == y)

        if transparent:
            conf = np.where(ai_correct,
                             rng.beta(4 * conf_beta, 1 * conf_beta, size=N),
                             rng.beta(1 * conf_beta, 4 * conf_beta, size=N))
            comply_prob = sigmoid(logit(trust) + 2.0 * (conf - 0.5))
        else:
            comply_prob = trust

        comply = rng.random(N) < comply_prob
        decision = np.where(comply, ai_signal, h_signal)
        correct = (decision == y)

        correct_hist[t] = correct
        comply_hist[t] = comply
        ai_correct_hist[t] = ai_correct
        human_correct_hist[t] = h_correct

        # trust update based on observed AI correctness this round (delayed feedback assumption)
        if heuristic == "bayesian":
            a_cnt = a_cnt + ai_correct.astype(float)
            b_cnt = b_cnt + (~ai_correct).astype(float)
            trust = a_cnt / (a_cnt + b_cnt)
        elif heuristic == "asymmetric":
            trust = np.where(ai_correct,
                              trust + eta_up * (1 - trust),
                              trust - eta_down * trust)
        elif heuristic == "fixed":
            pass  # no update
        else:
            raise ValueError(heuristic)

    # steady-state window
    sl = slice(burn_in, T)
    team_acc = correct_hist[sl].mean()
    human_acc = human_correct_hist[sl].mean()
    ai_acc = ai_correct_hist[sl].mean()
    comply_rate = comply_hist[sl].mean()
    ai_wrong_mask = ~ai_correct_hist[sl]
    ai_right_mask = ai_correct_hist[sl]
    over_reliance = comply_hist[sl][ai_wrong_mask].mean() if ai_wrong_mask.any() else np.nan
    under_reliance = (~comply_hist[sl][ai_right_mask]).mean() if ai_right_mask.any() else np.nan
    final_trust = trust.mean()

    return dict(
        team_acc=team_acc, human_acc=human_acc, ai_acc=ai_acc,
        comply_rate=comply_rate, over_reliance=over_reliance,
        under_reliance=under_reliance, final_trust=final_trust,
        complementarity=team_acc - max(human_acc, ai_acc),
    )


def main():
    t0 = time.time()
    N, T = 200, 300
    n_seeds = 30
    reliabilities = [0.65, 0.75, 0.85, 0.95]
    transparency_levels = [False, True]
    heuristics = ["bayesian", "asymmetric", "fixed"]

    rows = []
    for p_ai in reliabilities:
        for transparent in transparency_levels:
            for heuristic in heuristics:
                for s in range(n_seeds):
                    seed = RNG_GLOBAL_SEED + stable_hash(p_ai, transparent, heuristic, s)
                    res = run_one(seed=seed, N=N, T=T, p_ai=p_ai, transparent=transparent,
                                  heuristic=heuristic, hetero=False)
                    res.update(p_ai=p_ai, transparent=transparent, heuristic=heuristic, seed=s)
                    rows.append(res)
    df_main = pd.DataFrame(rows)
    df_main.to_csv("results_main_grid.csv", index=False)
    print(f"[main grid] {len(df_main)} runs, elapsed {time.time()-t0:.1f}s")

    # Secondary ablation: population heterogeneity (homogeneous vs heterogeneous ability)
    # crossed with AI reliability, for both heuristics that adapt (bayesian, asymmetric),
    # under transparent condition, same seed count as main grid.
    rows2 = []
    for p_ai in reliabilities:
        for hetero in [False, True]:
            for heuristic in ["bayesian", "asymmetric"]:
                for s in range(n_seeds):
                    seed = RNG_GLOBAL_SEED + 777 + stable_hash(p_ai, hetero, heuristic, s)
                    res = run_one(seed=seed, N=N, T=T, p_ai=p_ai, transparent=True,
                                  heuristic=heuristic, hetero=hetero)
                    res.update(p_ai=p_ai, hetero=hetero, heuristic=heuristic, seed=s)
                    rows2.append(res)
    df_hetero = pd.DataFrame(rows2)
    df_hetero.to_csv("results_hetero_ablation.csv", index=False)
    print(f"[hetero ablation] {len(df_hetero)} runs, elapsed {time.time()-t0:.1f}s")

    # ---- Statistical tests on main grid ----
    # 1) Complementarity: team_acc vs max(human_acc, ai_acc), paired across seeds, per cell
    test_rows = []
    cells = df_main.groupby(["p_ai", "transparent", "heuristic"])
    for (p_ai, transparent, heuristic), g in cells:
        diffs = g["complementarity"].values
        tstat, pval = stats.ttest_1samp(diffs, 0.0)
        test_rows.append(dict(p_ai=p_ai, transparent=transparent, heuristic=heuristic,
                               mean_complementarity=diffs.mean(), sd=diffs.std(ddof=1),
                               t=tstat, p=pval, n=len(diffs)))
    df_tests = pd.DataFrame(test_rows)
    n_tests = len(df_tests)
    df_tests["p_bonferroni"] = np.minimum(df_tests["p"] * n_tests, 1.0)
    df_tests.to_csv("results_complementarity_tests.csv", index=False)
    print(f"[complementarity tests] {n_tests} cells tested")

    # 2) Transparency effect on team_acc within asymmetric heuristic, paired by seed, per reliability
    transp_rows = []
    for p_ai in reliabilities:
        sub = df_main[(df_main.heuristic == "asymmetric") & (df_main.p_ai == p_ai)]
        opaque = sub[sub.transparent == False].sort_values("seed")["team_acc"].values
        transp = sub[sub.transparent == True].sort_values("seed")["team_acc"].values
        tstat, pval = stats.ttest_rel(transp, opaque)
        wstat, wpval = stats.wilcoxon(transp, opaque)
        transp_rows.append(dict(p_ai=p_ai, mean_diff=(transp - opaque).mean(),
                                 t=tstat, p_ttest=pval, p_wilcoxon=wpval, n=len(opaque)))
    df_transp = pd.DataFrame(transp_rows)
    df_transp["p_ttest_bonf"] = np.minimum(df_transp["p_ttest"] * len(df_transp), 1.0)
    df_transp.to_csv("results_transparency_asymmetric.csv", index=False)
    print("[transparency x asymmetric] tests done")

    # 3) Power analysis: transparency effect on team_acc, asymmetric heuristic, p_ai=0.75
    #    sweep n_seeds in {10,20,40,80,160}, 8 independent replicate batches each,
    #    report fraction reaching p<0.05 (paired t-test)
    power_rows = []
    seed_counts = [10, 20, 40, 80, 160]
    n_replicates = 8
    for n_s in seed_counts:
        sig_count = 0
        for rep in range(n_replicates):
            base = RNG_GLOBAL_SEED + 999999 * (rep + 1)
            opaque_vals = []
            transp_vals = []
            for s in range(n_s):
                seed_o = base + stable_hash("o", s)
                seed_t = base + stable_hash("t", s)
                ro = run_one(seed=seed_o, N=N, T=T, p_ai=0.75, transparent=False,
                             heuristic="asymmetric", hetero=False)
                rt = run_one(seed=seed_t, N=N, T=T, p_ai=0.75, transparent=True,
                             heuristic="asymmetric", hetero=False)
                opaque_vals.append(ro["team_acc"])
                transp_vals.append(rt["team_acc"])
            tstat, pval = stats.ttest_rel(np.array(transp_vals), np.array(opaque_vals))
            if pval < 0.05:
                sig_count += 1
        power_rows.append(dict(n_seeds=n_s, n_replicates=n_replicates,
                                fraction_significant=sig_count / n_replicates))
    df_power = pd.DataFrame(power_rows)
    df_power.to_csv("results_power_analysis.csv", index=False)
    print(f"[power analysis] done, elapsed {time.time()-t0:.1f}s")

    # ---- Summary JSON for paper writing ----
    summary = {
        "elapsed_sec": time.time() - t0,
        "main_grid_n_runs": len(df_main),
        "hetero_ablation_n_runs": len(df_hetero),
        "grid_means": df_main.groupby(["p_ai", "transparent", "heuristic"])[
            ["team_acc", "human_acc", "ai_acc", "over_reliance", "under_reliance",
             "final_trust", "complementarity"]].mean().reset_index().to_dict(orient="records"),
        "complementarity_tests": df_tests.to_dict(orient="records"),
        "transparency_asymmetric_tests": df_transp.to_dict(orient="records"),
        "power_analysis": df_power.to_dict(orient="records"),
        "hetero_grid_means": df_hetero.groupby(["p_ai", "hetero", "heuristic"])[
            ["team_acc", "human_acc", "ai_acc", "complementarity"]].mean().reset_index().to_dict(orient="records"),
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"TOTAL elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
