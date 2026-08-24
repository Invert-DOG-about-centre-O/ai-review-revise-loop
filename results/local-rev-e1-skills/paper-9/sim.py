"""
Simulation study of three probabilistic uncertainty methods for LLM outputs:
  1. Self-consistency (majority vote over k samples)
  2. Semantic entropy (cluster-entropy based selective prediction / hallucination flag)
  3. Calibration: verbalized (self-reported, overconfident) confidence vs.
     frequency-based (sample-agreement) confidence

All numbers are computed by actually running this script (deterministic,
seeded with hashlib-derived seeds, no reliance on Python's randomized hash()).
"""
import hashlib
import json
import time

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def stable_seed(*parts):
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def simulate_replicate(N, beta_a, beta_b, ks, W, over_a, over_b, noise_sd, seed):
    rng = np.random.default_rng(seed)
    p = rng.beta(beta_a, beta_b, size=N)
    # rescale/shift beta draw to have mean ~0.55, keep in (0.02,0.98)
    p = 0.55 + (p - p.mean()) * 1.0
    p = np.clip(p, 0.02, 0.98)

    max_k = max(ks)
    correct = rng.random((N, max_k)) < p[:, None]           # (N, max_k) bool
    wrong_cluster = rng.integers(1, W + 1, size=(N, max_k))  # cluster id if wrong
    cluster = np.where(correct, 0, wrong_cluster)            # 0 = correct cluster

    # verbalized confidence: overconfident, compressed, noisy function of true p
    verb_conf = sigmoid(over_a * logit(p) + over_b + rng.normal(0, noise_sd, size=N))

    out = {}
    for k in ks:
        sub = cluster[:, :k]
        # mode cluster per row (majority vote)
        maxc = sub.max() + 1
        counts = np.zeros((N, maxc), dtype=int)
        for c in range(maxc):
            counts[:, c] = (sub == c).sum(axis=1)
        mode_cluster = counts.argmax(axis=1)
        freq_conf = counts.max(axis=1) / k
        sc_correct = (mode_cluster == 0).astype(int)

        probs = counts / k
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)

        single_correct = correct[:, 0].astype(int)

        out[k] = dict(
            single_acc=single_correct.mean(),
            sc_acc=sc_correct.mean(),
            sc_correct=sc_correct,
            entropy=ent,
            freq_conf=freq_conf,
            verb_conf=verb_conf,
        )
    return out, p

def ece(conf, correct, n_bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, bins[1:-1])
    total = len(conf)
    e = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        e += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return e

def auroc_entropy(entropy, sc_correct):
    # predicting "is WRONG" from entropy (higher entropy => more likely wrong)
    y = 1 - sc_correct
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, entropy)

def run_setting(name, beta_a, beta_b, N=500, n_reps=30, ks=(1, 3, 5, 10, 20), W=4,
                 over_a=0.55, over_b=0.9, noise_sd=0.35):
    rows = []
    for rep in range(n_reps):
        seed = stable_seed("uncertainty_sim", name, rep)
        out, p = simulate_replicate(N, beta_a, beta_b, ks, W, over_a, over_b, noise_sd, seed)
        rec = {"rep": rep, "p_var": float(np.var(p)), "p_mean": float(np.mean(p))}
        for k in ks:
            rec[f"single_acc_k{k}"] = out[k]["single_acc"]
            rec[f"sc_acc_k{k}"] = out[k]["sc_acc"]
        # calibration + AUROC computed at k=10
        k_cal = 10
        rec["ece_verb"] = ece(out[k_cal]["verb_conf"], out[k_cal]["sc_correct"])
        rec["ece_freq"] = ece(out[k_cal]["freq_conf"], out[k_cal]["sc_correct"])
        rec["auroc_entropy_k10"] = auroc_entropy(out[k_cal]["entropy"], out[k_cal]["sc_correct"])
        rows.append(rec)
    return rows

def summarize(rows, ks):
    import numpy as np
    arr = {k: np.array([r[f"sc_acc_k{k}"] for r in rows]) for k in ks}
    single1 = np.array([r["single_acc_k1"] for r in rows])
    summary = {"p_var_mean": float(np.mean([r["p_var"] for r in rows]))}
    for k in ks:
        summary[f"sc_acc_k{k}_mean"] = float(arr[k].mean())
        summary[f"sc_acc_k{k}_std"] = float(arr[k].std(ddof=1))
    summary["single_acc_mean"] = float(single1.mean())
    # paired t-test: self-consistency at max k vs k=1 (single-sample)
    maxk = max(ks)
    t, pval = stats.ttest_rel(arr[maxk], single1)
    summary["ttest_sc_vs_single_t"] = float(t)
    summary["ttest_sc_vs_single_p"] = float(pval)
    summary["gain_pp_at_maxk"] = float((arr[maxk] - single1).mean() * 100)
    ece_v = np.array([r["ece_verb"] for r in rows])
    ece_f = np.array([r["ece_freq"] for r in rows])
    t2, p2 = stats.ttest_rel(ece_v, ece_f)
    summary["ece_verb_mean"] = float(ece_v.mean())
    summary["ece_freq_mean"] = float(ece_f.mean())
    summary["ttest_ece_verb_vs_freq_t"] = float(t2)
    summary["ttest_ece_verb_vs_freq_p"] = float(p2)
    auc = np.array([r["auroc_entropy_k10"] for r in rows])
    summary["auroc_entropy_mean"] = float(np.nanmean(auc))
    summary["auroc_entropy_std"] = float(np.nanstd(auc, ddof=1))
    return summary

def power_analysis(beta_a, beta_b, ns, n_reps=8, ks=(1, 20), W=4,
                    over_a=0.55, over_b=0.9, noise_sd=0.35, tag="power"):
    results = {}
    for N in ns:
        sig_count = 0
        for rep in range(n_reps):
            seed = stable_seed(tag, beta_a, beta_b, N, rep)
            out, p = simulate_replicate(N, beta_a, beta_b, ks, W, over_a, over_b, noise_sd, seed)
            # bootstrap-free: use per-question paired test via McNemar-ish approx:
            # treat replicate itself as one draw; instead run many small reps and paired t-test across them
            sig_count += 0  # placeholder, real test done via replicate-level below
        results[N] = None
    return results

if __name__ == "__main__":
    t0 = time.time()
    ks = (1, 3, 5, 10, 20)

    settings = {
        "low_heterogeneity":      dict(beta_a=20, beta_b=20),   # near-homogeneous difficulty
        "moderate_heterogeneity": dict(beta_a=4, beta_b=4),     # unimodal, moderate spread
        "mild_heterogeneity":     dict(beta_a=1, beta_b=1),     # uniform spread
        "extreme_heterogeneity":  dict(beta_a=0.4, beta_b=0.4), # bimodal (easy/hard mix)
    }

    all_results = {}
    for name, params in settings.items():
        rows = run_setting(name, N=500, n_reps=30, ks=ks, **params)
        summary = summarize(rows, ks)
        all_results[name] = {"summary": summary, "raw_rows": rows}
        print(f"=== {name} (p_var={summary['p_var_mean']:.4f}) ===")
        print(f"  single-sample acc: {summary['single_acc_mean']:.4f}")
        for k in ks:
            print(f"  self-consistency acc @k={k}: {summary[f'sc_acc_k{k}_mean']:.4f} "
                  f"(std {summary[f'sc_acc_k{k}_std']:.4f})")
        print(f"  gain @k=20 vs k=1: {summary['gain_pp_at_maxk']:.2f} pp, "
              f"paired t={summary['ttest_sc_vs_single_t']:.2f}, p={summary['ttest_sc_vs_single_p']:.3e}")
        print(f"  ECE verbalized-conf: {summary['ece_verb_mean']:.4f} vs "
              f"ECE frequency-conf: {summary['ece_freq_mean']:.4f}  "
              f"(paired t={summary['ttest_ece_verb_vs_freq_t']:.2f}, p={summary['ttest_ece_verb_vs_freq_p']:.3e})")
        print(f"  semantic-entropy AUROC (predicting wrong answer, k=10): "
              f"{summary['auroc_entropy_mean']:.4f} (std {summary['auroc_entropy_std']:.4f})")
        print()

    # --- Power analysis: how many replicate "runs" (of N=500 questions each) are needed
    # for the self-consistency gain (k=20 vs k=1) to be significant at alpha=0.05,
    # matched seed-count/procedure across N sweep (moderate-heterogeneity setting). ---
    print("=== Power analysis: self-consistency gain significance vs. #replicates (moderate het.) ===")
    power_rows = {}
    for n_reps in (3, 5, 8, 15, 30):
        n_success = 0
        n_trials = 5
        for trial in range(n_trials):
            rows = []
            for rep in range(n_reps):
                seed = stable_seed("power", n_reps, trial, rep)
                out, p = simulate_replicate(500, 4, 4, ks, 4, 0.55, 0.9, 0.35, seed)
                rows.append({
                    "single_acc_k1": out[1]["single_acc"],
                    "sc_acc_k20": out[20]["sc_acc"],
                })
            single1 = np.array([r["single_acc_k1"] for r in rows])
            sc20 = np.array([r["sc_acc_k20"] for r in rows])
            _, pval = stats.ttest_rel(sc20, single1)
            if pval < 0.05:
                n_success += 1
        power_rows[n_reps] = n_success / n_trials
        print(f"  n_reps={n_reps}: fraction of {n_trials} trials significant (p<0.05) = "
              f"{power_rows[n_reps]:.2f}")

    # --- Robustness check: does the verbalized-vs-frequency ECE ranking flip
    # (moderate -> verb better; extreme -> freq better) persist under different
    # overconfidence-transform parameters, or is it an artifact of one fixed
    # (over_a, over_b) choice? Sweep two alternative transforms. ---
    print("=== Robustness: ECE ranking (verb vs freq) under alternative overconfidence transforms ===")
    robustness = {}
    alt_transforms = {
        "weaker_overconf": dict(over_a=0.75, over_b=0.5, noise_sd=0.35),
        "stronger_overconf": dict(over_a=0.4, over_b=1.3, noise_sd=0.35),
    }
    for tname, tparams in alt_transforms.items():
        robustness[tname] = {}
        for sname, sparams in {"moderate_heterogeneity": settings["moderate_heterogeneity"],
                                "extreme_heterogeneity": settings["extreme_heterogeneity"]}.items():
            rows = run_setting(f"{tname}_{sname}", N=500, n_reps=30, ks=ks, **sparams, **tparams)
            summ = summarize(rows, ks)
            robustness[tname][sname] = {
                "ece_verb": summ["ece_verb_mean"],
                "ece_freq": summ["ece_freq_mean"],
                "p": summ["ttest_ece_verb_vs_freq_p"],
            }
            print(f"  [{tname}] {sname}: ECE_verb={summ['ece_verb_mean']:.4f}  "
                  f"ECE_freq={summ['ece_freq_mean']:.4f}  p={summ['ttest_ece_verb_vs_freq_p']:.2e}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Save everything for the paper / reproducibility
    dump = {
        "settings": {k: v for k, v in settings.items()},
        "ks": ks,
        "results": {
            name: {"summary": all_results[name]["summary"]} for name in all_results
        },
        "power_analysis": power_rows,
        "robustness_ece_ranking": robustness,
        "runtime_sec": elapsed,
    }
    with open("sim_results.json", "w") as f:
        json.dump(dump, f, indent=2)
    print("Saved sim_results.json")
