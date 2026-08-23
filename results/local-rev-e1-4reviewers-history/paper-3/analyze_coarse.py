"""Follow-up check requested by 3/4 round-3 reviewers: does the -0.85
Spearman correlation between semantic entropy and self-consistency, and
their near-identical AUROC, depend on exact-integer clustering? Retrains
the same 10 seeds (identical train_model/run_eval as analyze.py) and adds
a coarse clustering rule (group parsed integers within +-1 into the same
cluster before computing entropy) as a stand-in for looser entailment-style
matching. Also runs a pooled fixed-effect meta-analysis of the per-seed
AUROC-gap bootstrap already stored in multiseed_results.json (no retrain
needed for that part).
"""
import json, math, time
import numpy as np
from scipy.stats import spearmanr, norm
from experiment import train_model, run_eval, auroc_for_wrong

N_SEEDS = 10
K = 8

def coarse_entropy_and_cluster_sc(samples, tol=1):
    vals = [s for s in samples if s is not None]
    if not vals:
        return 0.0, 0.0
    # greedily cluster values within `tol` of each other
    vals_sorted = sorted(vals)
    clusters = []
    for v in vals_sorted:
        placed = False
        for c in clusters:
            if abs(c[0] - v) <= tol:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    n = len(samples)
    ent = 0.0
    for c in clusters:
        p = len(c) / n
        ent -= p * math.log(p + 1e-12)
    modal = max(len(c) for c in clusters)
    sc_coarse = modal / n
    return ent, sc_coarse

def main():
    t0 = time.time()
    rows = []
    for seed in range(N_SEEDS):
        model = train_model(seed=seed)
        records = run_eval(model, n_test=400, K=K, eval_seed=1000 + seed)
        isc = [r["is_correct"] for r in records]
        se_coarse, sc_coarse = [], []
        for r in records:
            e, s = coarse_entropy_and_cluster_sc(r["samples"][:K], tol=1)
            se_coarse.append(e)
            sc_coarse.append(s)
        rho, pval = spearmanr(se_coarse, sc_coarse)
        auc_se_c = auroc_for_wrong(se_coarse, isc, higher_means_wrong=True)
        auc_sc_c = auroc_for_wrong(sc_coarse, isc, higher_means_wrong=False)
        rows.append(dict(seed=seed, rho_coarse=float(rho), auc_se_coarse=float(auc_se_c),
                          auc_sc_coarse=float(auc_sc_c)))
        print(rows[-1])

    rhos = np.array([r["rho_coarse"] for r in rows])
    print(f"\nCoarse (+-1) clustering: Spearman(se,sc) mean={rhos.mean():.3f} sd={rhos.std():.3f} "
          f"range=[{rhos.min():.3f},{rhos.max():.3f}]")

    # Pooled fixed-effect meta-analysis of existing per-seed bootstrap AUROC gaps
    with open("multiseed_results.json") as f:
        multi = json.load(f)
    for key, label in [("diff_mlp_se", "log-prob - semantic entropy"), ("diff_mlp_sc", "log-prob - self-consistency")]:
        means, ses = [], []
        for r in multi:
            d = r[key]
            se_est = (d["hi"] - d["lo"]) / (2 * 1.96)
            means.append(d["mean"]); ses.append(se_est)
        means = np.array(means); ses = np.array(ses)
        w = 1.0 / (ses ** 2)
        pooled_mean = float(np.sum(w * means) / np.sum(w))
        pooled_se = float(np.sqrt(1.0 / np.sum(w)))
        z = pooled_mean / pooled_se
        p_two_sided = float(2 * (1 - norm.cdf(abs(z))))
        print(f"Pooled fixed-effect meta-analysis ({label}): mean_diff={pooled_mean:.4f} "
              f"SE={pooled_se:.4f} z={z:.2f} p={p_two_sided:.4g}")
        rows_meta = dict(label=label, pooled_mean=pooled_mean, pooled_se=pooled_se, z=z, p=p_two_sided)
        with open(f"meta_{key}.json", "w") as g:
            json.dump(rows_meta, g, indent=2)

    with open("coarse_clustering_results.json", "w") as f:
        json.dump(dict(per_seed=rows, rho_mean=float(rhos.mean()), rho_sd=float(rhos.std())), f, indent=2)
    print(f"\nTOTAL_TIME {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
