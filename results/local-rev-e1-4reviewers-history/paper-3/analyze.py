"""Full 10-seed analysis: AUROC, bootstrap, calibration (ECE/Brier), and
Spearman(semantic-entropy, self-consistency) per seed. Produces the artifacts
referenced in the paper: raw_results.json (seed-0 full records),
analysis_summary.json (per-seed table), analysis_log.txt (human-readable log),
results.png (AUROC + ECE across seeds).
"""
import json, math, time, sys
import numpy as np
from scipy.stats import spearmanr
from experiment import (train_model, run_eval, compute_signals, auroc_for_wrong,
                         bootstrap_auroc_diff)

N_SEEDS = 10
K = 8

def ece_brier(probs, is_correct, n_bins=10):
    probs = np.array(probs)
    y = np.array([1.0 if c else 0.0 for c in is_correct])
    brier = float(np.mean((probs - y) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    table = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(y[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
        table.append(dict(bin=[float(lo), float(hi)], n=cnt, conf=conf, acc=acc))
    return float(ece), brier, table

def main():
    t0 = time.time()
    log = []
    def L(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    per_seed = []
    seed0_records = None
    for seed in range(N_SEEDS):
        model = train_model(seed=seed)
        records = run_eval(model, n_test=400, K=K, eval_seed=1000 + seed)
        sig = compute_signals(records, K=K)
        acc = float(np.mean([r["is_correct"] for r in records]))
        isc = [s["is_correct"] for s in sig]
        mlp = [s["mean_logp"] for s in sig]
        se = [s["se"] for s in sig]
        sc = [s["sc"] for s in sig]
        ent1 = [s["ent1"] for s in sig]

        auc_mlp = auroc_for_wrong(mlp, isc, higher_means_wrong=False)
        auc_se = auroc_for_wrong(se, isc, higher_means_wrong=True)
        auc_sc = auroc_for_wrong(sc, isc, higher_means_wrong=False)
        auc_ent1 = auroc_for_wrong(ent1, isc, higher_means_wrong=True)

        conf_logp = [math.exp(x) for x in mlp]
        ece_logp, brier_logp, tab_logp = ece_brier(conf_logp, isc)
        ece_sc, brier_sc, tab_sc = ece_brier(sc, isc)

        rho, pval = spearmanr(se, sc)

        rec = dict(seed=seed, acc=acc, auc_ent1=auc_ent1, auc_mlp=auc_mlp, auc_se=auc_se, auc_sc=auc_sc,
                   ece_logp=ece_logp, brier_logp=brier_logp, ece_sc=ece_sc, brier_sc=brier_sc,
                   spearman_se_sc=float(rho), spearman_p=float(pval))
        per_seed.append(rec)
        L(f"seed={seed} acc={acc:.3f} auc_mlp={auc_mlp:.3f} auc_se={auc_se:.3f} auc_sc={auc_sc:.3f} "
          f"ece_logp={ece_logp:.3f} ece_sc={ece_sc:.3f} rho(se,sc)={rho:.3f}")

        if seed == 0:
            seed0_records = dict(records=records, signals=sig,
                                  reliability_logp=tab_logp, reliability_sc=tab_sc,
                                  ece_logp=ece_logp, brier_logp=brier_logp,
                                  ece_sc=ece_sc, brier_sc=brier_sc)

    accs = np.array([r["acc"] for r in per_seed])
    ece_logp_all = np.array([r["ece_logp"] for r in per_seed])
    ece_sc_all = np.array([r["ece_sc"] for r in per_seed])
    brier_logp_all = np.array([r["brier_logp"] for r in per_seed])
    brier_sc_all = np.array([r["brier_sc"] for r in per_seed])
    rhos = np.array([r["spearman_se_sc"] for r in per_seed])

    L("\n=== Multi-seed calibration (all 10 seeds) ===")
    L(f"ECE logp: mean={ece_logp_all.mean():.3f} sd={ece_logp_all.std():.3f}")
    L(f"ECE sc:   mean={ece_sc_all.mean():.3f} sd={ece_sc_all.std():.3f}")
    L(f"Brier logp: mean={brier_logp_all.mean():.3f} sd={brier_logp_all.std():.3f}")
    L(f"Brier sc:   mean={brier_sc_all.mean():.3f} sd={brier_sc_all.std():.3f}")
    L(f"sc better calibrated (lower ECE) in {int((ece_sc_all < ece_logp_all).sum())}/10 seeds")
    L(f"Spearman(se,sc): mean={rhos.mean():.3f} sd={rhos.std():.3f} (all seeds strongly negative: {(rhos < -0.5).sum()}/10)")

    # Sensitivity: exclude low-accuracy (near-degenerate) seeds, acc < 0.10
    low_mask = accs < 0.10
    high_idx = [i for i in range(N_SEEDS) if not low_mask[i]]
    L(f"\n=== Sensitivity: excluding {int(low_mask.sum())} seeds with acc<10% (seeds {[per_seed[i]['seed'] for i in range(N_SEEDS) if low_mask[i]]}) ===")
    for key in ["auc_mlp", "auc_se", "auc_sc"]:
        vals_all = np.array([r[key] for r in per_seed])
        vals_hi = vals_all[high_idx]
        L(f"{key}: all10 mean={vals_all.mean():.3f} sd={vals_all.std():.3f}  |  excl-low mean={vals_hi.mean():.3f} sd={vals_hi.std():.3f} (n={len(high_idx)})")
    win_se_hi = sum(per_seed[i]["auc_mlp"] > per_seed[i]["auc_se"] for i in high_idx)
    win_sc_hi = sum(per_seed[i]["auc_mlp"] > per_seed[i]["auc_sc"] for i in high_idx)
    L(f"win counts excl-low: mlp>se {win_se_hi}/{len(high_idx)}, mlp>sc {win_sc_hi}/{len(high_idx)}")

    with open("analysis_summary.json", "w") as f:
        json.dump(dict(per_seed=per_seed,
                        multiseed_calibration=dict(
                            ece_logp_mean=float(ece_logp_all.mean()), ece_logp_sd=float(ece_logp_all.std()),
                            ece_sc_mean=float(ece_sc_all.mean()), ece_sc_sd=float(ece_sc_all.std()),
                            brier_logp_mean=float(brier_logp_all.mean()), brier_logp_sd=float(brier_logp_all.std()),
                            brier_sc_mean=float(brier_sc_all.mean()), brier_sc_sd=float(brier_sc_all.std()),
                            sc_better_in=int((ece_sc_all < ece_logp_all).sum())),
                        spearman=dict(mean=float(rhos.mean()), sd=float(rhos.std())),
                        sensitivity_excl_low_acc=dict(
                            excluded_seeds=[per_seed[i]["seed"] for i in range(N_SEEDS) if low_mask[i]],
                            n_remaining=len(high_idx),
                            auc_mlp_mean=float(np.array([r["auc_mlp"] for r in per_seed])[high_idx].mean()),
                            auc_se_mean=float(np.array([r["auc_se"] for r in per_seed])[high_idx].mean()),
                            auc_sc_mean=float(np.array([r["auc_sc"] for r in per_seed])[high_idx].mean()),
                            win_mlp_gt_se=win_se_hi, win_mlp_gt_sc=win_sc_hi)),
                  f, indent=2)

    with open("raw_results.json", "w") as f:
        json.dump(seed0_records, f, indent=2)

    L(f"\nTOTAL_TIME {time.time() - t0:.1f}s")
    with open("analysis_log.txt", "w") as f:
        f.write("\n".join(log))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        seeds = [r["seed"] for r in per_seed]
        axes[0].plot(seeds, [r["auc_mlp"] for r in per_seed], "o-", label="log-prob")
        axes[0].plot(seeds, [r["auc_se"] for r in per_seed], "s-", label="semantic entropy")
        axes[0].plot(seeds, [r["auc_sc"] for r in per_seed], "^-", label="self-consistency")
        axes[0].set_xlabel("seed"); axes[0].set_ylabel("AUROC"); axes[0].legend(); axes[0].set_title("AUROC per seed")
        axes[1].plot(seeds, ece_logp_all, "o-", label="log-prob ECE")
        axes[1].plot(seeds, ece_sc_all, "^-", label="self-consistency ECE")
        axes[1].set_xlabel("seed"); axes[1].set_ylabel("ECE"); axes[1].legend(); axes[1].set_title("Calibration per seed")
        fig.tight_layout()
        fig.savefig("results.png", dpi=120)
    except Exception as e:
        L("plot failed:", e)

if __name__ == "__main__":
    main()
