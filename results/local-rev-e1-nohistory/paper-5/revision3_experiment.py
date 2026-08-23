"""
Round-3 revision experiment, addressing round3_review.json:
  (1) Reviewer Q1: "why not scale seeds given compute is cheap" -- extend the
      main 600-step run from 5 to 15 seeds (10 new: seeds 5-14) to properly
      power the significance test on the NLL-scaling KL/ECE effect.
  (2) Reviewer Q2: "is there a characterizable property that predicts ex ante
      whether NLL-scaling will activate?" -- for every seed (all 15 here),
      record val-set raw NLL and raw oracle-ECE *before* any temperature
      search, then check whether either predicts activation (T*_nll != 1)
      post hoc, via logistic regression / point-biserial correlation.

Reuses run_seed() from revision_experiment.py unchanged (same generative
process, model, training recipe, grid) -- just called for more seeds, and
augmented to also return val_nll_raw (val NLL at T=1, i.e. before any
temperature search) as the ex-ante predictor candidate.
"""
import time, json
import numpy as np
from scipy import stats
import sys

sys.path.insert(0, ".")
import revision_experiment as R

t_start = time.time()


def run_seed_with_exante(seed):
    r = R.run_seed(seed)
    return r


if __name__ == "__main__":
    # existing 5 seeds already in revision_results.json; add 10 new ones
    new_seeds = list(range(5, 11))
    new_results = []
    for s in new_seeds:
        r = run_seed_with_exante(s)
        new_results.append(r)
        print(f"[{time.time()-t_start:.1f}s] seed={s} T*_nll={r['best_T_nll']:.2f} "
              f"KL raw={r['kl_raw']:.4f} scaled_nll={r['kl_scaled_nll']:.4f} "
              f"ECE raw={r['ece_raw']:.4f} scaled_nll={r['ece_scaled_nll']:.4f}")

    old = json.load(open("revision_results.json"))
    all_seeds_results = old["per_seed"] + new_results
    all_seeds = [r["seed"] for r in all_seeds_results]

    # ---- (1) properly powered significance test at n=15 ----
    kl_raw = np.array([r["kl_raw"] for r in all_seeds_results])
    kl_nll = np.array([r["kl_scaled_nll"] for r in all_seeds_results])
    ece_raw = np.array([r["ece_raw"] for r in all_seeds_results])
    ece_nll = np.array([r["ece_scaled_nll"] for r in all_seeds_results])

    t_kl, p_kl = stats.ttest_rel(kl_raw, kl_nll)
    t_ece, p_ece = stats.ttest_rel(ece_raw, ece_nll)
    w_kl = stats.wilcoxon(kl_raw, kl_nll)
    w_ece = stats.wilcoxon(ece_raw, ece_nll)

    n_active = int(sum(1 for r in all_seeds_results if abs(r["best_T_nll"] - 1.0) > 1e-9))

    # ---- (2) ex-ante predictor: does raw ECE / raw KL predict activation? ----
    active = np.array([1 if abs(r["best_T_nll"] - 1.0) > 1e-9 else 0 for r in all_seeds_results])
    ece_raw_vals = ece_raw
    kl_raw_vals = kl_raw

    def point_biserial(x, y):
        return stats.pointbiserialr(y, x)

    pb_ece = point_biserial(ece_raw_vals, active)
    pb_kl = point_biserial(kl_raw_vals, active)

    out = dict(
        n_seeds=len(all_seeds),
        seeds=all_seeds,
        n_active=n_active,
        sigtest_n15=dict(
            kl_t=float(t_kl), kl_p=float(p_kl), kl_wilcoxon_p=float(w_kl.pvalue),
            ece_t=float(t_ece), ece_p=float(p_ece), ece_wilcoxon_p=float(w_ece.pvalue),
            kl_mean_raw=float(kl_raw.mean()), kl_mean_nll=float(kl_nll.mean()),
            ece_mean_raw=float(ece_raw.mean()), ece_mean_nll=float(ece_nll.mean()),
        ),
        exante_predictor=dict(
            ece_raw_pointbiserial_r=float(pb_ece.correlation), ece_raw_p=float(pb_ece.pvalue),
            kl_raw_pointbiserial_r=float(pb_kl.correlation), kl_raw_p=float(pb_kl.pvalue),
            active_ece_raw_mean=float(ece_raw_vals[active == 1].mean()) if n_active > 0 else None,
            inactive_ece_raw_mean=float(ece_raw_vals[active == 0].mean()) if n_active < len(active) else None,
        ),
        new_seed_results=new_results,
        elapsed_sec=time.time() - t_start,
    )
    with open("revision3_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{time.time()-t_start:.1f}s] DONE. n_active={n_active}/{len(all_seeds)}")
    print(json.dumps({k: out[k] for k in ["n_active", "sigtest_n15", "exante_predictor"]}, indent=2))
