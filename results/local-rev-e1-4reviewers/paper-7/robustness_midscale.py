"""
Mid-scale multi-seed sweep, added in response to round-2 review feedback
(reviewers 1-4 independently noted the 5-seed sweep in robustness.py runs at
~3% accuracy, far from the main run's 65.4%, so it doesn't establish whether
self-consistency's seed-instability finding holds at the accuracy regime the
headline AUROC numbers come from).

This uses a larger configuration than robustness.py (closer to, but still
below, the main run's cost) and fewer seeds (3, to fit the compute budget),
to check whether the same qualitative pattern (self-consistency much
higher-variance than mean-max-prob/entropy) reproduces at much higher
accuracy.
"""
import time
import numpy as np
import torch

import experiment as E
from robustness import get_prompt_hidden

SEEDS = [0, 1, 2]
TRAIN_N = 12000
TEST_N = 1000
EPOCHS = 6
K_SAMPLES = 8


def run_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = E.build_dataset(TRAIN_N, digit_range=[1, 2, 3, 4])
    test_data = E.build_dataset(TEST_N, digit_range=[1, 2, 3, 4])

    model = E.TinyTransformer(len(E.VOCAB), d_model=64, n_heads=4, n_layers=3, max_len=32)
    E.train(model, train_data, epochs=EPOCHS, batch_size=128, lr=3e-3)
    model.eval()

    results, hiddens = [], []
    for ex in test_data:
        r = E.evaluate_example(model, ex, k_samples=K_SAMPLES, sample_temp=1.0)
        results.append(r)
        hiddens.append(get_prompt_hidden(model, ex["prompt"]))

    labels = np.array([r["correct"] for r in results])
    mmp = np.array([r["mean_max_prob"] for r in results])
    nme = np.array([r["neg_mean_entropy"] for r in results])
    sc = np.array([r["self_consistency"] for r in results])
    H = np.stack(hiddens, axis=0)

    idx = np.arange(len(results))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    half = len(idx) // 2
    fit_idx, eval_idx = idx[:half], idx[half:]

    aurocs = {
        "mean_max_prob": E.auroc(mmp[eval_idx], labels[eval_idx]),
        "neg_mean_entropy": E.auroc(nme[eval_idx], labels[eval_idx]),
        "self_consistency": E.auroc(sc[eval_idx], labels[eval_idx]),
    }

    mmp_s, mmp_e = E.standardize(mmp[fit_idx], mmp[eval_idx])
    nme_s, nme_e = E.standardize(nme[fit_idx], nme[eval_idx])
    sc_s, sc_e = E.standardize(sc[fit_idx], sc[eval_idx])
    X_fit = np.stack([mmp_s, nme_s, sc_s], axis=1)
    X_eval = np.stack([mmp_e, nme_e, sc_e], axis=1)
    clf = E.LogisticCalibrator(n_features=3).fit(X_fit, labels[fit_idx])
    combo_eval = clf.predict_proba(X_eval)
    aurocs["logistic_combo"] = E.auroc(combo_eval, labels[eval_idx])

    mu, sd = H[fit_idx].mean(axis=0), H[fit_idx].std(axis=0) + 1e-8
    Hf, He = (H[fit_idx] - mu) / sd, (H[eval_idx] - mu) / sd
    probe = E.LogisticCalibrator(n_features=H.shape[1], epochs=400).fit(Hf, labels[fit_idx])
    probe_eval = probe.predict_proba(He)
    aurocs["hidden_state_probe"] = E.auroc(probe_eval, labels[eval_idx])

    acc = float(labels.mean())
    return aurocs, acc


def main():
    t0 = time.time()
    all_aurocs = {k: [] for k in
                  ["mean_max_prob", "neg_mean_entropy", "self_consistency", "logistic_combo", "hidden_state_probe"]}
    accs = []
    for s in SEEDS:
        aurocs, acc = run_seed(s)
        accs.append(acc)
        for k, v in aurocs.items():
            all_aurocs[k].append(v)
        print(f"seed={s} acc={acc:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in aurocs.items()) +
              f" t={time.time()-t0:.1f}s")

    print("\n=== summary (mean +/- std over seeds) ===")
    summary = {}
    for k, vals in all_aurocs.items():
        m, sd = float(np.mean(vals)), float(np.std(vals))
        summary[k] = dict(mean=m, std=sd, values=vals)
        print(f"{k}: {m:.3f} +/- {sd:.3f}  (values={[round(v,3) for v in vals]})")

    print(f"mean acc={np.mean(accs):.3f}")
    print(f"total wall clock: {time.time()-t0:.1f}s")

    import json
    with open("robustness_midscale_results.json", "w") as f:
        json.dump(dict(summary=summary, n_seeds=len(SEEDS),
                        mean_acc=float(np.mean(accs)),
                        wall_clock_sec=time.time() - t0), f, indent=2)


if __name__ == "__main__":
    main()
