"""
Robustness follow-up experiment, run in response to reviewer feedback on v1.

Two additions, both re-using the exact model/data/training code in
experiment.py (imported, not duplicated):

(1) Multi-seed variance: repeat the whole pipeline (fresh data draw, fresh
    model init, fresh training, fresh eval) at 5 seeds, at a reduced scale
    (fewer train examples / epochs / test examples / samples) so it fits in
    a short CPU budget, and report mean +/- std AUROC per signal plus a
    paired sign check on whether mean-max-prob beats self-consistency in
    each seed. This checks whether the *ranking* of signals is stable, which
    is what the paper's claims rely on, not exact absolute numbers.

(2) A linear probe on the model's own hidden state (mean-pooled over the
    prompt, taken right before generation starts, d_model=64 features),
    logistic-regression-trained per seed to predict greedy-decode
    correctness -- a minimal, from-scratch analogue of the "hidden-state
    probe" family (Semantic Entropy Probes) the paper cites but did not
    previously compare against. Fit/eval split is disjoint, exactly as for
    the logistic combo in experiment.py.
"""
import time
import numpy as np
import torch

import experiment as E

SEEDS = [0, 1, 2, 3, 4]
TRAIN_N = 4000
TEST_N = 400
EPOCHS = 4
K_SAMPLES = 5


@torch.no_grad()
def get_prompt_hidden(model, prompt):
    ids = E.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    B, T = x.shape
    pos = torch.arange(T).unsqueeze(0).expand(B, T)
    h = model.tok_emb(x) + model.pos_emb(pos)
    mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
    h = model.encoder(h, mask=mask)
    h = model.ln(h)
    return h[0, -1].numpy()  # hidden state at last prompt token, pre-generation


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

    # logistic combo (3 training-free signals)
    mmp_s, mmp_e = E.standardize(mmp[fit_idx], mmp[eval_idx])
    nme_s, nme_e = E.standardize(nme[fit_idx], nme[eval_idx])
    sc_s, sc_e = E.standardize(sc[fit_idx], sc[eval_idx])
    X_fit = np.stack([mmp_s, nme_s, sc_s], axis=1)
    X_eval = np.stack([mmp_e, nme_e, sc_e], axis=1)
    clf = E.LogisticCalibrator(n_features=3).fit(X_fit, labels[fit_idx])
    combo_eval = clf.predict_proba(X_eval)
    aurocs["logistic_combo"] = E.auroc(combo_eval, labels[eval_idx])

    # hidden-state linear probe (training-required, SEP-style)
    mu, sd = H[fit_idx].mean(axis=0), H[fit_idx].std(axis=0) + 1e-8
    Hf, He = (H[fit_idx] - mu) / sd, (H[eval_idx] - mu) / sd
    probe = E.LogisticCalibrator(n_features=H.shape[1], epochs=400).fit(Hf, labels[fit_idx])
    probe_eval = probe.predict_proba(He)
    aurocs["hidden_state_probe"] = E.auroc(probe_eval, labels[eval_idx])
    ece_probe = E.ece(probe_eval, labels[eval_idx])

    acc = float(labels.mean())
    return aurocs, acc, ece_probe


def main():
    t0 = time.time()
    all_aurocs = {k: [] for k in
                  ["mean_max_prob", "neg_mean_entropy", "self_consistency", "logistic_combo", "hidden_state_probe"]}
    accs, probe_eces = [], []
    for s in SEEDS:
        aurocs, acc, ece_probe = run_seed(s)
        accs.append(acc)
        probe_eces.append(ece_probe)
        for k, v in aurocs.items():
            all_aurocs[k].append(v)
        print(f"seed={s} acc={acc:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in aurocs.items()) +
              f" t={time.time()-t0:.1f}s")

    print("\n=== summary (mean +/- std over 5 seeds) ===")
    summary = {}
    for k, vals in all_aurocs.items():
        m, sd = float(np.mean(vals)), float(np.std(vals))
        summary[k] = dict(mean=m, std=sd, values=vals)
        print(f"{k}: {m:.3f} +/- {sd:.3f}  (values={[round(v,3) for v in vals]})")

    mmp_vals = np.array(all_aurocs["mean_max_prob"])
    sc_vals = np.array(all_aurocs["self_consistency"])
    wins = int((mmp_vals > sc_vals).sum())
    print(f"\nmean-max-prob > self-consistency in {wins}/{len(SEEDS)} seeds")
    print(f"mean acc={np.mean(accs):.3f}, mean hidden-state-probe ECE={np.mean(probe_eces):.4f}")
    print(f"total wall clock: {time.time()-t0:.1f}s")

    import json
    with open("robustness_results.json", "w") as f:
        json.dump(dict(summary=summary, mmp_beats_sc=wins, n_seeds=len(SEEDS),
                        mean_acc=float(np.mean(accs)), mean_probe_ece=float(np.mean(probe_eces)),
                        wall_clock_sec=time.time() - t0), f, indent=2)


if __name__ == "__main__":
    main()
