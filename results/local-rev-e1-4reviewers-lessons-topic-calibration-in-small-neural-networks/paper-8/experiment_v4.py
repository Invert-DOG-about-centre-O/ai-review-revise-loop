"""
v4 additions addressing round-3 review feedback (independently raised by 2+ reviewers
where noted):
  - label smoothing at widths 8 and 32, the crossover-region widths that Table 2
    omitted in v3 (flagged by reviewer 1 and reviewer 4)
  - train-set accuracy at epoch 80 vs 400 for width-2 models, to check whether the
    network was still improving on the training set at the 80-epoch cutoff
    (reviewer 1, Q2)
  - extra seeds (10 more, seeds 10-19) at widths 8 and 16 to check whether the sharp
    jump there is sensitive to the specific seed set 0-9 (reviewer 3, Q2)
"""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from experiment_v2 import run_instance, MLP, D, NUM_CLASSES, make_gda_params, sample_gda, fit_temperature, ece

if __name__ == "__main__":
    t0 = time.time()
    seeds = list(range(10))
    extra_seeds = list(range(10, 20))

    # 1. Label smoothing at the crossover-region widths (8, 32)
    _, res_ls_dense = run_instance(data_seed=0, widths=[], seeds=seeds, ls_widths=[8, 32])
    print(f"[{time.time()-t0:.1f}s] LS at widths 8/32 done")

    # 2. Train-accuracy check at width 2, epoch 80 vs 400
    rng = np.random.default_rng(0)
    means, variances, priors = make_gda_params(rng)
    N_TRAIN, N_VAL, N_TEST = 2000, 800, 2000
    x_train, y_train = sample_gda(N_TRAIN, means, variances, priors, rng)
    _, _ = sample_gda(N_VAL, means, variances, priors, rng)
    _, _ = sample_gda(N_TEST, means, variances, priors, rng)
    xt_train, yt_train = torch.tensor(x_train), torch.tensor(y_train)

    def train_acc_at(width, epochs, seed):
        torch.manual_seed(seed)
        model = MLP(width)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        n = xt_train.shape[0]
        bs = 256
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(model(xt_train[idx]), yt_train[idx])
                loss.backward()
                opt.step()
        with torch.no_grad():
            logits = model(xt_train)
            loss_final = loss_fn(logits, yt_train).item()
            acc_final = (logits.argmax(dim=1).numpy() == y_train).mean()
        return acc_final, loss_final

    acc80, acc400 = [], []
    loss80, loss400 = [], []
    for seed in seeds:
        a, l = train_acc_at(2, 80, seed)
        acc80.append(a); loss80.append(l)
        a, l = train_acc_at(2, 400, seed)
        acc400.append(a); loss400.append(l)
    print(f"[{time.time()-t0:.1f}s] width-2 train-acc/loss check done")
    print(f"  80ep:  train_acc={np.mean(acc80):.4f}+-{np.std(acc80):.4f}  train_loss={np.mean(loss80):.4f}+-{np.std(loss80):.4f}")
    print(f"  400ep: train_acc={np.mean(acc400):.4f}+-{np.std(acc400):.4f}  train_loss={np.mean(loss400):.4f}+-{np.std(loss400):.4f}")

    # 3. Extra seeds at widths 8, 16 to test seed-sensitivity of the sharp jump
    _, res_extra = run_instance(data_seed=0, widths=[8, 16], seeds=extra_seeds, ls_widths=[])
    print(f"[{time.time()-t0:.1f}s] extra-seed check (widths 8,16, seeds 10-19) done")

    out = dict(
        ls_dense=dict(widths=[8, 32], seeds=seeds, results=res_ls_dense),
        w2_train_check=dict(
            epochs80=dict(train_acc=acc80, train_loss=loss80),
            epochs400=dict(train_acc=acc400, train_loss=loss400),
        ),
        extra_seeds=dict(widths=[8, 16], seeds=extra_seeds, results=res_extra),
        total_time_s=time.time() - t0,
    )
    with open("results_v4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s")

    # Summaries
    with open("results_v3.json") as f:
        v3 = json.load(f)
    T_8_orig = [r["T_star"] for r in v3["dense"]["results"] if r["width"] == 8]
    T_16_orig = [r["T_star"] for r in json.load(open("results_v2.json"))["main"]["results"]
                 if r["width"] == 16 and r["label_smoothing"] == 0.0]
    T_8_extra = [r["T_star"] for r in res_extra if r["width"] == 8]
    T_16_extra = [r["T_star"] for r in res_extra if r["width"] == 16]
    T_8_all = T_8_orig + T_8_extra
    T_16_all = T_16_orig + T_16_extra
    tstat, pval = stats.ttest_ind(T_8_all, T_16_all, equal_var=False)
    print(f"\nWidth 8 vs 16, n=20 each: mean8={np.mean(T_8_all):.3f} mean16={np.mean(T_16_all):.3f} t={tstat:.2f} p={pval:.2e}")
    print(f"  width 8  (seeds 10-19): {np.mean(T_8_extra):.3f}+-{np.std(T_8_extra):.3f}")
    print(f"  width 16 (seeds 10-19): {np.mean(T_16_extra):.3f}+-{np.std(T_16_extra):.3f}")

    print("\nLS at widths 8/32:")
    noLS_v3 = v3["dense"]["results"]
    for w in [8, 32]:
        ece_noLS = np.mean([r["ece"] for r in noLS_v3 if r["width"] == w])
        T_noLS = np.mean([r["T_star"] for r in noLS_v3 if r["width"] == w])
        ece_LS = np.mean([r["ece"] for r in res_ls_dense if r["width"] == w])
        T_LS = np.mean([r["T_star"] for r in res_ls_dense if r["width"] == w])
        print(f"  width {w}: T*(noLS)={T_noLS:.3f} T*(LS)={T_LS:.3f}  ECE {ece_noLS:.4f}->{ece_LS:.4f} ({ece_LS/ece_noLS:.1f}x)")
