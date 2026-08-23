import time, json
import numpy as np
from experiment import (train_model, sample_pairs, greedy_eval, sample_k,
                         sample_agree_and_entropy, N_TEST, K)
from sklearn.metrics import roc_auc_score

t0 = time.time()
results = {}

# ---- 1. Reimplementation-gap probe (all 4 v3 reviewers): try higher lr / longer warmup
# to see if the reimplementation can approach v1's reported 81% accuracy at 260 steps. ----
def train_model_lr(seed, steps, lr, batch_size=128):
    import torch, torch.nn as nn
    from experiment import TinyTransformer, sample_pairs as sp, batch_tensor, V
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyTransformer().to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        pairs = sp(rng, batch_size)
        x = batch_tensor(pairs)
        logits, _ = model(x[:, :-1])
        target_logits = logits[:, 5:8, :]
        target = x[:, 6:9]
        loss = nn.functional.cross_entropy(target_logits.reshape(-1, V), target.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

lr_probe = []
for lr in [3e-3, 1e-2, 3e-2]:
    for seed in [0, 1, 2]:
        model = train_model_lr(seed=seed, steps=260, lr=lr)
        rng = np.random.default_rng(1000 + seed)
        test_pairs = sample_pairs(rng, N_TEST)
        correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
        lr_probe.append(dict(lr=lr, seed=seed, accuracy=float(correct.mean())))
results["lr_probe"] = lr_probe
lr_agg = {}
for lr in [3e-3, 1e-2, 3e-2]:
    accs = [r["accuracy"] for r in lr_probe if r["lr"] == lr]
    lr_agg[str(lr)] = dict(mean_acc=float(np.mean(accs)), accs=accs)
results["lr_probe_agg"] = lr_agg

# ---- 2. Extend step-budget sweep to 5 seeds per budget (reviewers 1,2,3 asked: does a
# 4th/5th seed change the 2/3 win rate at 150/260 steps?) ----
STEP_BUDGETS = [150, 260, 450]
SWEEP_SEEDS = [0, 1, 2, 3, 4]
sweep5 = []
for steps in STEP_BUDGETS:
    for seed in SWEEP_SEEDS:
        model = train_model(seed=seed, steps=steps)
        rng = np.random.default_rng(3000 + steps * 10 + seed)
        test_pairs = sample_pairs(rng, N_TEST)
        correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
        incorrect = (~correct).astype(int)
        acc = correct.mean()
        if incorrect.sum() == 0 or incorrect.sum() == N_TEST:
            sweep5.append(dict(steps=steps, seed=seed, accuracy=float(acc), auroc_undefined=True))
            continue
        samples = sample_k(model, test_pairs, K=K, seed=steps * 10 + seed)
        agree, sent = sample_agree_and_entropy(samples, greedy_pred, K)
        auc_maxprob = roc_auc_score(incorrect, -maxprob)
        auc_agree = roc_auc_score(incorrect, -agree)
        sweep5.append(dict(steps=steps, seed=seed, accuracy=float(acc),
                            auc_maxprob=float(auc_maxprob), auc_agree=float(auc_agree),
                            agree_wins=bool(auc_agree > auc_maxprob)))
results["sweep5"] = sweep5
agg5 = {}
for steps in STEP_BUDGETS:
    rows = [r for r in sweep5 if r["steps"] == steps and "auc_maxprob" in r]
    agg5[steps] = dict(n=len(rows),
                        auc_maxprob_mean=float(np.mean([r["auc_maxprob"] for r in rows])),
                        auc_agree_mean=float(np.mean([r["auc_agree"] for r in rows])),
                        agree_win_rate=float(np.mean([r["agree_wins"] for r in rows])))
results["sweep5_agg"] = agg5

# ---- 3. Bootstrap CI on the per-seed (agree - maxprob) AUROC gap across the 5 main
# seeds, to check whether the band-stratification gap is distinguishable from noise
# (reviewers 1,2,3 asked this). ----
with open("results_v2.json") as f:
    v2 = json.load(f)
main_runs = v2["main_multiseed"]
gaps = np.array([r["auc_agree"] - r["auc_maxprob"] for r in main_runs])
rng = np.random.default_rng(42)
boot_means = [np.mean(rng.choice(gaps, size=5, replace=True)) for _ in range(5000)]
results["gap_bootstrap"] = dict(
    per_seed_gaps=gaps.tolist(),
    mean_gap=float(gaps.mean()),
    ci_2_5=float(np.percentile(boot_means, 2.5)),
    ci_97_5=float(np.percentile(boot_means, 97.5)),
    frac_positive=float(np.mean(gaps > 0)),
)

results["elapsed_seconds"] = time.time() - t0
with open("results_v4_extra.json", "w") as f:
    json.dump(results, f, indent=2)

print("lr_probe_agg:", json.dumps(lr_agg, indent=2))
print("sweep5_agg:", json.dumps(agg5, indent=2))
print("gap_bootstrap:", json.dumps(results["gap_bootstrap"], indent=2))
print("elapsed:", results["elapsed_seconds"])
