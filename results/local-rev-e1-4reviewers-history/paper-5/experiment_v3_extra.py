import time, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from experiment import (train_model, sample_pairs, greedy_eval, sample_k,
                         sample_agree_and_entropy, N_TEST, K)

t0 = time.time()
results = {}

# ---- 1. Multi-seed step-budget sweep (reviewers 2,3,4: sweep was seed-0 only) ----
STEP_BUDGETS = [150, 260, 450]
SWEEP_SEEDS = [0, 1, 2]
sweep_multiseed = []
for steps in STEP_BUDGETS:
    for seed in SWEEP_SEEDS:
        model = train_model(seed=seed, steps=steps)
        rng = np.random.default_rng(3000 + steps * 10 + seed)
        test_pairs = sample_pairs(rng, N_TEST)
        correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
        incorrect = (~correct).astype(int)
        acc = correct.mean()
        if incorrect.sum() == 0 or incorrect.sum() == N_TEST:
            sweep_multiseed.append(dict(steps=steps, seed=seed, accuracy=float(acc), auroc_undefined=True))
            continue
        samples = sample_k(model, test_pairs, K=K, seed=steps * 10 + seed)
        agree, sent = sample_agree_and_entropy(samples, greedy_pred, K)
        auc_maxprob = roc_auc_score(incorrect, -maxprob)
        auc_agree = roc_auc_score(incorrect, -agree)
        sweep_multiseed.append(dict(steps=steps, seed=seed, accuracy=float(acc),
                                     auc_maxprob=float(auc_maxprob), auc_agree=float(auc_agree),
                                     agree_wins=bool(auc_agree > auc_maxprob)))
results["sweep_multiseed"] = sweep_multiseed

# aggregate per step budget
agg_by_step = {}
for steps in STEP_BUDGETS:
    rows = [r for r in sweep_multiseed if r["steps"] == steps and "auc_maxprob" in r]
    agg_by_step[steps] = dict(
        n=len(rows),
        auc_maxprob_mean=float(np.mean([r["auc_maxprob"] for r in rows])),
        auc_agree_mean=float(np.mean([r["auc_agree"] for r in rows])),
        agree_win_rate=float(np.mean([r["agree_wins"] for r in rows])),
    )
results["sweep_multiseed_agg"] = agg_by_step

# ---- 2. Stratify main 5-seed runs by accuracy band (reviewers 2,3,4) ----
with open("results_v2.json") as f:
    v2 = json.load(f)
main_runs = v2["main_multiseed"]
accs = [r["accuracy"] for r in main_runs]
order = np.argsort(accs)
low = [main_runs[i] for i in order[:2]]
high = [main_runs[i] for i in order[2:]]
def band_stats(rows):
    return dict(n=len(rows),
                acc_mean=float(np.mean([r["accuracy"] for r in rows])),
                auc_maxprob_mean=float(np.mean([r["auc_maxprob"] for r in rows])),
                auc_agree_mean=float(np.mean([r["auc_agree"] for r in rows])))
results["accuracy_band_stratification"] = dict(
    low_accuracy_seeds=band_stats(low), high_accuracy_seeds=band_stats(high))

# ---- 3. Ridge alpha sweep on full probe (reviewers 1,2: is negative R2 a regularization artifact) ----
ALPHAS = [1.0, 10.0, 50.0, 200.0, 1000.0]
alpha_sweep = {a: [] for a in ALPHAS}
half = N_TEST // 2
for seed in [0, 1, 2, 3, 4]:
    model = train_model(seed=seed, steps=260)
    rng = np.random.default_rng(1000 + seed)
    test_pairs = sample_pairs(rng, N_TEST)
    correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
    incorrect = (~correct).astype(int)
    all_samples = sample_k(model, test_pairs, K=K, seed=seed)
    agree20, sent20 = sample_agree_and_entropy(all_samples, greedy_pred, K)
    feat_full = np.concatenate([maxprob[:, None], entropy[:, None], hidden], axis=1)
    mu, sd = feat_full[:half].mean(0), feat_full[:half].std(0) + 1e-8
    Xtr = (feat_full[:half] - mu) / sd
    Xte = (feat_full[half:] - mu) / sd
    for a in ALPHAS:
        ridge = Ridge(alpha=a)
        ridge.fit(Xtr, sent20[:half])
        r2 = ridge.score(Xte, sent20[half:])
        alpha_sweep[a].append(r2)
results["probe_alpha_sweep"] = {str(a): dict(mean_r2=float(np.mean(v)), std_r2=float(np.std(v)))
                                 for a, v in alpha_sweep.items()}

# ---- 4. Construct validity: are low-accuracy seeds emitting near-miss errors or near-random noise? ----
# Mean digit-level Hamming distance between predicted and target 3-digit string, incorrect examples only
digit_dist = []
for seed in [0, 1, 2, 3, 4]:
    model = train_model(seed=seed, steps=260)
    rng = np.random.default_rng(1000 + seed)
    test_pairs = sample_pairs(rng, N_TEST)
    correct, maxprob, entropy, hidden, greedy_pred = greedy_eval(model, test_pairs)
    from experiment import batch_tensor
    import torch
    x = batch_tensor(test_pairs)
    target_ids = x[:, 6:9].numpy()
    wrong_mask = ~correct
    if wrong_mask.sum() == 0:
        continue
    hd = (greedy_pred[wrong_mask] != target_ids[wrong_mask]).sum(axis=1).mean()
    digit_dist.append(dict(seed=seed, accuracy=float(correct.mean()), mean_hamming_on_errors=float(hd)))
results["error_type_by_seed"] = digit_dist

results["elapsed_seconds"] = time.time() - t0
with open("results_v3_extra.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results["sweep_multiseed_agg"], indent=2))
print("band strat:", json.dumps(results["accuracy_band_stratification"], indent=2))
print("alpha sweep:", json.dumps(results["probe_alpha_sweep"], indent=2))
print("error type:", json.dumps(results["error_type_by_seed"], indent=2))
print("elapsed:", results["elapsed_seconds"])
