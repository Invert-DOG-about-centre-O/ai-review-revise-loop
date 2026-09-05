"""
Extended calibration sweep addressing review feedback on v1:
  - 10 seeds per width (was 2) -> report mean +/- std / CI on T*
  - label smoothing ablation across the FULL width sweep (was only widths 2,256)
  - a second, independently drawn data-generating instance (DATA_SEED=1) to
    check whether the crossover width is an artifact of one fixed Gaussian draw
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

D = 8
NUM_CLASSES = 3

def make_gda_params(rng):
    means = rng.normal(scale=2.2, size=(NUM_CLASSES, D))
    variances = rng.uniform(0.6, 2.5, size=(NUM_CLASSES, D))
    priors = np.array([1.0, 1.0, 1.0]) / 3.0
    return means, variances, priors

def sample_gda(n, means, variances, priors, rng):
    y = rng.choice(NUM_CLASSES, size=n, p=priors)
    x = rng.normal(size=(n, D)) * np.sqrt(variances[y]) + means[y]
    return x.astype(np.float32), y.astype(np.int64)

def bayes_posterior(x, means, variances, priors):
    logp = np.zeros((x.shape[0], NUM_CLASSES))
    for k in range(NUM_CLASSES):
        diff2 = (x - means[k]) ** 2 / variances[k]
        logp[:, k] = -0.5 * diff2.sum(axis=1) - 0.5 * np.log(variances[k]).sum()
    logp += np.log(priors)[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    return p

class MLP(nn.Module):
    def __init__(self, width, depth=2, in_dim=D):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, NUM_CLASSES)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def ece(confidences, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece_val += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece_val

def fit_temperature(logits, labels, lo=0.05, hi=10.0, steps=200):
    logits = logits.detach()
    labels = labels.detach()
    grid = np.geomspace(lo, hi, steps)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        scaled = logits / T
        nll = F.cross_entropy(scaled, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T), best_nll

def run_instance(data_seed, widths, seeds, ls_widths, epochs=80, lr=1e-2):
    rng = np.random.default_rng(data_seed)
    means, variances, priors = make_gda_params(rng)
    N_TRAIN, N_VAL, N_TEST = 2000, 800, 2000
    x_train, y_train = sample_gda(N_TRAIN, means, variances, priors, rng)
    x_val, y_val = sample_gda(N_VAL, means, variances, priors, rng)
    x_test, y_test = sample_gda(N_TEST, means, variances, priors, rng)
    bayes_post_test = bayes_posterior(x_test, means, variances, priors)
    bayes_acc = float((bayes_post_test.argmax(axis=1) == y_test).mean())

    xt_train, yt_train = torch.tensor(x_train), torch.tensor(y_train)
    xt_val, yt_val = torch.tensor(x_val), torch.tensor(y_val)
    xt_test, yt_test = torch.tensor(x_test), torch.tensor(y_test)

    def train_model(width, label_smoothing, seed):
        torch.manual_seed(seed)
        model = MLP(width)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
        return model

    def evaluate(model):
        with torch.no_grad():
            logits_test = model(xt_test)
            probs_test = F.softmax(logits_test, dim=1).numpy()
            preds_test = probs_test.argmax(axis=1)
            conf_test = probs_test.max(axis=1)
            correct_test = (preds_test == y_test).astype(np.float32)
            logits_val = model(xt_val)
        acc = correct_test.mean()
        ece_val = ece(conf_test, correct_test)
        bayes_conf_of_pred = bayes_post_test[np.arange(len(preds_test)), preds_test]
        bayes_ece_val = np.mean(np.abs(conf_test - bayes_conf_of_pred))
        T_star, _ = fit_temperature(logits_val, yt_val)
        with torch.no_grad():
            probs_ts = F.softmax(logits_test / T_star, dim=1).numpy()
        ece_ts = ece(probs_ts.max(axis=1), correct_test)
        return dict(acc=float(acc), ece=float(ece_val), bayes_ece=float(bayes_ece_val),
                    T_star=float(T_star), ece_after_ts=float(ece_ts))

    results = []
    for width in widths:
        for seed in seeds:
            model = train_model(width, 0.0, seed)
            r = evaluate(model)
            r.update(width=width, seed=seed, label_smoothing=0.0, data_seed=data_seed)
            results.append(r)
    for width in ls_widths:
        for seed in seeds:
            model = train_model(width, 0.1, seed)
            r = evaluate(model)
            r.update(width=width, seed=seed, label_smoothing=0.1, data_seed=data_seed)
            results.append(r)
    return bayes_acc, results

if __name__ == "__main__":
    t0 = time.time()
    widths = [2, 4, 16, 64, 256]
    seeds = list(range(10))  # 10 seeds instead of 2

    bayes_acc0, res0 = run_instance(data_seed=0, widths=widths, seeds=seeds, ls_widths=widths)
    print(f"[{time.time()-t0:.1f}s] main data instance (seed=0) done, bayes_acc={bayes_acc0:.3f}")

    # Second independent draw of the Gaussian task to test robustness of the crossover
    bayes_acc1, res1 = run_instance(data_seed=1, widths=widths, seeds=list(range(5)), ls_widths=[])
    print(f"[{time.time()-t0:.1f}s] robustness data instance (seed=1) done, bayes_acc={bayes_acc1:.3f}")

    out = dict(
        d=D, num_classes=NUM_CLASSES,
        main=dict(bayes_acc=bayes_acc0, widths=widths, seeds=seeds, results=res0),
        robustness=dict(bayes_acc=bayes_acc1, widths=widths, seeds=list(range(5)), results=res1),
        total_time_s=time.time() - t0,
    )
    with open("round2_review_1_results_rerun.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s")
