"""
Reviewer-requested check: is the H2 null (LS adds nothing beyond TS) specific
to eps=0.1, or does it hold at a stronger smoothing value too?
Runs eps=0.3 label smoothing on digits only (all 8 widths, 15 seeds each),
using identical seeds/splits to the main study, and paired-compares its
post-TS ECE against the main study's digits/baseline post-TS ECE.
"""
import json

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from run_experiment import (
    WIDTHS,
    N_SEEDS,
    make_seed,
    load_dataset,
    MLP,
    smoothed_ce_loss,
    ece,
    fit_temperature,
    EPOCHS,
    LR,
)

EPS = 0.3
results = []
for width in WIDTHS:
    for seed_idx in range(N_SEEDS):
        seed = make_seed("digits", "label_smooth", width, seed_idx)
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_train, y_train, X_val, y_val, X_test, y_test, n_classes = load_dataset(
            "digits", seed
        )
        model = MLP(X_train.shape[1], width, n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=LR * 0.02)
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        for epoch in range(EPOCHS):
            opt.zero_grad()
            logits = model(Xt)
            loss = smoothed_ce_loss(logits, yt, n_classes, EPS)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
            test_logits = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        T = fit_temperature(val_logits, y_val)

        def softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)

        probs_ts = softmax(test_logits / T)
        results.append(
            {
                "width": width,
                "seed_idx": seed_idx,
                "ece_ts": ece(probs_ts, y_test),
            }
        )

# pair against main-study digits/baseline post-TS ECE at matched (width, seed_idx)
main = json.load(open("raw_results.json"))["results"]
base = {
    (r["width"], r["seed_idx"]): r["ece_ts"]
    for r in main
    if r["dataset"] == "digits" and r["condition"] == "baseline"
}
eps03 = {(r["width"], r["seed_idx"]): r["ece_ts"] for r in results}
keys = sorted(base.keys())
b = np.array([base[k] for k in keys])
e = np.array([eps03[k] for k in keys])
diff = b - e
w_stat, p = stats.wilcoxon(diff)

out = {
    "eps": EPS,
    "n_pairs": len(keys),
    "mean_ece_ts_baseline": float(b.mean()),
    "mean_ece_ts_eps03": float(e.mean()),
    "mean_diff": float(diff.mean()),
    "wilcoxon_p": float(p),
}
with open("eps_sensitivity_results.json", "w") as f:
    json.dump(out, f, indent=1)
print(out)
