"""
Synthetic 2-class, difficulty-matched-to-digits control.
Goal: disentangle "class count" from "task difficulty" as the gating factor
for the width-4-8 calibration peak, per reviewer request. breast_cancer was
easy (>93% acc even at width=2); here we build a HARD 2-class task
(low class separation) at a similar sample size to digits, and see if the
peak reappears despite only having 2 classes.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import json, time, sys
sys.path.insert(0, ".")
from experiment import train_mlp, softmax, ece_score, nll_score, fit_temperature

t_start = time.time()

X, y = make_classification(
    n_samples=1797, n_features=64, n_informative=20, n_redundant=20,
    n_classes=2, n_clusters_per_class=3, class_sep=0.9, flip_y=0.03,
    random_state=0)
X = StandardScaler().fit_transform(X)
n_classes = 2

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=y_trainval)

print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

widths = [2, 4, 8, 16, 32, 64]
n_seeds = 10
results = []
for width in widths:
    for seed in range(n_seeds):
        model = train_mlp(X_train, y_train, n_classes, width, seed,
                           epochs=150, lr=0.03, l2=1e-4, batch_size=64)
        train_probs = softmax(model.logits(X_train))
        train_acc = (train_probs.argmax(1) == y_train).mean()
        test_logits = model.logits(X_test)
        test_probs = softmax(test_logits)
        test_acc = (test_probs.argmax(1) == y_test).mean()
        test_ece = ece_score(test_probs, y_test)
        test_nll = nll_score(test_probs, y_test)
        val_logits = model.logits(X_val)
        T = fit_temperature(val_logits, y_val, n_classes)
        test_probs_ts = softmax(test_logits / T)
        test_ece_ts = ece_score(test_probs_ts, y_test)
        results.append(dict(width=width, seed=seed, train_acc=float(train_acc),
            test_acc=float(test_acc), test_ece=float(test_ece), test_nll=float(test_nll),
            temperature=float(T), test_ece_ts=float(test_ece_ts)))
    accs = [r['test_acc'] for r in results if r['width']==width]
    eces = [r['test_ece'] for r in results if r['width']==width]
    print(f"width={width} acc={np.mean(accs):.3f}+-{np.std(accs):.3f} ece={np.mean(eces):.3f}+-{np.std(eces):.3f} elapsed={time.time()-t_start:.1f}s")

with open("results_synth2class.json", "w") as f:
    json.dump(results, f, indent=2)

ece48 = [r['test_ece'] for r in results if r['width'] in (4,8)]
ece_plateau = [r['test_ece'] for r in results if r['width'] in (32,64)]
t, p = ttest_ind(ece48, ece_plateau, equal_var=False)
print(f"4+8 (n={len(ece48)}) vs 32+64 (n={len(ece_plateau)}) plateau: t={t:.3f} p={p:.4g}")
print(f"mean ECE 4+8={np.mean(ece48):.4f}, mean ECE 32+64={np.mean(ece_plateau):.4f}")
print(f"Total elapsed: {time.time()-t_start:.1f}s")
