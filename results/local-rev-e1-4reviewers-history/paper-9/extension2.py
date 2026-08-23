"""
Extension study addressing round-3 reviewer requests (all 4 reviewers, convergent):
(1) RAPS (k_reg, lambda) sweep -- is "RAPS recovers ~half the gap" robust to hyperparameters?
(2) Larger class count (20-way instead of 6-way) -- does score-family dominance persist
    closer to LLM/MCQA-scale label spaces?
Reuses data/model/scoring code from experiment.py and the RAPS scorer from extension.py.
"""
import json
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
import experiment as E
from extension import conformal_raps

N_SEEDS_SWEEP = 5
N_SEEDS_SCALE = 5


def run_raps_sweep_seed(seed):
    X_train, y_train, X_cal, y_cal, X_test, y_test = E.make_data(seed)
    base = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                          solver="adam", max_iter=80, random_state=seed,
                          batch_size=len(X_train))
    base.fit(X_train, y_train)
    cal_proba = base.predict_proba(X_cal)
    test_proba = base.predict_proba(X_test)

    out = {}
    for k_reg in [1, 2, 3]:
        for lam in [0.01, 0.05, 0.2]:
            _, ps = conformal_raps(cal_proba, y_cal, test_proba, alpha=0.10, k_reg=k_reg, lam=lam)
            cov, size, empty = E.eval_conformal(ps, y_test)
            out[f"k{k_reg}_l{lam}"] = {"coverage": cov, "avg_set_size": size, "empty_frac": empty}
    return out


def run_scale_seed(seed, n_classes):
    orig_nc = E.N_CLASSES
    E.N_CLASSES = n_classes
    try:
        X_train, y_train, X_cal, y_cal, X_test, y_test = E.make_data(seed)
        base = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                              solver="adam", max_iter=80, random_state=seed,
                              batch_size=len(X_train))
        base.fit(X_train, y_train)
        row = {}
        for name, temp_scale in [("raw_softmax", False), ("temp_scaled", True)]:
            cal_proba = base.predict_proba(X_cal)
            test_proba = base.predict_proba(X_test)
            if temp_scale:
                cal_logits = E.to_logits(cal_proba)
                T = E.fit_temperature(cal_logits, y_cal)
                cal_proba = E.softmax(cal_logits / T)
                test_proba = E.softmax(E.to_logits(test_proba) / T)
            acc = float((test_proba.argmax(axis=1) == y_test).mean())
            ece_val = float(E.ece(test_proba, y_test))
            sub = {"accuracy": acc, "ece": ece_val, "T": float(T) if temp_scale else 1.0}
            for score_name, fn in [("LAC", E.conformal_lac), ("APS", E.conformal_aps)]:
                _, ps = fn(cal_proba, y_cal, test_proba, 0.10)
                cov, size, empty = E.eval_conformal(ps, y_test)
                sub[score_name] = {"coverage": cov, "avg_set_size": size, "empty_frac": empty}
            row[name] = sub
        return row
    finally:
        E.N_CLASSES = orig_nc


def main():
    t0 = time.time()

    # (1) RAPS hyperparameter sweep, main regime, raw_softmax confidences
    per_seed = [run_raps_sweep_seed(s) for s in range(N_SEEDS_SWEEP)]
    sweep = {}
    for key in per_seed[0]:
        for stat in ["coverage", "avg_set_size", "empty_frac"]:
            vals = [r[key][stat] for r in per_seed]
            sweep.setdefault(key, {})[stat] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    t1 = time.time()

    # (2) larger class count (20-way) to probe LLM/MCQA-scale label spaces
    scale_per_seed = [run_scale_seed(s, 20) for s in range(N_SEEDS_SCALE)]
    scale_summary = {}
    for method in ["raw_softmax", "temp_scaled"]:
        scale_summary[method] = {}
        for stat in ["accuracy", "ece", "T"]:
            vals = [r[method][stat] for r in scale_per_seed]
            scale_summary[method][stat] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for score in ["LAC", "APS"]:
            scale_summary[method][score] = {}
            for stat in ["coverage", "avg_set_size", "empty_frac"]:
                vals = [r[method][score][stat] for r in scale_per_seed]
                scale_summary[method][score][stat] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    t2 = time.time()

    out = {
        "n_seeds_sweep": N_SEEDS_SWEEP,
        "n_seeds_scale": N_SEEDS_SCALE,
        "raps_sweep": sweep,
        "scale_20class": scale_summary,
        "wallclock_sweep_sec": t1 - t0,
        "wallclock_scale_sec": t2 - t1,
    }
    with open("extension2_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"RAPS sweep done in {t1-t0:.1f}s")
    for key, v in sweep.items():
        print(key, "size=", round(v["avg_set_size"]["mean"], 3), "+/-", round(v["avg_set_size"]["std"], 3),
              "empty=", round(v["empty_frac"]["mean"], 3))

    print(f"20-class scale done in {t2-t1:.1f}s")
    for method in ["raw_softmax", "temp_scaled"]:
        s = scale_summary[method]
        print(method, "acc=", round(s["accuracy"]["mean"], 4), "ece=", round(s["ece"]["mean"], 4),
              "LAC=", round(s["LAC"]["avg_set_size"]["mean"], 3), "+/-", round(s["LAC"]["avg_set_size"]["std"], 3),
              "APS=", round(s["APS"]["avg_set_size"]["mean"], 3), "+/-", round(s["APS"]["avg_set_size"]["std"], 3))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
