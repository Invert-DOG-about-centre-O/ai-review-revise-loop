"""
Extension study addressing round-2 reviewer requests:
(1) RAPS (regularized APS) vs vanilla APS vs LAC, same seeds/regime as main study.
(2) LAC floor-effect check at 80% target coverage (alpha=0.20) vs 90%.
(3) Ablation isolating sigma vs label-noise as driver of T<1 underconfidence.
Reuses data/model/scoring code from experiment.py.
"""
import json
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import experiment as E

N_SEEDS_EXT = 5
N_SEEDS_ABL = 3


def conformal_raps(cal_probs, y_cal, test_probs, alpha, k_reg=2, lam=0.05):
    def raps_scores(probs, y):
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])
        base = cum[np.arange(len(y)), rank]
        reg = lam * np.maximum(rank + 1 - k_reg, 0)
        return base + reg, order, sorted_p, cum

    scores, _, _, _ = raps_scores(cal_probs, y_cal)
    n = len(y_cal)
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")

    order = np.argsort(-test_probs, axis=1)
    sorted_p = np.take_along_axis(test_probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    ranks = np.arange(sorted_p.shape[1])[None, :]
    reg = lam * np.maximum(ranks + 1 - k_reg, 0)
    cum_reg = cum + reg
    include = cum_reg <= qhat
    include[:, 0] = True
    pred_sets = np.zeros_like(test_probs, dtype=bool)
    for i in range(len(test_probs)):
        pred_sets[i, order[i][include[i]]] = True
    return qhat, pred_sets


def run_score_and_coverage_ext(seed):
    X_train, y_train, X_cal, y_cal, X_test, y_test = E.make_data(seed)
    base = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                          solver="adam", max_iter=80, random_state=seed,
                          batch_size=len(X_train))
    base.fit(X_train, y_train)

    out = {}
    for name, temp_scale in [("raw_softmax", False), ("temp_scaled", True)]:
        cal_proba = base.predict_proba(X_cal)
        test_proba = base.predict_proba(X_test)
        if temp_scale:
            cal_logits = E.to_logits(cal_proba)
            T = E.fit_temperature(cal_logits, y_cal)
            cal_proba = E.softmax(cal_logits / T)
            test_proba = E.softmax(E.to_logits(test_proba) / T)

        row = {}
        # RAPS at alpha=0.10
        qhat, ps = conformal_raps(cal_proba, y_cal, test_proba, alpha=0.10)
        cov, size, empty = E.eval_conformal(ps, y_test)
        row["RAPS"] = {"coverage": cov, "avg_set_size": size, "empty_frac": empty}
        # LAC / APS at alpha=0.10 (sanity, should match experiment.py)
        for score_name, fn in [("LAC", E.conformal_lac), ("APS", E.conformal_aps)]:
            qhat, ps = fn(cal_proba, y_cal, test_proba, 0.10)
            cov, size, empty = E.eval_conformal(ps, y_test)
            row[score_name] = {"coverage": cov, "avg_set_size": size, "empty_frac": empty}
        # LAC at alpha=0.20 (80% target coverage) -- floor-effect check
        qhat, ps = E.conformal_lac(cal_proba, y_cal, test_proba, alpha=0.20)
        cov, size, empty = E.eval_conformal(ps, y_test)
        row["LAC_80"] = {"coverage": cov, "avg_set_size": size, "empty_frac": empty}
        out[name] = row
    return out


def run_ablation(sigma, label_noise, seed):
    orig_sigma, orig_ln = E.NOISE_SIGMA, E.LABEL_NOISE
    E.NOISE_SIGMA, E.LABEL_NOISE = sigma, label_noise
    try:
        X_train, y_train, X_cal, y_cal, X_test, y_test = E.make_data(seed)
        base = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu",
                              solver="adam", max_iter=80, random_state=seed,
                              batch_size=len(X_train))
        base.fit(X_train, y_train)
        cal_proba = base.predict_proba(X_cal)
        cal_logits = E.to_logits(cal_proba)
        T = E.fit_temperature(cal_logits, y_cal)
        test_proba = base.predict_proba(X_test)
        acc = float((test_proba.argmax(axis=1) == y_test).mean())
        ece_val = float(E.ece(test_proba, y_test))
        return {"T": float(T), "accuracy": acc, "ece": ece_val}
    finally:
        E.NOISE_SIGMA, E.LABEL_NOISE = orig_sigma, orig_ln


def main():
    t0 = time.time()

    # (1)+(2) RAPS and 80%-coverage LAC, 5 seeds, main regime
    per_seed = [run_score_and_coverage_ext(s) for s in range(N_SEEDS_EXT)]
    summary = {}
    for method in ["raw_softmax", "temp_scaled"]:
        summary[method] = {}
        for score in ["LAC", "APS", "RAPS", "LAC_80"]:
            for key in ["coverage", "avg_set_size", "empty_frac"]:
                vals = [r[method][score][key] for r in per_seed]
                summary[method].setdefault(score, {})[key] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals))}
    t1 = time.time()

    # (3) ablation: isolate sigma vs label noise
    ablation = {}
    for label, (sigma, ln) in {
        "main_sigma1.8_ln0.12": (1.8, 0.12),
        "sigma1.8_ln0.0": (1.8, 0.0),
        "sigma0.8_ln0.12": (0.8, 0.12),
        "sigma0.8_ln0.0": (0.8, 0.0),
    }.items():
        rows = [run_ablation(sigma, ln, s) for s in range(N_SEEDS_ABL)]
        ablation[label] = {
            "T": {"mean": float(np.mean([r["T"] for r in rows])), "std": float(np.std([r["T"] for r in rows]))},
            "accuracy": {"mean": float(np.mean([r["accuracy"] for r in rows])), "std": float(np.std([r["accuracy"] for r in rows]))},
            "ece": {"mean": float(np.mean([r["ece"] for r in rows])), "std": float(np.std([r["ece"] for r in rows]))},
        }
    t2 = time.time()

    out = {
        "n_seeds_ext": N_SEEDS_EXT,
        "n_seeds_ablation": N_SEEDS_ABL,
        "score_coverage_summary": summary,
        "ablation": ablation,
        "wallclock_score_cov_sec": t1 - t0,
        "wallclock_ablation_sec": t2 - t1,
    }
    with open("extension_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"score/coverage ext done in {t1-t0:.1f}s")
    for method in ["raw_softmax", "temp_scaled"]:
        s = summary[method]
        print(method,
              "LAC90=", round(s["LAC"]["avg_set_size"]["mean"], 3),
              "LAC80=", round(s["LAC_80"]["avg_set_size"]["mean"], 3),
              "APS=", round(s["APS"]["avg_set_size"]["mean"], 3),
              "RAPS=", round(s["RAPS"]["avg_set_size"]["mean"], 3),
              "LAC80_empty=", round(s["LAC_80"]["empty_frac"]["mean"], 3))
    print(f"ablation done in {t2-t1:.1f}s")
    for k, v in ablation.items():
        print(k, "T=", round(v["T"]["mean"], 3), "acc=", round(v["accuracy"]["mean"], 4), "ece=", round(v["ece"]["mean"], 4))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
