"""
Simulation-based audit protocol for sycophancy detection.

Reconstruction of the simulator described in v1.md, extended per reviewer
request with (a) a coefficient-attenuation sensitivity sweep and (b)
multi-seed variance reporting, so the headline AUC/accuracy numbers are
reported with explicit dependence on the simulator's class-conditional
effect sizes rather than as a single-seed point estimate.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

N = 6000
BASE_ACC = 0.72


def simulate_dataset(n=N, base_acc=BASE_ACC, seed=0, effect_scale=1.0):
    """effect_scale multiplies every class-conditional (label-encoding)
    offset in the feature generators; effect_scale=1.0 reproduces the
    original simulator, effect_scale=0.0 removes all label-dependent signal
    (features become label-independent noise)."""
    rng = np.random.default_rng(seed)

    persona_syco = rng.beta(2.2, 3.0, n)
    pushback_strength = rng.uniform(0.1, 1.0, n)
    correct_pre = rng.random(n) < base_acc
    orig_conf = np.clip(rng.normal(0.78, 0.12, n), 0.05, 0.99)

    flip_logit = -1.5 + 3.2 * persona_syco + 1.4 * pushback_strength
    flip_prob = 1 / (1 + np.exp(-flip_logit))
    changed = rng.random(n) < flip_prob

    idx = np.where(changed)[0]
    m = len(idx)
    p_syco = persona_syco[idx]
    pb = pushback_strength[idx]
    correct = correct_pre[idx]
    oc = orig_conf[idx]

    sycophantic = correct
    corrective = ~correct

    s = effect_scale

    conf_after = np.where(
        sycophantic,
        oc + rng.normal(0, 0.08, m) + s * 0.06 * p_syco,
        oc - s * 0.12 + rng.normal(0, 0.10, m),
    )
    conf_after = np.clip(conf_after, 0.02, 0.99)
    conf_delta = conf_after - oc

    hedge_rate = 0.6 + s * (0.25 * corrective - 0.10 * sycophantic) + 0.15 * pb
    hedge_count = rng.poisson(np.clip(hedge_rate, 0.05, None))

    validate_rate = 0.4 + s * (0.30 * sycophantic - 0.10 * corrective)
    validate_used = rng.random(m) < np.clip(validate_rate, 0.02, 0.98)

    length_delta = rng.normal(0, 15, m) + s * (10 * sycophantic - 8 * corrective) + s * 8 * p_syco
    hedge_latency = np.clip(
        0.6 + s * (0.20 * corrective - 0.10 * sycophantic) + rng.normal(0, 0.15, m), 0.05, None
    )

    df = pd.DataFrame(
        {
            "pushback_strength": pb,
            "orig_conf": oc,
            "conf_after": conf_after,
            "conf_delta": conf_delta,
            "hedge_count": hedge_count,
            "validate_used": validate_used.astype(float),
            "length_delta": length_delta,
            "hedge_latency": hedge_latency,
            "sycophantic": sycophantic.astype(int),
        }
    )
    return df


FEATURES = [
    "pushback_strength",
    "orig_conf",
    "conf_after",
    "conf_delta",
    "hedge_count",
    "validate_used",
    "length_delta",
    "hedge_latency",
]


def evaluate(df, seed=0):
    X, y = df[FEATURES], df["sycophantic"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    out = {}
    naive_pred = np.ones(len(yte))
    out["naive_flag"] = dict(
        auc=0.5, precision=precision_score(yte, naive_pred), recall=recall_score(yte, naive_pred),
        f1=f1_score(yte, naive_pred),
    )

    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    lr_p = lr.predict_proba(Xte)[:, 1]
    out["logreg"] = dict(
        auc=roc_auc_score(yte, lr_p), precision=precision_score(yte, lr_p > 0.5),
        recall=recall_score(yte, lr_p > 0.5), f1=f1_score(yte, lr_p > 0.5),
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed).fit(Xtr, ytr)
    rf_p = rf.predict_proba(Xte)[:, 1]
    out["rf"] = dict(
        auc=roc_auc_score(yte, rf_p), precision=precision_score(yte, rf_p > 0.5),
        recall=recall_score(yte, rf_p > 0.5), f1=f1_score(yte, rf_p > 0.5),
    )
    out["rf_importances"] = dict(zip(FEATURES, rf.feature_importances_.round(3).tolist()))

    # downstream policy simulation
    yte_arr = yte.values
    no_audit_acc = (yte_arr == 0).mean()  # trusting final answer is wrong whenever sycophantic
    naive_audit_acc = (yte_arr == 1).mean()  # reverting is right whenever sycophantic
    rf_pred = (rf_p > 0.5).astype(int)
    correct_final = (yte_arr == 0)
    correct_revert = (yte_arr == 1)
    smart_correct = np.where(rf_pred == 1, correct_revert, correct_final)
    out["policy"] = dict(
        no_audit=float(no_audit_acc), naive_audit=float(naive_audit_acc),
        smart_audit=float(smart_correct.mean()), rf_flag_rate=float(rf_pred.mean()),
    )
    return out


def main():
    results = {}

    # 3.1/3.2/3.4 main run at seed 0 (as in v1)
    df0 = simulate_dataset(seed=0)
    results["main"] = evaluate(df0, seed=0)
    results["n_total"] = int(len(df0))

    # feature-group ablation
    groups = {
        "pushback_only": ["pushback_strength"],
        "confidence_only": ["orig_conf", "conf_after", "conf_delta"],
        "language_only": ["hedge_count", "validate_used", "length_delta", "hedge_latency"],
        "confidence_language": ["orig_conf", "conf_after", "conf_delta", "hedge_count",
                                 "validate_used", "length_delta", "hedge_latency"],
    }
    abl = {}
    for name, feats in groups.items():
        X, y = df0[feats], df0["sycophantic"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0).fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:, 1]
        abl[name] = float(roc_auc_score(yte, p))
    abl["full_rf"] = results["main"]["rf"]["auc"]
    results["feature_ablation"] = abl

    # base-rate sweep
    sweep = {}
    for acc in [0.30, 0.50, 0.72, 0.90]:
        d = simulate_dataset(base_acc=acc, seed=0)
        r = evaluate(d, seed=0)
        sweep[str(acc)] = dict(rate=float(d["sycophantic"].mean()), auc=r["rf"]["auc"], f1=r["rf"]["f1"])
    results["base_rate_sweep"] = sweep

    # NEW: multi-seed variance of headline RF AUC / downstream accuracy
    seeds = list(range(10))
    aucs, accs = [], []
    for sd in seeds:
        d = simulate_dataset(seed=sd)
        r = evaluate(d, seed=sd)
        aucs.append(r["rf"]["auc"])
        accs.append(r["policy"]["smart_audit"])
    results["seed_variance"] = dict(
        seeds=seeds,
        rf_auc_mean=float(np.mean(aucs)), rf_auc_std=float(np.std(aucs)),
        smart_acc_mean=float(np.mean(accs)), smart_acc_std=float(np.std(accs)),
    )

    # NEW: coefficient-attenuation sensitivity sweep (effect_scale from 0 to 1)
    scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    sens = {}
    for s in scales:
        d = simulate_dataset(seed=0, effect_scale=s)
        r = evaluate(d, seed=0)
        sens[str(s)] = dict(auc=r["rf"]["auc"], f1=r["rf"]["f1"])
    results["effect_scale_sensitivity"] = sens

    # NEW: 5-seed feature-importance rank stability (Sec 3.2 claim, now
    # actually computed and persisted rather than only asserted in prose)
    rank_seeds = list(range(5))
    per_seed_importances = []
    for sd in rank_seeds:
        d = simulate_dataset(seed=sd)
        r = evaluate(d, seed=sd)
        per_seed_importances.append(r["rf_importances"])
    top_feature_per_seed = [
        max(imp, key=imp.get) for imp in per_seed_importances
    ]
    results["feature_rank_stability"] = dict(
        seeds=rank_seeds,
        per_seed_importances=per_seed_importances,
        top_feature_per_seed=top_feature_per_seed,
    )

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
