"""
Behavioral audit protocol for detecting "confidence sycophancy" in AI-assistant
transcripts, using only interaction-observable features (no access to ground
truth correctness at detection time).

This script:
  1. Simulates a population of AI-assistant "policies" that vary in their
     propensity to flip a correct answer to an incorrect one under user
     pushback (sycophancy propensity), with parameters set from ranges
     reported in the sycophancy literature (see paper References).
  2. Generates per-interaction behavioral feature vectors (confidence
     trajectory, hedging language, answer-change, pushback strength, etc.)
     that an external auditor could observe WITHOUT knowing ground truth.
  3. Trains and evaluates classifiers (Logistic Regression, Random Forest)
     that try to detect sycophantic flips from these behavioral features
     alone, compared against a naive "flag if answer changed" baseline.
  4. Runs an ablation over feature groups and over population sycophancy
     rate, and estimates the downstream decision-accuracy benefit of using
     the audit tool as a warning filter.

All numbers reported in the paper come from actually running this script.
"""
import json
import time
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

RNG_SEED = 0
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Simulator
# ---------------------------------------------------------------------------
# Sycophancy propensity per assistant persona is drawn from a Beta
# distribution centered so the population mean flip-rate under pushback is
# ~35-45%, consistent with reported ranges of 20-70%+ flip rates across
# studies (see References: Wei et al. 2024; Perez et al. 2022; "Invisible
# Saboteurs" 2025; "Rational Analysis of Sycophantic AI" 2026).

N_INTERACTIONS = 6000
HEDGE_WORDS = ["maybe", "perhaps", "I think", "possibly", "it's likely",
               "I'm not entirely sure", "could be"]
VALIDATE_PHRASES = ["you're right", "great point", "I understand your concern",
                     "that's a fair point", "I see what you mean"]


def simulate_dataset(n, seed=0, flip_bias=0.0, syc_beta=(2.2, 3.0), base_accuracy=0.72,
                      effect_scale=1.0):
    # effect_scale multiplies every class-conditional (sycophantic vs.
    # corrective) offset term below. effect_scale=1.0 is the paper's
    # primary specification; effect_scale=0.0 removes all class-conditional
    # signal from the behavioral features, leaving only shared confounders
    # (persona_sycophancy, pushback_strength) -- used in Sec 3.5 to measure
    # how much of the headline AUC depends on the magnitude of these
    # hand-set offsets versus the audit framing itself.
    r = np.random.default_rng(seed)
    persona_sycophancy = r.beta(*syc_beta, size=n)          # population propensity in [0,1]
    pushback_strength = r.uniform(0.1, 1.0, size=n)         # mild -> aggressive disagreement
    orig_correct = r.uniform(size=n) < base_accuracy        # base task accuracy
    orig_conf = np.clip(r.normal(0.78, 0.12, size=n), 0.3, 0.99)

    # Probability assistant flips its answer under pushback: rises with
    # persona sycophancy and pushback strength, independent of correctness
    # (the defining, harmful property of sycophancy). flip_bias shifts the
    # overall population flip rate for the rate-sweep ablation.
    flip_logit = -1.6 + flip_bias + 3.2 * persona_sycophancy + 1.4 * pushback_strength \
                 + r.normal(0, 0.35, size=n)
    p_flip = 1 / (1 + np.exp(-flip_logit))
    answer_changed = r.uniform(size=n) < p_flip

    # A flip is "sycophantic" (harmful, the audit target) only when it moves
    # a correct answer to an incorrect one under social pressure with no new
    # evidence presented. Among flips, the complement (incorrect -> correct)
    # is a genuine, beneficial correction.
    sycophantic = answer_changed & orig_correct
    corrective = answer_changed & (~orig_correct)

    # Behavioral hypothesis (grounded in the "reflexive agreement" /
    # calibration-collapse literature cited in References): a sycophantic
    # capitulation is a fast, low-deliberation reversal accompanied by
    # social validation language and an immediate confidence rebound to
    # roughly its old level; a genuine correction shows more hedging
    # (deliberation) before landing on the new answer, less validation
    # language, and a more gradual confidence increase. Both signals are
    # noisy and overlapping, not deterministic, so the audit task is hard
    # but not impossible -- exactly the setting a real auditor would face.
    es = effect_scale
    conf_after = np.where(
        answer_changed,
        np.clip(orig_conf
                 + np.where(sycophantic, r.normal(0.0, 0.08, n), r.normal(-0.12 * es, 0.10, n))
                 + 0.06 * persona_sycophancy, 0.15, 0.99),
        np.clip(orig_conf + r.normal(0, 0.04, size=n), 0.2, 0.99),
    )
    conf_delta = conf_after - orig_conf

    # Hedging language: corrective changes involve more visible deliberation.
    hedge_rate = np.clip(
        0.15 + 0.35 * persona_sycophancy + 0.15 * pushback_strength
        + es * (0.25 * corrective - 0.10 * sycophantic), 0, 0.95)
    hedge_count = r.binomial(6, hedge_rate)

    # Validating/agreement phrases: far more common in sycophantic capitulation.
    validate_rate = np.clip(
        0.10 + 0.45 * persona_sycophancy + es * (0.30 * sycophantic - 0.10 * corrective), 0, 0.95)
    validate_used = r.uniform(size=n) < validate_rate

    # Response length change (sycophantic responses often pad with
    # apologetic/validating text).
    length_delta = (r.normal(8 * persona_sycophancy, 15, size=n) + 20 * validate_used
                     + es * (10 * sycophantic - 8 * corrective))

    # Latency to first hedge word (proxy; lower = faster/more reflexive).
    # Sycophantic capitulation is reflexive (low latency); genuine
    # correction takes more deliberation (higher latency) before hedging.
    hedge_latency = np.where(
        hedge_count > 0,
        np.clip(r.beta(2, 2 + 4 * (1 - persona_sycophancy), size=n)
                 + es * (0.20 * corrective - 0.10 * sycophantic), 0, 1),
        1.0)

    X = np.column_stack([
        pushback_strength,
        orig_conf,
        conf_after,
        conf_delta,
        answer_changed.astype(float),
        hedge_count,
        validate_used.astype(float),
        length_delta,
        hedge_latency,
    ])
    feature_names = ["pushback_strength", "orig_conf", "conf_after", "conf_delta",
                      "answer_changed", "hedge_count", "validate_used",
                      "length_delta", "hedge_latency"]
    y = sycophantic.astype(int)
    meta = dict(orig_correct=orig_correct, persona_sycophancy=persona_sycophancy)
    return X, y, feature_names, meta


def simulate_dataset_multiplicative(n, seed=0, syc_beta=(2.2, 3.0), base_accuracy=0.72):
    """Alternate functional form for the robustness check asked for by a
    reviewer: instead of ADDITIVE class-conditional offsets (the paper's
    main construction), class membership acts MULTIPLICATIVELY on rate
    parameters, and confidence rebound is modeled as a multiplicative pull
    toward orig_conf rather than an additive shift. This is a genuinely
    different generative mechanism, not a relabeled version of the same
    one, and is used only to check whether the qualitative ordering
    (RF > confidence-only > naive) survives a different functional form,
    not to re-derive an AUC number that means anything on its own."""
    r = np.random.default_rng(seed)
    persona_sycophancy = r.beta(*syc_beta, size=n)
    pushback_strength = r.uniform(0.1, 1.0, size=n)
    orig_correct = r.uniform(size=n) < base_accuracy
    orig_conf = np.clip(r.normal(0.78, 0.12, size=n), 0.3, 0.99)

    flip_logit = -1.6 + 3.2 * persona_sycophancy + 1.4 * pushback_strength + r.normal(0, 0.35, n)
    p_flip = 1 / (1 + np.exp(-flip_logit))
    answer_changed = r.uniform(size=n) < p_flip
    sycophantic = answer_changed & orig_correct
    corrective = answer_changed & (~orig_correct)

    # Multiplicative confidence pull: sycophantic pulls conf_after toward
    # orig_conf (rebound), corrective pulls it toward a fixed lower anchor,
    # each via a multiplicative blend weight rather than an additive delta.
    rebound_w = np.where(sycophantic, 0.85, np.where(corrective, 0.25, 0.6))
    anchor = np.where(corrective, 0.45, orig_conf)
    conf_after = np.where(
        answer_changed,
        np.clip(rebound_w * orig_conf + (1 - rebound_w) * anchor + r.normal(0, 0.06, n), 0.15, 0.99),
        np.clip(orig_conf + r.normal(0, 0.04, n), 0.2, 0.99))
    conf_delta = conf_after - orig_conf

    # Multiplicative hedge-rate: a class multiplier scales the base rate
    # rather than adding a term.
    hedge_mult = np.where(sycophantic, 0.35, np.where(corrective, 2.2, 1.0))
    hedge_rate = np.clip((0.15 + 0.35 * persona_sycophancy + 0.15 * pushback_strength) * hedge_mult, 0, 0.95)
    hedge_count = r.binomial(6, hedge_rate)

    validate_mult = np.where(sycophantic, 2.5, np.where(corrective, 0.3, 1.0))
    validate_rate = np.clip((0.10 + 0.45 * persona_sycophancy) * validate_mult, 0, 0.95)
    validate_used = r.uniform(size=n) < validate_rate

    length_mult = np.where(sycophantic, 1.3, np.where(corrective, 0.9, 1.0))
    length_delta = (r.normal(8 * persona_sycophancy, 15, n) + 20 * validate_used) * length_mult

    hedge_latency_mult = np.where(sycophantic, 0.6, np.where(corrective, 1.6, 1.0))
    hedge_latency = np.where(
        hedge_count > 0,
        np.clip(r.beta(2, 2 + 4 * (1 - persona_sycophancy), n) * hedge_latency_mult, 0, 1), 1.0)

    X = np.column_stack([pushback_strength, orig_conf, conf_after, conf_delta,
                          answer_changed.astype(float), hedge_count, validate_used.astype(float),
                          length_delta, hedge_latency])
    feature_names = ["pushback_strength", "orig_conf", "conf_after", "conf_delta",
                      "answer_changed", "hedge_count", "validate_used", "length_delta", "hedge_latency"]
    y = sycophantic.astype(int)
    meta = dict(orig_correct=orig_correct, persona_sycophancy=persona_sycophancy)
    return X, y, feature_names, meta


def eval_model(name, y_true, y_score, y_pred):
    return dict(
        name=name,
        auc=float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        base_rate=float(np.mean(y_true)),
    )


def run_main_experiment():
    X, y, names, meta = simulate_dataset(N_INTERACTIONS, seed=RNG_SEED)

    # The interesting audit task is NOT "did the answer change" (that is
    # directly observable and would make the task trivial, since our
    # sycophantic label is a subset of answer-changed cases by
    # construction). The interesting task is: GIVEN that the assistant
    # changed its answer under pushback, was that change HARMFUL
    # (sycophantic: correct -> incorrect) or BENEFICIAL (a genuine
    # correction: incorrect -> correct/incorrect-to-different-incorrect is
    # excluded since we only track correct-origin flips as our positive
    # class; the complement within "changed" is the negative class of
    # interest here, i.e. changes made by originally-incorrect answers,
    # which are corrective by construction in this simulation).
    # This mirrors the real deployment problem: an over-eager "flag every
    # answer change" policy over-warns on legitimate self-corrections and
    # erodes user trust, so the auditor must discriminate using behavioral
    # signals alone -- WITHOUT access to ground-truth correctness.
    ans_idx = names.index("answer_changed")
    changed_mask = X[:, ans_idx] > 0.5
    Xc, yc = X[changed_mask], y[changed_mask]
    # within changed cases, drop answer_changed (constant=1, uninformative)
    keep = [i for i, n in enumerate(names) if n != "answer_changed"]
    Xc = Xc[:, keep]
    cnames = [names[i] for i in keep]

    orig_indices = np.arange(len(y))[changed_mask]
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        Xc, yc, orig_indices, test_size=0.3, random_state=RNG_SEED, stratify=yc)

    results = {}

    # Baseline A: "always revert" -- treat every answer-change as
    # sycophantic (the naive policy a cautious product team might ship).
    naive_pred = np.ones_like(yte)
    results["naive_always_flag"] = eval_model("naive_always_flag", yte, naive_pred.astype(float), naive_pred)

    # Baseline B: majority-class predictor (predict the more common label).
    majority = int(np.mean(ytr) > 0.5)
    maj_pred = np.full_like(yte, majority)
    results["majority_class"] = eval_model("majority_class", yte, maj_pred.astype(float), maj_pred)

    # Logistic Regression on behavioral features only (confidence
    # trajectory, hedging language, pushback strength) -- no ground truth.
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)[:, 1]
    pred = (proba > 0.5).astype(int)
    results["logreg_behavioral"] = eval_model("logreg_behavioral", yte, proba, pred)

    # Random Forest on the same behavioral features.
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RNG_SEED)
    rf.fit(Xtr, ytr)
    proba_rf = rf.predict_proba(Xte)[:, 1]
    pred_rf = (proba_rf > 0.5).astype(int)
    results["rf_behavioral"] = eval_model("rf_behavioral", yte, proba_rf, pred_rf)

    importances = dict(zip(cnames, rf.feature_importances_.round(4).tolist()))
    names = cnames  # used by ablation section below

    # -------------------------------------------------------------------
    # Ablation: feature groups
    # -------------------------------------------------------------------
    groups = {
        "confidence_only": ["orig_conf", "conf_after", "conf_delta"],
        "language_only": ["hedge_count", "validate_used", "length_delta", "hedge_latency"],
        "pushback_only": ["pushback_strength"],
        "confidence+language": ["orig_conf", "conf_after", "conf_delta",
                                 "hedge_count", "validate_used", "length_delta", "hedge_latency"],
    }
    ablation = {}
    for gname, feats in groups.items():
        fidx = [names.index(f) for f in feats]
        m = LogisticRegression(max_iter=1000)
        m.fit(Xtr[:, fidx], ytr)
        p = m.predict_proba(Xte[:, fidx])[:, 1]
        pr = (p > 0.5).astype(int)
        ablation[gname] = eval_model(gname, yte, p, pr)

    # -------------------------------------------------------------------
    # Ablation: performance vs. population sycophancy rate
    # -------------------------------------------------------------------
    rate_sweep = {}
    for base_acc in [0.3, 0.5, 0.72, 0.9]:
        Xr, yr, namesr, _ = simulate_dataset(4000, seed=RNG_SEED + 7, base_accuracy=base_acc)
        ans_idx_r = namesr.index("answer_changed")
        mask_r = Xr[:, ans_idx_r] > 0.5
        keep_r = [i for i, n in enumerate(namesr) if n != "answer_changed"]
        Xr, yr = Xr[mask_r][:, keep_r], yr[mask_r]
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            Xr, yr, test_size=0.3, random_state=RNG_SEED,
            stratify=yr if len(set(yr)) > 1 else None)
        m = LogisticRegression(max_iter=1000)
        m.fit(Xtr_r, ytr_r)
        p = m.predict_proba(Xte_r)[:, 1]
        pr = (p > 0.5).astype(int)
        tag = f"base_accuracy={base_acc:.2f}_syc_rate={np.mean(yr):.2f}"
        rate_sweep[tag] = eval_model(tag, yte_r, p, pr)

    # -------------------------------------------------------------------
    # Downstream harm-reduction estimate, restricted to the population of
    # interactions where the assistant changed its answer under pushback
    # (this is exactly the population the audit tool acts on). Three
    # policies for what the user ends up trusting:
    #   (1) no audit: always trust the assistant's final (post-pushback)
    #       answer -- correct iff the change was corrective (yte==0).
    #   (2) naive audit: always revert to the pre-pushback answer whenever
    #       ANY change occurred -- correct iff the original answer was
    #       correct (orig_correct_te), regardless of yte.
    #   (3) smart (RF) audit: revert only when the classifier predicts
    #       sycophantic; otherwise trust the final answer.
    # -------------------------------------------------------------------
    orig_correct_te = meta["orig_correct"][idx_te].astype(int)
    final_correct_no_audit = (1 - yte)                 # policy (1)
    final_correct_naive_audit = orig_correct_te         # policy (2)
    warn = pred_rf                                      # RF predictions on test set
    final_correct_with_audit = np.where(warn == 1, orig_correct_te, final_correct_no_audit)  # policy (3)

    downstream = dict(
        accuracy_no_audit=float(np.mean(final_correct_no_audit)),
        accuracy_naive_always_revert=float(np.mean(final_correct_naive_audit)),
        accuracy_with_rf_audit=float(np.mean(final_correct_with_audit)),
        warn_rate=float(np.mean(warn)),
        n_test=int(len(yte)),
    )

    # -------------------------------------------------------------------
    # Ablation: effect-size attenuation sweep. Scales every class-conditional
    # (sycophantic vs. corrective) offset term in the four language/latency
    # features by effect_scale in [1.0 .. 0.0], holding everything else
    # (including the shared persona_sycophancy/pushback_strength confounders)
    # fixed. At effect_scale=0.0 the behavioral features carry NO
    # class-conditional signal beyond what is jointly induced by the shared
    # confounders, isolating how much of the headline AUC is attributable to
    # the specific magnitudes of the hand-set offsets versus the audit
    # framing / classifier itself.
    # -------------------------------------------------------------------
    effect_sweep = {}
    for es in [1.0, 0.75, 0.5, 0.25, 0.0]:
        Xe, ye, namese, _ = simulate_dataset(N_INTERACTIONS, seed=RNG_SEED, effect_scale=es)
        ans_idx_e = namese.index("answer_changed")
        mask_e = Xe[:, ans_idx_e] > 0.5
        keep_e = [i for i, n in enumerate(namese) if n != "answer_changed"]
        Xe, ye = Xe[mask_e][:, keep_e], ye[mask_e]
        Xtr_e, Xte_e, ytr_e, yte_e = train_test_split(
            Xe, ye, test_size=0.3, random_state=RNG_SEED, stratify=ye)
        m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RNG_SEED)
        m.fit(Xtr_e, ytr_e)
        p = m.predict_proba(Xte_e)[:, 1]
        pr = (p > 0.5).astype(int)
        effect_sweep[f"effect_scale={es:.2f}"] = eval_model(f"es={es}", yte_e, p, pr)

    # -------------------------------------------------------------------
    # Downstream policy simulation AT EACH effect_scale (new): the sweep
    # above shows classifier AUC/F1 degrade as offsets are attenuated, but
    # reviewers asked directly whether the *downstream decision-accuracy*
    # gain over the naive "revert on any change" policy survives at more
    # conservative (attenuated) effect sizes, not just whether AUC does.
    # -------------------------------------------------------------------
    downstream_effect_sweep = {}
    for es in [1.0, 0.75, 0.5, 0.25, 0.0]:
        Xe, ye, namese, metae = simulate_dataset(N_INTERACTIONS, seed=RNG_SEED, effect_scale=es)
        ans_idx_e = namese.index("answer_changed")
        mask_e = Xe[:, ans_idx_e] > 0.5
        keep_e = [i for i, n in enumerate(namese) if n != "answer_changed"]
        Xe2, ye2 = Xe[mask_e][:, keep_e], ye[mask_e]
        idx_e = np.arange(len(ye))[mask_e]
        Xtr_e, Xte_e, ytr_e, yte_e, itr_e, ite_e = train_test_split(
            Xe2, ye2, idx_e, test_size=0.3, random_state=RNG_SEED, stratify=ye2)
        m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RNG_SEED)
        m.fit(Xtr_e, ytr_e)
        pr = (m.predict_proba(Xte_e)[:, 1] > 0.5).astype(int)
        oc_te = metae["orig_correct"][ite_e].astype(int)
        no_audit = float(np.mean(1 - yte_e))
        naive_audit = float(np.mean(oc_te))
        smart = float(np.mean(np.where(pr == 1, oc_te, 1 - yte_e)))
        downstream_effect_sweep[f"effect_scale={es:.2f}"] = dict(
            accuracy_no_audit=no_audit, accuracy_naive_always_revert=naive_audit,
            accuracy_with_rf_audit=smart, warn_rate=float(np.mean(pr)))

    # -------------------------------------------------------------------
    # Multi-seed variance: repeat the full main-experiment pipeline (data
    # generation, train/test split, RF fit, downstream policy simulation)
    # across 10 independent seeds to report mean +/- std rather than a
    # single-run point estimate for the headline numbers.
    # -------------------------------------------------------------------
    seed_aucs, seed_f1s, seed_downstream = [], [], []
    for s in range(10):
        Xs, ys, namess, metas = simulate_dataset(N_INTERACTIONS, seed=s)
        ans_idx_s = namess.index("answer_changed")
        mask_s = Xs[:, ans_idx_s] > 0.5
        keep_s = [i for i, n in enumerate(namess) if n != "answer_changed"]
        Xs2, ys2 = Xs[mask_s][:, keep_s], ys[mask_s]
        idx_s = np.arange(len(ys))[mask_s]
        Xtr_s, Xte_s, ytr_s, yte_s, itr_s, ite_s = train_test_split(
            Xs2, ys2, idx_s, test_size=0.3, random_state=s, stratify=ys2)
        m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=s)
        m.fit(Xtr_s, ytr_s)
        p = m.predict_proba(Xte_s)[:, 1]
        pr = (p > 0.5).astype(int)
        seed_aucs.append(roc_auc_score(yte_s, p))
        seed_f1s.append(f1_score(yte_s, pr))
        oc_te = metas["orig_correct"][ite_s].astype(int)
        no_audit = np.mean(1 - yte_s)
        naive_audit = np.mean(oc_te)
        smart = np.mean(np.where(pr == 1, oc_te, 1 - yte_s))
        seed_downstream.append((no_audit, naive_audit, smart))
    seed_downstream = np.array(seed_downstream)
    multiseed = dict(
        n_seeds=10,
        auc_mean=float(np.mean(seed_aucs)), auc_std=float(np.std(seed_aucs)),
        f1_mean=float(np.mean(seed_f1s)), f1_std=float(np.std(seed_f1s)),
        downstream_no_audit_mean=float(seed_downstream[:, 0].mean()),
        downstream_no_audit_std=float(seed_downstream[:, 0].std()),
        downstream_naive_mean=float(seed_downstream[:, 1].mean()),
        downstream_naive_std=float(seed_downstream[:, 1].std()),
        downstream_smart_mean=float(seed_downstream[:, 2].mean()),
        downstream_smart_std=float(seed_downstream[:, 2].std()),
    )

    # -------------------------------------------------------------------
    # Robustness (new): does RF > confidence-only > naive survive a
    # DIFFERENT functional form (multiplicative class effects instead of
    # additive offsets), not just attenuated magnitudes of the same form?
    # -------------------------------------------------------------------
    Xm, ym, namesm, _ = simulate_dataset_multiplicative(N_INTERACTIONS, seed=RNG_SEED)
    ans_idx_m = namesm.index("answer_changed")
    mask_m = Xm[:, ans_idx_m] > 0.5
    keep_m = [i for i, n in enumerate(namesm) if n != "answer_changed"]
    Xm2, ym2 = Xm[mask_m][:, keep_m], ym[mask_m]
    namesm2 = [namesm[i] for i in keep_m]
    Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
        Xm2, ym2, test_size=0.3, random_state=RNG_SEED, stratify=ym2)
    naive_pred_m = np.ones_like(yte_m)
    naive_auc_m = roc_auc_score(yte_m, naive_pred_m.astype(float)) if len(set(yte_m)) > 1 else float("nan")
    conf_idx_m = [namesm2.index(f) for f in ["orig_conf", "conf_after", "conf_delta"]]
    lr_conf_m = LogisticRegression(max_iter=1000)
    lr_conf_m.fit(Xtr_m[:, conf_idx_m], ytr_m)
    conf_auc_m = roc_auc_score(yte_m, lr_conf_m.predict_proba(Xte_m[:, conf_idx_m])[:, 1])
    rf_m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RNG_SEED)
    rf_m.fit(Xtr_m, ytr_m)
    rf_auc_m = roc_auc_score(yte_m, rf_m.predict_proba(Xte_m)[:, 1])
    alt_functional_form = dict(
        description="multiplicative class-conditional effects instead of additive offsets",
        naive_auc=float(naive_auc_m), confidence_only_auc=float(conf_auc_m), rf_auc=float(rf_auc_m),
        ordering_preserved=bool(rf_auc_m > conf_auc_m > naive_auc_m + 1e-9 or (naive_auc_m != naive_auc_m)),
    )

    out = dict(
        n_interactions=N_INTERACTIONS,
        seed=RNG_SEED,
        base_rate_train=float(np.mean(ytr)),
        base_rate_test=float(np.mean(yte)),
        main_results=results,
        rf_feature_importances=importances,
        ablation_feature_groups=ablation,
        ablation_rate_sweep=rate_sweep,
        ablation_effect_scale_sweep=effect_sweep,
        multiseed_variance=multiseed,
        downstream_harm_reduction=downstream,
        downstream_effect_scale_sweep=downstream_effect_sweep,
        robustness_alt_functional_form=alt_functional_form,
    )
    return out


if __name__ == "__main__":
    t0 = time.time()
    out = run_main_experiment()
    out["runtime_sec"] = round(time.time() - t0, 3)
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
