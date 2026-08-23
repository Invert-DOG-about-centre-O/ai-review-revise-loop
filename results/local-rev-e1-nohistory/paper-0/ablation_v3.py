"""
Round-2 review follow-ups, all cheap (<5s each):

(A) Fine-grained lambda sweep (step 0.01) on the continuous-belief protocol,
    to measure transition WIDTH (not just location) per skill level, and to
    check whether "discontinuity" is the right word at fine resolution.

(B) Sensitivity of the confidence-threshold progression to the specific
    linear confidence map h_prob = 0.5 +/- 0.5*acc: repeat with a concave
    (sqrt) map and a convex (squared) map that both keep h_prob in [0.5,1]
    and preserve monotonicity in acc, and check whether the low<medium<high
    collapse-point ORDERING survives.

(C) Alternative blending rule: sigmoid-blend instead of linear-blend, to see
    if the sharp-transition structural claim survives a change of mechanism
    (Limitations promised this and round1/round2 reviewers both flagged it
    as unrun).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RNG_SEED_BASE = 12345
N_SEEDS = 30
SKILL_LEVELS = {"low": 0.55, "medium": 0.70, "high": 0.85}
TRUST_LR = 0.15
TRUST_INIT = 0.5

def load_ai_predictions():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    p_ai = clf.predict_proba(X_test)[:, 1]
    ai_acc = clf.score(X_test, y_test)
    return p_ai, y_test, ai_acc

def acc_curve(difficulty, skill_peak):
    return skill_peak - (skill_peak - 0.5) * (difficulty / difficulty.max())

def human_judgment(rng, y_true, acc, conf_map):
    correct = rng.random(len(y_true)) < acc
    h_belief = np.where(correct, y_true, 1 - y_true)
    h_prob = np.where(correct, 0.5 + 0.5 * conf_map(acc), 0.5 - 0.5 * conf_map(acc))
    return h_belief, h_prob

def run_condition(p_ai, y_test, ai_acc, lam, skill_peak, seed, conf_map, blend):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    r0 = (p_ai > 0.5).astype(int)
    difficulty = 1 - np.abs(p_ai - 0.5) * 2
    acc = acc_curve(difficulty, skill_peak)
    h_belief, h_prob = human_judgment(rng, y_test, acc, conf_map)

    disagree = (h_belief != r0)
    if blend == "linear":
        p_shifted = np.where(disagree, (1 - lam) * p_ai + lam * h_prob, p_ai)
    elif blend == "sigmoid":
        # logit-space blend: pull AI's logit toward belief's logit by lam, then squash
        eps = 1e-6
        p_c = np.clip(p_ai, eps, 1 - eps)
        h_c = np.clip(h_prob, eps, 1 - eps)
        logit_ai = np.log(p_c / (1 - p_c))
        logit_h = np.log(h_c / (1 - h_c))
        logit_shift = (1 - lam) * logit_ai + lam * logit_h
        p_shifted = np.where(disagree, 1 / (1 + np.exp(-logit_shift)), p_ai)
    r1 = (p_shifted > 0.5).astype(int)

    trust = TRUST_INIT
    team_decision = np.zeros(n, dtype=int)
    for i in range(n):
        follow_ai = rng.random() < trust
        team_decision[i] = r1[i] if follow_ai else h_belief[i]
        trust = trust + TRUST_LR * (int(r1[i] == y_test[i]) - trust)
    team_acc = (team_decision == y_test).mean()
    return team_acc

def sweep(p_ai, y_test, ai_acc, lambdas, conf_map, blend):
    rows = []
    for skill_name, skill_peak in SKILL_LEVELS.items():
        for lam in lambdas:
            accs = [run_condition(p_ai, y_test, ai_acc, lam, skill_peak, RNG_SEED_BASE + s, conf_map, blend)
                    for s in range(N_SEEDS)]
            rows.append(dict(skill=skill_name, lam=round(float(lam), 3), team_acc_mean=np.mean(accs)))
    return pd.DataFrame(rows)

def collapse_point(df, skill, drop_thresh=0.03):
    sub = df[df.skill == skill].sort_values("lam").reset_index(drop=True)
    diffs = sub.team_acc_mean.diff()
    idx = diffs[diffs < -drop_thresh].index
    if len(idx) == 0:
        return None, None
    start = sub.lam.iloc[idx[0] - 1]
    end = sub.lam.iloc[idx[-1]]
    return start, end

def main():
    t0 = time.time()
    p_ai, y_test, ai_acc = load_ai_predictions()

    # (A) fine-grained lambda grid around the coarse collapse region
    fine_lambdas = np.round(np.arange(0.40, 0.85, 0.01), 3)
    fine_df = sweep(p_ai, y_test, ai_acc, fine_lambdas, conf_map=lambda a: a, blend="linear")
    fine_df.to_csv("results_fine_continuous.csv", index=False)

    print("=== (A) Fine-grained continuous-belief sweep: transition windows ===")
    widths = {}
    for skill in SKILL_LEVELS:
        start, end = collapse_point(fine_df, skill)
        widths[skill] = (start, end, None if start is None else round(end - start, 3))
        print(f"  {skill}: transition from lam={start} to lam={end}  (width={widths[skill][2]})")

    # (B) sensitivity of confidence map: sqrt (concave) and square (convex)
    print("\n=== (B) Sensitivity to confidence-mapping functional form ===")
    coarse_lambdas = np.round(np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]), 2)
    map_results = {}
    for name, fn in [("linear", lambda a: a), ("sqrt", lambda a: np.sqrt(a)), ("square", lambda a: a ** 2)]:
        df = sweep(p_ai, y_test, ai_acc, coarse_lambdas, conf_map=fn, blend="linear")
        cps = {}
        for skill in SKILL_LEVELS:
            start, end = collapse_point(df, skill, drop_thresh=0.03)
            cps[skill] = start
        map_results[name] = cps
        print(f"  {name:7s}: low={cps['low']}, medium={cps['medium']}, high={cps['high']}  "
              f"ordering_low<med<high={cps['low'] is not None and cps['medium'] is not None and cps['high'] is not None and cps['low'] >= cps['medium'] >= cps['high']}")

    # (C) alternative blending mechanism: sigmoid/logit-space blend
    print("\n=== (C) Alternative blend mechanism: logit-space (sigmoid) blend ===")
    sig_df = sweep(p_ai, y_test, ai_acc, coarse_lambdas, conf_map=lambda a: a, blend="sigmoid")
    sig_df.to_csv("results_sigmoid_blend.csv", index=False)
    for skill in SKILL_LEVELS:
        start, end = collapse_point(sig_df, skill, drop_thresh=0.03)
        print(f"  {skill}: collapse window lam={start}-{end}")
    print(sig_df.to_string(index=False))

    elapsed = time.time() - t0
    with open("run_log_v3.txt", "w") as f:
        f.write(f"elapsed_sec={elapsed:.2f}\n\n")
        f.write("(A) fine-grained continuous transition widths:\n")
        f.write(json.dumps({k: v for k, v in widths.items()}, default=str, indent=2) + "\n\n")
        f.write("(B) confidence-map sensitivity collapse points:\n")
        f.write(json.dumps(map_results, indent=2) + "\n\n")
        f.write("(C) sigmoid-blend results:\n")
        f.write(sig_df.to_string(index=False))
    print(f"\nElapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
