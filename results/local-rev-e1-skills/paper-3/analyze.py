import json
import numpy as np
from scipy import stats

with open("sim_results.json") as f:
    data = json.load(f)

results = data["results"]
params = data["params"]
LAMBDAS = params["LAMBDAS"]
EPSILONS = params["EPSILONS"]

# organize into arrays
by_cell = {}
for r in results:
    key = (r["lambda"], r["epsilon"])
    by_cell.setdefault(key, []).append(r)

print("=== Grid means: final_var (rows=lambda, cols=epsilon) ===")
header = "lambda\\eps  " + "  ".join(f"{e:5.2f}" for e in EPSILONS)
print(header)
grid_mean = {}
grid_sem = {}
for lam in LAMBDAS:
    row = []
    for eps in EPSILONS:
        vals = np.array([r["final_var"] for r in by_cell[(lam, eps)]])
        grid_mean[(lam, eps)] = vals.mean()
        grid_sem[(lam, eps)] = vals.std(ddof=1) / np.sqrt(len(vals))
        row.append(vals.mean())
    print(f"{lam:8.2f}  " + "  ".join(f"{v:5.3f}" for v in row))

print()
print("=== Grid means: mean_exposure_var (mediator) ===")
for lam in LAMBDAS:
    row = []
    for eps in EPSILONS:
        vals = np.array([r["mean_exposure_var"] for r in by_cell[(lam, eps)]])
        row.append(vals.mean())
    print(f"{lam:8.2f}  " + "  ".join(f"{v:5.3f}" for v in row))

# Main effect: lambda=1.0 vs lambda=0.0 paired by (seed, epsilon)
print()
print("=== Paired lambda=1.0 vs lambda=0.0 (paired by seed, within each epsilon) ===")
overall_diffs = []
for eps in EPSILONS:
    v0 = {r["seed"]: r["final_var"] for r in by_cell[(0.0, eps)]}
    v1 = {r["seed"]: r["final_var"] for r in by_cell[(1.0, eps)]}
    seeds = sorted(v0.keys())
    a = np.array([v1[s] for s in seeds])
    b = np.array([v0[s] for s in seeds])
    diff = a - b
    overall_diffs.extend(diff.tolist())
    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(a, b)
    except ValueError:
        w, p_w = np.nan, np.nan
    print(f"eps={eps:.2f}: mean_diff(engagement-random)={diff.mean():+.4f}  "
          f"paired-t p={p_t:.4g}  wilcoxon p={p_w:.4g}  n={len(seeds)}")

overall_diffs = np.array(overall_diffs)
print(f"\nPooled across all epsilon (n={len(overall_diffs)}): "
      f"mean diff = {overall_diffs.mean():+.4f}, "
      f"paired-t p={stats.ttest_1samp(overall_diffs, 0).pvalue:.3g}")

# Regression: final_var ~ lambda + epsilon (+ interaction), OLS via numpy
print()
print("=== OLS: final_var ~ lambda + epsilon + lambda*epsilon ===")
lam_arr = np.array([r["lambda"] for r in results])
eps_arr = np.array([r["epsilon"] for r in results])
y = np.array([r["final_var"] for r in results])
X = np.column_stack([np.ones_like(lam_arr), lam_arr, eps_arr, lam_arr * eps_arr])
beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
resid = y - yhat
n, k = X.shape
sigma2 = (resid @ resid) / (n - k)
cov = sigma2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
tvals = beta / se
pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=n - k))
names = ["intercept", "lambda", "epsilon", "lambda*epsilon"]
for name, b_, s_, t_, p_ in zip(names, beta, se, tvals, pvals):
    print(f"{name:16s} coef={b_:+.4f}  se={s_:.4f}  t={t_:.2f}  p={p_:.3g}")
r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
print(f"R^2 = {r2:.4f}, n={n}")

# Mediation: correlation between mean_exposure_var and final_var, controlling for lambda, epsilon
print()
print("=== Mediation check: partial correlation(final_var, mean_exposure_var | lambda, epsilon) ===")
med = np.array([r["mean_exposure_var"] for r in results])
# residualize both final_var and mediator on [1, lambda, epsilon]
Xc = np.column_stack([np.ones_like(lam_arr), lam_arr, eps_arr])
def residualize(v):
    b, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ b
y_res = residualize(y)
med_res = residualize(med)
r_partial, p_partial = stats.pearsonr(y_res, med_res)
print(f"partial r = {r_partial:.4f}, p = {p_partial:.3g}, n={n}")

# simple correlation too
r_simple, p_simple = stats.pearsonr(med, y)
print(f"simple (unadjusted) correlation(mediator, final_var) = {r_simple:.4f}, p={p_simple:.3g}")

# Fine-resolution check around eps=0.2 "threshold" already covered by 5-pt grid;
# report slope of final_var vs epsilon at lambda=1.0 and lambda=0.0 to check for
# a smooth trend vs a sharp jump.
print()
print("=== Per-lambda variance vs epsilon (checking for sharp jump vs smooth trend) ===")
for lam in [0.0, 1.0]:
    vals = [grid_mean[(lam, eps)] for eps in EPSILONS]
    diffs = np.diff(vals)
    print(f"lambda={lam}: means={['%.3f'%v for v in vals]}  step-diffs={['%.3f'%d for d in diffs]}")

print(f"\nsim elapsed_sec (from sim.py) = {params['elapsed_sec']:.1f}")
