"""
Revision-3 follow-up experiments, addressing round3_review.json questions:
(1) Does a threshold-proximity-biased lottery (vs. uniform Bernoulli) change
    the targeted-vs-blind conclusion?
(2) How sensitive is the ~15-22% mitigation-cost figure to the accuracy
    definition (recall@p vs. point-biserial correlation with true skill vs.
    a fixed population-quantile threshold instead of empirical top-p)?
"""
import numpy as np
import json
import hashlib
from scipy import stats

N = 3000
DEFAULT_K = 8
DEFAULT_P = 0.20
N_SEEDS = 30


def hash_seed(*args):
    s = "|".join(str(a) for a in args)
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def draw_noise(rng, size, dist="gaussian"):
    if dist == "gaussian":
        return rng.standard_normal(size)
    elif dist == "student_t":
        x = rng.standard_t(df=3, size=size)
        return x / np.sqrt(3 / (3 - 2))
    raise ValueError(dist)


# ---------- (1) Boundary-biased lottery: block model, targeted vs blind ----------
def run_block_trial_boundary(seed, K=8, n_clusters=2, rho=0.8, p=0.2, dist="gaussian",
                              window_by_cluster=None):
    """Instead of a uniform-random lottery mask, deterministically re-randomize
    (Bernoulli(p) draw) the decisions of the `window` applicants whose score is
    CLOSEST to that institution's own acceptance cutoff -- i.e. concentrate the
    same total randomized-decision mass on the boundary-adjacent applicants
    rather than spreading it uniformly across all N."""
    rng = np.random.default_rng(seed)
    skill = rng.standard_normal(N)
    cluster_of = [i % n_clusters for i in range(K)]
    cluster_noise = {c: draw_noise(rng, N, dist) for c in range(n_clusters)}
    n_accept = max(1, int(round(p * N)))
    decisions = np.zeros((K, N), dtype=bool)
    window_by_cluster = window_by_cluster or {c: 0 for c in range(n_clusters)}
    for k in range(K):
        c = cluster_of[k]
        idio = draw_noise(rng, N, dist)
        score = skill + (np.sqrt(rho) * cluster_noise[c] + np.sqrt(1 - rho) * idio)
        order = np.argsort(-score)
        accept = np.zeros(N, dtype=bool)
        accept[order[:n_accept]] = True
        w = window_by_cluster[c]
        if w > 0:
            lo = max(0, n_accept - w // 2)
            hi = min(N, lo + w)
            boundary_idx = order[lo:hi]
            redraw = rng.random(len(boundary_idx)) < p
            accept[boundary_idx] = redraw
        decisions[k] = accept
    return skill, decisions


def homogenization_corr(decisions):
    K = decisions.shape[0]
    corrs = []
    for i in range(K):
        for j in range(i + 1, K):
            a, b = decisions[i].astype(float), decisions[j].astype(float)
            if a.std() == 0 or b.std() == 0:
                continue
            corrs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(corrs))


def accuracy_recall_at_p(skill, decisions, p):
    n = len(skill)
    n_top = max(1, int(round(p * n)))
    true_top = np.argsort(-skill)[:n_top]
    mask = np.zeros(n, dtype=bool)
    mask[true_top] = True
    return float(np.mean([decisions[k][mask].mean() for k in range(decisions.shape[0])]))


def block_boundary_sweep(window_by_cluster, rho=0.8, K=8, p=0.2, dist="gaussian",
                          n_seeds=30, tag=""):
    Hs, Accs = [], []
    for s in range(n_seeds):
        seed = hash_seed("blockbnd", tag, rho, dist, s, tuple(sorted(window_by_cluster.items())))
        skill, decisions = run_block_trial_boundary(seed, K=K, rho=rho, p=p, dist=dist,
                                                      window_by_cluster=window_by_cluster)
        Hs.append(homogenization_corr(decisions))
        Accs.append(accuracy_recall_at_p(skill, decisions, p))
    return float(np.mean(Hs)), float(np.mean(Accs))


def boundary_targeted_vs_blind(rho=0.8, K=8, p=0.2, dist="gaussian", n_seeds=30):
    # match total redrawn-decision mass: blind spreads window w across BOTH clusters
    # (institutions), targeted concentrates 2w on cluster 0 only, cluster 1 untouched.
    rows = []
    for w in [0, 100, 300, 600, 900, 1200]:
        H, Acc = block_boundary_sweep({0: w, 1: w}, rho, K, p, dist, n_seeds, tag="blind")
        rows.append(dict(mode="blind", window_cluster0=w, window_cluster1=w,
                          total_mass=2 * w, H=H, Acc=Acc))
    for w in [0, 200, 600, 1200, 1800, 2400]:
        H, Acc = block_boundary_sweep({0: w, 1: 0}, rho, K, p, dist, n_seeds, tag="targeted")
        rows.append(dict(mode="targeted", window_cluster0=w, window_cluster1=0,
                          total_mass=w, H=H, Acc=Acc))
    return rows


# ---------- (2) Sensitivity of mitigation-cost figure to accuracy metric ----------
def run_trial_flat(seed, K, rho, p, dist, eta):
    rng = np.random.default_rng(seed)
    skill = rng.standard_normal(N)
    shared = draw_noise(rng, N, dist)
    n_accept = max(1, int(round(p * N)))
    decisions = np.zeros((K, N), dtype=bool)
    for k in range(K):
        idio = draw_noise(rng, N, dist)
        score = skill + (np.sqrt(rho) * shared + np.sqrt(1 - rho) * idio)
        idx = np.argsort(-score)[:n_accept]
        accept = np.zeros(N, dtype=bool)
        accept[idx] = True
        if eta > 0:
            lottery_mask = rng.random(N) < eta
            lottery_draw = rng.random(N) < p
            accept = np.where(lottery_mask, lottery_draw, accept)
        decisions[k] = accept
    return skill, decisions


def accuracy_pointbiserial(skill, decisions):
    corrs = [stats.pointbiserialr(decisions[k].astype(int), skill).correlation
             for k in range(decisions.shape[0])]
    return float(np.mean(corrs))


def accuracy_fixed_quantile(skill, decisions, p):
    # ground truth = population quantile of N(0,1), not empirical top-p of this sample
    z = stats.norm.ppf(1 - p)
    mask = skill > z
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean([decisions[k][mask].mean() for k in range(decisions.shape[0])]))


def mitigation_cost_by_metric(rho, K=8, p=0.2, dist="gaussian", n_seeds=30, target_frac=0.52):
    """target_frac: fraction of the eta=0 H value we require the mitigation to
    reach (matches the paper's 'H returns to rho=0 baseline' logic in relative
    terms across metrics whose absolute scale differs)."""
    etas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Hs, Acc_recall, Acc_corr, Acc_fixedq = [], [], [], []
    for eta in etas:
        Hl, Ar, Ac, Af = [], [], [], []
        for s in range(n_seeds):
            seed = hash_seed("metricsens", rho, dist, eta, s)
            skill, decisions = run_trial_flat(seed, K, rho, p, dist, eta)
            Hl.append(homogenization_corr(decisions))
            Ar.append(accuracy_recall_at_p(skill, decisions, p))
            Ac.append(accuracy_pointbiserial(skill, decisions))
            Af.append(accuracy_fixed_quantile(skill, decisions, p))
        Hs.append(float(np.mean(Hl)))
        Acc_recall.append(float(np.mean(Ar)))
        Acc_corr.append(float(np.mean(Ac)))
        Acc_fixedq.append(float(np.mean(Af)))
    Hs_arr = np.array(Hs)
    target_H = 0.2955  # same rho=0 Gaussian baseline used throughout the paper
    if target_H >= Hs_arr[0]:
        idx0, frac = 0, 0.0
    elif target_H <= Hs_arr[-1]:
        idx0, frac = len(Hs_arr) - 2, 1.0
    else:
        idx0 = max(0, min(np.searchsorted(-Hs_arr, -target_H) - 1, len(Hs_arr) - 2))
        h1, h2 = Hs_arr[idx0], Hs_arr[idx0 + 1]
        frac = (target_H - h1) / (h2 - h1) if h2 != h1 else 0.0

    def interp(vals):
        v1, v2 = vals[idx0], vals[idx0 + 1]
        star = v1 + frac * (v2 - v1)
        cost = 100 * (vals[0] - star) / vals[0] if vals[0] != 0 else float("nan")
        return dict(val0=vals[0], val_star=float(star), rel_cost_pct=float(cost))

    return dict(rho=rho, etas=etas, Hs=Hs,
                recall_at_p=interp(Acc_recall),
                pointbiserial_corr=interp(Acc_corr),
                fixed_quantile=interp(Acc_fixedq))


def main():
    out = {}
    out["boundary_targeted_vs_blind"] = boundary_targeted_vs_blind()
    out["mitigation_cost_by_metric"] = [mitigation_cost_by_metric(rho) for rho in [0.5, 0.6, 0.8]]
    with open("extend2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
