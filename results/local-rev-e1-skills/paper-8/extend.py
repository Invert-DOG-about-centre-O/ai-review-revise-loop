"""
Revision-2 follow-up experiments, addressing round2_review.json:
(1) Why Student-t accuracy > Gaussian accuracy at matched rho (Q1).
(2) Targeted (block/vendor-cluster) decorrelation vs. blind lottery (Q2).
(3) Sensitivity of the ~22% mitigation-cost figure to rho (Q3).
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


# ---------- (1) Why does Student-t noise yield higher recall@p? ----------
# Hypothesis: heavy-tailed noise (df=3) is *more concentrated near 0* than
# Gaussian at matched variance (leptokurtic), so most rank comparisons near
# the selection threshold are LESS perturbed even though rare large outliers
# exist. Test directly: correlation of (skill+noise) rank with skill rank.
def rank_corr_check(n_seeds=30):
    out = {}
    for dist in ["gaussian", "student_t"]:
        corrs = []
        near_zero_frac = []
        for s in range(n_seeds):
            seed = hash_seed("rankcheck", dist, s)
            rng = np.random.default_rng(seed)
            skill = rng.standard_normal(N)
            noise = draw_noise(rng, N, dist)
            score = skill + noise  # sigma=1, single noise source (rho irrelevant to this check)
            corrs.append(stats.spearmanr(skill, score).correlation)
            near_zero_frac.append(float(np.mean(np.abs(noise) < 0.5)))
        out[dist] = dict(mean_spearman=float(np.mean(corrs)),
                          mean_frac_noise_within_0p5=float(np.mean(near_zero_frac)),
                          noise_kurtosis_excess=float(stats.kurtosis(noise)))
    return out


# ---------- (2) Targeted (block) decorrelation vs. blind lottery ----------
# Block model: K institutions split into 2 vendor clusters of size K/2.
# Institution k in cluster c: score = skill + sigma*(sqrt(rho)*cluster_noise[c] + sqrt(1-rho)*idio_k)
# Across-cluster pairs are therefore uncorrelated in shared noise; only
# within-cluster pairs (half of all K(K-1)/2 pairs, for 2 equal clusters)
# carry the rho-driven correlation. This lets us ask: if a platform can
# identify which institutions share a vendor, does decorrelating ONLY the
# institutions in the (fewer) high-correlation cluster(s) buy similar H
# reduction at lower accuracy cost than blind, uniformly-applied lottery?

def run_block_trial(seed, K=8, n_clusters=2, rho=0.5, p=0.2, dist="gaussian",
                     eta_by_cluster=None):
    rng = np.random.default_rng(seed)
    skill = rng.standard_normal(N)
    cluster_of = [i % n_clusters for i in range(K)]  # interleave assignment
    cluster_noise = {c: draw_noise(rng, N, dist) for c in range(n_clusters)}
    n_accept = max(1, int(round(p * N)))
    decisions = np.zeros((K, N), dtype=bool)
    eta_by_cluster = eta_by_cluster or {c: 0.0 for c in range(n_clusters)}
    for k in range(K):
        c = cluster_of[k]
        idio = draw_noise(rng, N, dist)
        score = skill + (np.sqrt(rho) * cluster_noise[c] + np.sqrt(1 - rho) * idio)
        idx = np.argsort(-score)[:n_accept]
        accept = np.zeros(N, dtype=bool)
        accept[idx] = True
        eta = eta_by_cluster[c]
        if eta > 0:
            lottery_mask = rng.random(N) < eta
            lottery_draw = rng.random(N) < p
            accept = np.where(lottery_mask, lottery_draw, accept)
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


def block_sweep(eta_by_cluster, rho=0.8, K=8, p=0.2, dist="gaussian", n_seeds=30, tag=""):
    Hs, Accs = [], []
    for s in range(n_seeds):
        seed = hash_seed("block", tag, rho, dist, s, tuple(sorted(eta_by_cluster.items())))
        skill, decisions = run_block_trial(seed, K=K, rho=rho, p=p, dist=dist,
                                            eta_by_cluster=eta_by_cluster)
        Hs.append(homogenization_corr(decisions))
        Accs.append(accuracy_recall_at_p(skill, decisions, p))
    return float(np.mean(Hs)), float(np.mean(Accs))


def targeted_vs_blind(rho=0.8, K=8, p=0.2, dist="gaussian", n_seeds=30):
    # Baseline block H with no mitigation
    H0, Acc0 = block_sweep({0: 0.0, 1: 0.0}, rho, K, p, dist, n_seeds, tag="base")
    rows = []
    # Blind: same eta applied to BOTH clusters (all K institutions)
    for eta in [0.0, 0.1, 0.2, 0.3, 0.4]:
        H, Acc = block_sweep({0: eta, 1: eta}, rho, K, p, dist, n_seeds, tag="blind")
        rows.append(dict(mode="blind", eta_cluster0=eta, eta_cluster1=eta, H=H, Acc=Acc))
    # Targeted: eta applied ONLY to cluster 0 (half the institutions)
    for eta in [0.0, 0.2, 0.4, 0.6, 0.8]:
        H, Acc = block_sweep({0: eta, 1: 0.0}, rho, K, p, dist, n_seeds, tag="targeted")
        rows.append(dict(mode="targeted", eta_cluster0=eta, eta_cluster1=0.0, H=H, Acc=Acc))
    return dict(H0=H0, Acc0=Acc0, rows=rows)


# ---------- (3) Sensitivity of mitigation-cost figure to rho ----------
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


def mitigation_full_decorr_point(rho, K=8, p=0.2, dist="gaussian", n_seeds=30, target_H=0.2955):
    etas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Hs, Accs = [], []
    for eta in etas:
        Hl, Al = [], []
        for s in range(n_seeds):
            seed = hash_seed("sens", rho, dist, eta, s)
            skill, decisions = run_trial_flat(seed, K, rho, p, dist, eta)
            Hl.append(homogenization_corr(decisions))
            Al.append(accuracy_recall_at_p(skill, decisions, p))
        Hs.append(float(np.mean(Hl)))
        Accs.append(float(np.mean(Al)))
    Hs, Accs, etas = np.array(Hs), np.array(Accs), np.array(etas)
    # interpolate eta where H crosses target_H (H is decreasing in eta)
    if target_H >= Hs[0]:
        eta_star, acc_star = 0.0, Accs[0]
    elif target_H <= Hs[-1]:
        eta_star, acc_star = etas[-1], Accs[-1]
    else:
        idx = np.searchsorted(-Hs, -target_H) - 1
        idx = max(0, min(idx, len(Hs) - 2))
        h1, h2 = Hs[idx], Hs[idx + 1]
        e1, e2 = etas[idx], etas[idx + 1]
        a1, a2 = Accs[idx], Accs[idx + 1]
        frac = (target_H - h1) / (h2 - h1) if h2 != h1 else 0.0
        eta_star = e1 + frac * (e2 - e1)
        acc_star = a1 + frac * (a2 - a1)
    return dict(rho=rho, etas=etas.tolist(), Hs=Hs.tolist(), Accs=Accs.tolist(),
                eta_star=float(eta_star), acc_star=float(acc_star), acc_at_eta0=float(Accs[0]),
                rel_cost_pct=float(100 * (Accs[0] - acc_star) / Accs[0]))


def main():
    out = {}
    out["rank_corr_check"] = rank_corr_check()
    out["targeted_vs_blind"] = targeted_vs_blind()
    out["sensitivity_rho"] = [mitigation_full_decorr_point(rho) for rho in [0.5, 0.6, 0.8]]
    with open("extend_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:3000])


if __name__ == "__main__":
    main()
