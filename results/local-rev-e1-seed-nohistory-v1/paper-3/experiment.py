"""Controlled synthetic key->value UQ study. Trains a tiny transformer from
scratch per seed on a fully synthetic key->value task with known aleatoric
distributions and controlled exposure, then measures calibration, sampling
cost, error detection, and two diagnostics for the unseen-key entropy match.
"""
import json, time, math, random
import numpy as np
import torch
import torch.nn as nn

N_KEYS = 60
N_VALUES = 8
TIERS = {"unseen": 0, "rare": 3, "medium": 20, "frequent": 150}
STOCH_PROBS = [0.6, 0.3, 0.1]
D_MODEL = 64
N_HEADS = 4
FF = 128
N_LAYERS = 2
EPOCHS = 25
LR = 3e-3
BATCH = 128
SEEDS = list(range(8))

# vocab: 0=PAD,1=BOS,2=SEP,3=EOS, 4..63=keys(60), 64..71=values(8)
KEY0, VAL0 = 4, 4 + N_KEYS
VOCAB = VAL0 + N_VALUES


class TinyTransformer(nn.Module):
    def __init__(self, seq_len=4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(seq_len, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, FF, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)


def build_task(seed, n_cand=3, det_mass=1.0):
    rng = random.Random(seed)
    key_types = (["det"] * (N_KEYS // 2) + ["stoch"] * (N_KEYS // 2))
    rng.shuffle(key_types)
    tiers = list(TIERS.keys()) * (N_KEYS // len(TIERS))
    rng.shuffle(tiers)
    keys = {}
    for i in range(N_KEYS):
        kt, tier = key_types[i], tiers[i]
        if kt == "det":
            v = rng.randrange(N_VALUES)
            dist = [0.0] * N_VALUES
            if det_mass >= 1.0:
                dist[v] = 1.0
            else:
                others = rng.sample([j for j in range(N_VALUES) if j != v], 2)
                leftover = (1.0 - det_mass) / 2
                dist[v] = det_mass
                dist[others[0]] = leftover
                dist[others[1]] = leftover
        else:
            cand = rng.sample(range(N_VALUES), n_cand)
            probs = STOCH_PROBS[:n_cand]
            probs = [p / sum(probs) for p in probs]
            rng.shuffle(probs)
            dist = [0.0] * N_VALUES
            for c, p in zip(cand, probs):
                dist[c] = p
        keys[i] = {"type": kt, "tier": tier, "dist": dist}
    return keys


def make_training_data(keys, seed):
    rng = random.Random(seed + 10000)
    seqs = []
    for k, info in keys.items():
        n = TIERS[info["tier"]]
        dist = info["dist"]
        for _ in range(n):
            v = rng.choices(range(N_VALUES), weights=dist, k=1)[0]
            seqs.append([1, KEY0 + k, 2, VAL0 + v])  # BOS key SEP value (EOS via next-token loss)
    rng.shuffle(seqs)
    return seqs


def train_model(seqs, seed):
    torch.manual_seed(seed)
    model = TinyTransformer(seq_len=4)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    X = torch.tensor(seqs, dtype=torch.long)
    inp, tgt = X[:, :-1], X[:, 1:]
    losses = []
    for ep in range(EPOCHS):
        perm = torch.randperm(len(inp))
        tot, n = 0.0, 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            logits = model(inp[idx])
            loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgt[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        losses.append(tot / n)
    return model, losses


def key_softmax(model, k):
    x = torch.tensor([[1, KEY0 + k, 2]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)[0, -1, VAL0:VAL0 + N_VALUES]
        p = torch.softmax(logits, dim=-1).numpy()
    return p


def kl(p_true, p_model, eps=1e-9):
    p_true = np.array(p_true); p_model = np.array(p_model)
    mask = p_true > 0
    return float(np.sum(p_true[mask] * np.log((p_true[mask] + eps) / (p_model[mask] + eps))))


def tv(p_true, p_model):
    return float(0.5 * np.sum(np.abs(np.array(p_true) - np.array(p_model))))


def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def run_seed(seed, n_cand=3, det_mass=1.0):
    keys = build_task(seed, n_cand=n_cand, det_mass=det_mass)
    seqs = make_training_data(keys, seed)
    t0 = time.time()
    model, losses = train_model(seqs, seed)
    train_time = time.time() - t0

    # random-init model for the "moderate-entropy prior" mechanism check
    torch.manual_seed(seed + 99999)
    rand_model = TinyTransformer(seq_len=4)

    per_key = {}
    for k, info in keys.items():
        p_model = key_softmax(model, k)
        p_rand = key_softmax(rand_model, k)
        p_true = info["dist"]
        per_key[k] = {
            "type": info["type"], "tier": info["tier"],
            "kl": kl(p_true, p_model), "tv": tv(p_true, p_model),
            "h_model": entropy(p_model), "h_true": entropy(p_true),
            "h_rand_init": entropy(p_rand),
            "argmax_correct": int(np.argmax(p_model) == np.argmax(p_true)) if info["type"] == "det" else None,
            "top_true_mass": float(sum(p_model[i] for i, pt in enumerate(p_true) if pt > 0)),
        }
        if info["type"] == "det":
            true_v = int(np.argmax(p_true))
            per_key[k]["argmax_correct"] = int(np.argmax(p_model) == true_v)

    # Experiment B: MC sampling cost, medium/frequent keys
    mc_rows = []
    for k, info in keys.items():
        if info["tier"] not in ("medium", "frequent"):
            continue
        p_model = key_softmax(model, k)
        h_true_analytic = entropy(p_model)
        for T in [5, 20, 100, 400]:
            errs = []
            mc_rng = np.random.RandomState(seed * 1000003 + k * 1009 + T)
            for _ in range(30):
                samp = mc_rng.choice(N_VALUES, size=T, p=p_model)
                counts = np.bincount(samp, minlength=N_VALUES) / T
                errs.append(entropy(counts) - h_true_analytic)
            mc_rows.append({"T": T, "errs": errs})

    # Experiment C: AUROC entropy / 1-maxprob for error detection on det keys
    det_keys = [k for k, v in per_key.items() if v["type"] == "det"]
    y_err = np.array([1 - per_key[k]["argmax_correct"] for k in det_keys])
    h_scores = np.array([per_key[k]["h_model"] for k in det_keys])
    mp_scores = np.array([1 - max(key_softmax(model, k)) for k in det_keys])

    def auroc(scores, labels):
        if len(set(labels.tolist())) < 2:
            return None
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = labels.sum(); n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        sum_ranks_pos = ranks[labels == 1].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    auroc_h = auroc(h_scores, y_err)
    auroc_mp = auroc(mp_scores, y_err)

    # unseen-stochastic diagnostic (Experiment D)
    unseen_stoch = [k for k, v in per_key.items() if v["type"] == "stoch" and v["tier"] == "unseen"]
    mass = [per_key[k]["top_true_mass"] for k in unseen_stoch]
    h_unseen_stoch = [per_key[k]["h_model"] for k in unseen_stoch]
    acc_unseen_stoch = [int(np.argmax(key_softmax(model, k)) == np.argmax(keys[k]["dist"])) for k in unseen_stoch]
    rand_init_h_unseen = [per_key[k]["h_rand_init"] for k in unseen_stoch]

    global_marginal = np.zeros(N_VALUES)
    for s in seqs:
        global_marginal[s[3] - VAL0] += 1
    global_marginal = global_marginal / global_marginal.sum()
    h_global_marginal = entropy(global_marginal)

    return {
        "seed": seed, "n_cand": n_cand, "train_time_s": train_time,
        "final_loss": losses[-1], "per_key": per_key,
        "mc_rows": mc_rows, "auroc_entropy": auroc_h, "auroc_1minusmax": auroc_mp,
        "unseen_stoch_mass_mean": float(np.mean(mass)) if mass else None,
        "unseen_stoch_h_model_mean": float(np.mean(h_unseen_stoch)) if h_unseen_stoch else None,
        "unseen_stoch_argmax_acc": float(np.mean(acc_unseen_stoch)) if acc_unseen_stoch else None,
        "unseen_stoch_h_rand_init_mean": float(np.mean(rand_init_h_unseen)) if rand_init_h_unseen else None,
        "h_global_marginal": h_global_marginal,
    }


def aggregate(results, key_fn):
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k, v in r["per_key"].items():
            g = (v["type"], v["tier"])
            groups[g]["kl"].append(v["kl"])
            groups[g]["tv"].append(v["tv"])
            groups[g]["h_model"].append(v["h_model"])
            if v["argmax_correct"] is not None:
                groups[g]["acc"].append(v["argmax_correct"])
    out = {}
    for g, d in groups.items():
        out["/".join(g)] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in d.items()}
    return out


def main():
    t_start = time.time()
    print("=== Experiment A/B/C/D: n_cand=3 (original design), 8 seeds ===")
    results = []
    for s in SEEDS:
        r = run_seed(s, n_cand=3)
        results.append(r)
        print(f"seed {s}: train_time={r['train_time_s']:.1f}s final_loss={r['final_loss']:.3f} "
              f"auroc_h={r['auroc_entropy']:.3f} unseen_stoch_mass={r['unseen_stoch_mass_mean']:.3f} "
              f"rand_init_h={r['unseen_stoch_h_rand_init_mean']:.3f}")

    agg = aggregate(results, None)
    print("\nPer (type,tier) aggregates (mean, std) over 8 seeds:")
    for g in ["det/unseen", "det/rare", "det/medium", "det/frequent",
              "stoch/unseen", "stoch/rare", "stoch/medium", "stoch/frequent"]:
        gt = g.replace("det", "det").replace("stoch", "stoch")
    for g, d in agg.items():
        print(g, d)

    aurocs_h = [r["auroc_entropy"] for r in results]
    aurocs_mp = [r["auroc_1minusmax"] for r in results]
    print(f"\nAUROC entropy: {np.mean(aurocs_h):.3f} +/- {np.std(aurocs_h):.3f}")
    print(f"AUROC 1-maxprob: {np.mean(aurocs_mp):.3f} +/- {np.std(aurocs_mp):.3f}")

    masses = [r["unseen_stoch_mass_mean"] for r in results]
    accs = [r["unseen_stoch_argmax_acc"] for r in results]
    rand_hs = [r["unseen_stoch_h_rand_init_mean"] for r in results]
    trained_hs = [r["unseen_stoch_h_model_mean"] for r in results]
    global_hs = [r["h_global_marginal"] for r in results]
    print(f"\nUnseen-stoch true-candidate mass: {np.mean(masses):.3f} +/- {np.std(masses):.3f}")
    print(f"Unseen-stoch argmax acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"Unseen-stoch H_model (trained): {np.mean(trained_hs):.3f} +/- {np.std(trained_hs):.3f}")
    print(f"Unseen-stoch H at random init: {np.mean(rand_hs):.3f} +/- {np.std(rand_hs):.3f}")
    print(f"Global marginal entropy: {np.mean(global_hs):.3f} +/- {np.std(global_hs):.3f}")
    corr_global = np.corrcoef(trained_hs, global_hs)[0, 1]
    corr_randinit = np.corrcoef(trained_hs, rand_hs)[0, 1]
    print(f"corr(unseen_stoch H_model, global marginal H) = {corr_global:.3f}")
    print(f"corr(unseen_stoch H_model, random-init H) = {corr_randinit:.3f}")

    # MC sampling cost (representative seed 0)
    mc = results[0]["mc_rows"]
    from collections import defaultdict
    by_T = defaultdict(list)
    for row in mc:
        by_T[row["T"]].extend(row["errs"])
    print("\nMC sampling cost (seed 0):")
    mc_summary = {}
    for T in [5, 20, 100, 400]:
        errs = np.array(by_T[T])
        mae = float(np.mean(np.abs(errs))); rmse = float(np.sqrt(np.mean(errs ** 2)))
        mc_summary[T] = {"mae": mae, "rmse": rmse}
        print(f"T={T}: MAE={mae:.3f} RMSE={rmse:.3f}")

    with open("multi_seed_results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "mc_rows"} for r in results], f, indent=2)

    summary = {
        "aggregate_by_type_tier": agg,
        "auroc_entropy_mean_std": [float(np.mean(aurocs_h)), float(np.std(aurocs_h))],
        "auroc_1minusmax_mean_std": [float(np.mean(aurocs_mp)), float(np.std(aurocs_mp))],
        "unseen_stoch_mass_mean_std": [float(np.mean(masses)), float(np.std(masses))],
        "unseen_stoch_argmax_acc_mean_std": [float(np.mean(accs)), float(np.std(accs))],
        "unseen_stoch_h_trained_mean_std": [float(np.mean(trained_hs)), float(np.std(trained_hs))],
        "unseen_stoch_h_randinit_mean_std": [float(np.mean(rand_hs)), float(np.std(rand_hs))],
        "global_marginal_h_mean_std": [float(np.mean(global_hs)), float(np.std(global_hs))],
        "corr_trained_vs_global_marginal": float(corr_global),
        "corr_trained_vs_random_init": float(corr_randinit),
        "mc_sampling_cost_seed0": mc_summary,
        "total_time_s": time.time() - t_start,
    }

    # Harder task: n_cand=5 stochastic candidates, seeds 0-2 only (compute budget)
    print("\n=== Harder task: n_cand=5 stochastic candidates, 3 seeds ===")
    hard_results = []
    for s in range(3):
        r = run_seed(s, n_cand=5)
        hard_results.append(r)
        print(f"seed {s}: auroc_h={r['auroc_entropy']:.3f}")
    hard_agg = aggregate(hard_results, None)
    hard_aurocs = [r["auroc_entropy"] for r in hard_results]
    print(f"Hard-task AUROC entropy: {np.mean(hard_aurocs):.3f} +/- {np.std(hard_aurocs):.3f}")
    for g, d in hard_agg.items():
        print(g, d)

    summary["hard_task_n_cand5"] = {
        "aggregate_by_type_tier": hard_agg,
        "auroc_entropy_mean_std": [float(np.mean(hard_aurocs)), float(np.std(hard_aurocs))],
        "n_seeds": 3,
    }

    # Experiment F: near-deterministic keys (mass 0.9/0.05/0.05) to test whether
    # perturbing deterministic-key hardness itself (rather than stochastic-key
    # hardness) makes the AUROC/accuracy step function graded.
    print("\n=== Near-deterministic task: det_mass=0.9, 3 seeds ===")
    neardet_results = []
    for s in range(3):
        r = run_seed(s, n_cand=3, det_mass=0.9)
        neardet_results.append(r)
        print(f"seed {s}: auroc_h={r['auroc_entropy']:.3f}")
    neardet_agg = aggregate(neardet_results, None)
    neardet_aurocs = [r["auroc_entropy"] for r in neardet_results]
    print(f"Near-det AUROC entropy: {np.mean(neardet_aurocs):.3f} +/- {np.std(neardet_aurocs):.3f}")
    for g, d in neardet_agg.items():
        print(g, d)

    summary["near_det_mass0.9"] = {
        "aggregate_by_type_tier": neardet_agg,
        "auroc_entropy_mean_std": [float(np.mean(neardet_aurocs)), float(np.std(neardet_aurocs))],
        "n_seeds": 3,
    }

    with open("multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
