"""
Revision experiment addressing round-1 review:
  (1) Multi-seed replication of the 600-step main run (5 seeds) to get
      variance on the KL/ECE gap, instead of a single seed.
  (2) For each seed, in addition to the NLL-minimizing T*, also grid-search
      the KL-minimizing T (argmin KL(p_true || p_model_scaled)) to test
      whether the dissociation is inherent to scalar rescaling itself or
      specific to fitting T by NLL against realized outcomes.
  (3) Sensitivity of the oracle-ECE improvement factor to n_bins.

Reuses the exact same generative process / model / training recipe as
experiment.py, just wrapped in a function parameterized by seed, and run
for 5 seeds instead of 1.
"""
import time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()

V = 10
K = 3
T_LEN = 40
N_TRAIN, N_VAL, N_CAL, N_TEST = 4000, 800, 800, 1200
N_STEPS = 600
BATCH = 64

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=48, nhead=4, nlayers=2, max_len=T_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                            dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)


def kl_div(p, q, eps=1e-9):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def true_ece(model_probs, true_probs, n_bins=10):
    conf = model_probs.max(axis=-1)
    top_tok = model_probs.argmax(axis=-1)
    true_prob_of_top = np.take_along_axis(true_probs, top_tok[..., None], axis=-1).squeeze(-1)
    conf = conf.flatten(); true_prob_of_top = true_prob_of_top.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / N) * abs(conf[m].mean() - true_prob_of_top[m].mean())
    return ece


def run_seed(seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    alphas = [0.3, 0.3, 0.3]
    mode_T = [rng.dirichlet(a * np.ones(V), size=V) for a in alphas]
    mode_prior = np.ones(K) / K

    def gen_sequence(rng):
        m = rng.integers(K)
        x = [rng.integers(V)]
        for _ in range(T_LEN - 1):
            p = mode_T[m][x[-1]]
            x.append(rng.choice(V, p=p))
        return np.array(x, dtype=np.int64)

    def gen_dataset(n, rng):
        return np.stack([gen_sequence(rng) for _ in range(n)])

    train_x = gen_dataset(N_TRAIN, rng)
    val_x = gen_dataset(N_VAL, rng)
    test_x = gen_dataset(N_TEST, rng)

    def true_predictive_probs(x):
        Tt = len(x)
        post = mode_prior.copy()
        out = np.zeros((Tt - 1, V))
        for t in range(Tt - 1):
            pred = np.zeros(V)
            for m in range(K):
                pred += post[m] * mode_T[m][x[t]]
            out[t] = pred
            lik = np.array([mode_T[m][x[t], x[t + 1]] for m in range(K)])
            post = post * lik
            post = post / post.sum()
        return out

    def batch_true_predictive(X):
        return np.stack([true_predictive_probs(x) for x in X])

    device = "cpu"
    model = TinyTransformerLM(V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    train_t = torch.from_numpy(train_x).to(device)

    for step in range(N_STEPS):
        idx = rng.choice(N_TRAIN, BATCH, replace=False)
        b = train_t[idx]
        model.train()
        inp = b[:, :-1]; tgt = b[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def model_predictive_probs(X, T_scale=1.0):
        model.eval()
        xt = torch.from_numpy(X).to(device)
        inp = xt[:, :-1]
        logits = model(inp) / T_scale
        return F.softmax(logits, dim=-1).numpy()

    def nll_for_T(Tval, X):
        xt = torch.from_numpy(X)
        inp = xt[:, :-1]; tgt = xt[:, 1:]
        with torch.no_grad():
            logits = model(inp) / Tval
            logp = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(logp.reshape(-1, V), tgt.reshape(-1))
        return nll.item()

    Ts = np.linspace(0.5, 2.5, 41)
    nlls = [nll_for_T(float(Tv), val_x) for Tv in Ts]
    best_T_nll = float(Ts[int(np.argmin(nlls))])

    test_true = batch_true_predictive(test_x)

    # KL-optimal T: grid search T that minimizes mean KL(p_true || p_model_scaled) on TEST
    # (oracle grid search — only possible because we know p_true; used purely to
    # characterize the best any scalar T could do, not as a fittable recipe)
    kl_by_T = []
    for Tv in Ts:
        probs = model_predictive_probs(test_x, float(Tv))
        kl_by_T.append(kl_div(probs, test_true).mean())
    best_T_kl = float(Ts[int(np.argmin(kl_by_T))])
    min_kl_at_best = float(min(kl_by_T))

    test_model_raw = model_predictive_probs(test_x, 1.0)
    test_model_scaled_nll = model_predictive_probs(test_x, best_T_nll)
    test_model_scaled_kl = model_predictive_probs(test_x, best_T_kl)

    kl_raw = float(kl_div(test_model_raw, test_true).mean())
    kl_scaled_nll = float(kl_div(test_model_scaled_nll, test_true).mean())
    kl_scaled_kl = float(kl_div(test_model_scaled_kl, test_true).mean())

    ece_raw = true_ece(test_model_raw, test_true, n_bins=10)
    ece_scaled_nll = true_ece(test_model_scaled_nll, test_true, n_bins=10)
    ece_scaled_kl = true_ece(test_model_scaled_kl, test_true, n_bins=10)

    # bin-count sensitivity for the NLL-scaled improvement factor
    bin_sensitivity = {}
    for nb in [5, 10, 15, 20]:
        er = true_ece(test_model_raw, test_true, n_bins=nb)
        es = true_ece(test_model_scaled_nll, test_true, n_bins=nb)
        bin_sensitivity[nb] = dict(ece_raw=er, ece_scaled=es, ratio=(er / es if es > 0 else None))

    return dict(
        seed=seed,
        best_T_nll=best_T_nll,
        best_T_kl=best_T_kl,
        min_kl_at_best_T=min_kl_at_best,
        kl_raw=kl_raw, kl_scaled_nll=kl_scaled_nll, kl_scaled_kl=kl_scaled_kl,
        ece_raw=float(ece_raw), ece_scaled_nll=float(ece_scaled_nll), ece_scaled_kl=float(ece_scaled_kl),
        bin_sensitivity=bin_sensitivity,
    )


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    all_results = []
    for s in seeds:
        r = run_seed(s)
        all_results.append(r)
        print(f"[{time.time()-t_start:.1f}s] seed={s} T*_nll={r['best_T_nll']:.2f} T*_kl={r['best_T_kl']:.2f} "
              f"KL raw={r['kl_raw']:.4f} scaled_nll={r['kl_scaled_nll']:.4f} scaled_kl={r['kl_scaled_kl']:.4f} "
              f"ECE raw={r['ece_raw']:.4f} scaled_nll={r['ece_scaled_nll']:.4f} scaled_kl={r['ece_scaled_kl']:.4f}")

    def agg(key):
        vals = [r[key] for r in all_results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {k: agg(k) for k in
               ["best_T_nll", "best_T_kl", "kl_raw", "kl_scaled_nll", "kl_scaled_kl",
                "ece_raw", "ece_scaled_nll", "ece_scaled_kl"]}

    out = dict(seeds=seeds, per_seed=all_results, summary_mean_std=summary,
               elapsed_sec=time.time() - t_start)
    with open("revision_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{time.time()-t_start:.1f}s] DONE.")
    print(json.dumps(summary, indent=2))
