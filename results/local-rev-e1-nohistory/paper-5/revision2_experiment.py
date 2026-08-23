"""
Round-2 revision experiment, addressing round2_review.json:
  (1) Multi-seed training-budget ablation (seeds 1,2 in addition to the
      existing seed-0 ablation_results.json) to check whether the
      late-training-only dissociation pattern survives reseeding.
  (2) At the 600-step budget, in addition to scalar temperature, fit a
      two-parameter affine recalibration (scale a + additive per-token bias
      vector b, i.e. Platt-style logits' = a*logits + b) by val NLL, and
      check whether this richer family can improve KL and ECE simultaneously
      (reviewer question 3).
Kept deliberately smaller than the round-1 revision run (fewer seeds/budgets)
to fit the compute budget; reuses the exact generative process/model/training
recipe as experiment.py / ablation.py.
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
V, K, T_LEN = 10, 3, 40
N_TRAIN, N_VAL, N_TEST = 4000, 800, 1200
BATCH = 64
ABLATION_BUDGETS = [300, 600, 1200]
AFFINE_BUDGET = 600


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=48, nhead=4, nlayers=2, max_len=T_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                            dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
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
    ece = 0.0; N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / N) * abs(conf[m].mean() - true_prob_of_top[m].mean())
    return float(ece)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mode_T = [rng.dirichlet(0.3 * np.ones(V), size=V) for _ in range(K)]
    mode_prior = np.ones(K) / K

    def gen_sequence(rng):
        m = rng.integers(K)
        x = [rng.integers(V)]
        for _ in range(T_LEN - 1):
            x.append(rng.choice(V, p=mode_T[m][x[-1]]))
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

    test_true = np.stack([true_predictive_probs(x) for x in test_x])
    train_t = torch.from_numpy(train_x)

    ablation_rows = []
    steps_done = 0
    torch.manual_seed(seed)
    model = TinyTransformerLM(V)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    def nll_for_T(Tval, X):
        xt = torch.from_numpy(X)
        inp = xt[:, :-1]; tgt = xt[:, 1:]
        with torch.no_grad():
            logp = F.log_softmax(model(inp) / Tval, dim=-1)
            return F.nll_loss(logp.reshape(-1, V), tgt.reshape(-1)).item()

    @torch.no_grad()
    def probs_at_T(X, Tval=1.0):
        model.eval()
        xt = torch.from_numpy(X)
        return F.softmax(model(xt[:, :-1]) / Tval, dim=-1).numpy()

    for target_steps in ABLATION_BUDGETS:
        while steps_done < target_steps:
            idx = rng.choice(N_TRAIN, BATCH, replace=False)
            b = train_t[idx]
            model.train()
            inp = b[:, :-1]; tgt = b[:, 1:]
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            steps_done += 1

        Ts = np.linspace(0.5, 2.5, 21)
        nlls = [nll_for_T(float(Tv), val_x) for Tv in Ts]
        best_T = float(Ts[int(np.argmin(nlls))])

        raw = probs_at_T(test_x, 1.0)
        scaled = probs_at_T(test_x, best_T)
        row = dict(seed=seed, steps=target_steps, best_T=best_T,
                   kl_raw=float(kl_div(raw, test_true).mean()),
                   kl_scaled=float(kl_div(scaled, test_true).mean()),
                   ece_raw=true_ece(raw, test_true), ece_scaled=true_ece(scaled, test_true))
        ablation_rows.append(row)
        print(f"[{time.time()-t_start:.1f}s] seed={seed} steps={target_steps} T*={best_T:.2f} "
              f"KL {row['kl_raw']:.4f}->{row['kl_scaled']:.4f} ECE {row['ece_raw']:.4f}->{row['ece_scaled']:.4f}")

        if target_steps == AFFINE_BUDGET:
            # Two-parameter affine recalibration: logits' = a*logits + b (b in R^V),
            # fit by gradient descent on val NLL (Platt-style), initialized at a=1,b=0.
            a = torch.tensor(1.0, requires_grad=True)
            b_vec = torch.zeros(V, requires_grad=True)
            aff_opt = torch.optim.Adam([a, b_vec], lr=0.05)
            xt_val = torch.from_numpy(val_x)
            inp_val = xt_val[:, :-1]; tgt_val = xt_val[:, 1:]
            with torch.no_grad():
                base_logits_val = model(inp_val)
            for _ in range(300):
                logp = F.log_softmax(a * base_logits_val + b_vec, dim=-1)
                l = F.nll_loss(logp.reshape(-1, V), tgt_val.reshape(-1))
                aff_opt.zero_grad(); l.backward(); aff_opt.step()
            a_f, b_f = float(a.detach()), b_vec.detach().numpy()

            with torch.no_grad():
                base_logits_test = model(torch.from_numpy(test_x)[:, :-1])
                aff_probs = F.softmax(a_f * base_logits_test + torch.from_numpy(b_f), dim=-1).numpy()
            kl_affine = float(kl_div(aff_probs, test_true).mean())
            ece_affine = true_ece(aff_probs, test_true)
            row["affine_a"] = a_f
            row["affine_b_norm"] = float(np.linalg.norm(b_f))
            row["kl_affine"] = kl_affine
            row["ece_affine"] = ece_affine
            print(f"[{time.time()-t_start:.1f}s] seed={seed} AFFINE a={a_f:.3f} |b|={np.linalg.norm(b_f):.3f} "
                  f"KL {row['kl_raw']:.4f}->{kl_affine:.4f} ECE {row['ece_raw']:.4f}->{ece_affine:.4f}")

    return ablation_rows


if __name__ == "__main__":
    all_rows = []
    for s in [1, 2]:
        all_rows.extend(run_seed(s))
    with open("revision2_results.json", "w") as f:
        json.dump(dict(rows=all_rows, elapsed_sec=time.time() - t_start), f, indent=2)
    print(f"[{time.time()-t_start:.1f}s] DONE")
