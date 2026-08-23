import torch, torch.nn as nn, numpy as np, json, math, time

V = 10
K = 3
T_LEN = 40
D_MODEL = 48

def make_process(seed):
    rng = np.random.default_rng(seed)
    Ts = []
    for m in range(K):
        rows = rng.dirichlet(alpha=[0.3]*V, size=V)
        Ts.append(rows)
    return np.stack(Ts)  # K,V,V

def gen_data(Ts, n, seed):
    rng = np.random.default_rng(seed)
    K_ = Ts.shape[0]
    seqs = np.zeros((n, T_LEN), dtype=np.int64)
    modes = rng.integers(0, K_, size=n)
    seqs[:, 0] = rng.integers(0, V, size=n)
    for t in range(T_LEN - 1):
        for i in range(n):
            probs = Ts[modes[i], seqs[i, t]]
            seqs[i, t+1] = rng.choice(V, p=probs)
    return seqs

def true_predictive(Ts, seqs):
    # returns array (n, T_LEN-1, V) of p_true(x_{t+1}|x_1:t) for each position
    n = seqs.shape[0]
    K_ = Ts.shape[0]
    post = np.ones((n, K_)) / K_
    out = np.zeros((n, T_LEN - 1, V))
    for t in range(T_LEN - 1):
        xt = seqs[:, t]
        pred = np.einsum('nk,nkv->nv', post, Ts[:, xt, :].transpose(1, 0, 2))
        out[:, t, :] = pred
        xt1 = seqs[:, t+1]
        lik = Ts[:, xt, xt1].T  # n,K
        post = post * lik
        post = post / post.sum(axis=1, keepdims=True)
    return out

class TinyTransformer(nn.Module):
    def __init__(self, vocab=V, d=D_MODEL, nhead=4, nlayers=2, maxlen=T_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=4*d, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d, vocab)
        self.maxlen = maxlen

    def forward(self, x):
        n, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)  # n,t,vocab

def train_model(seqs, steps, seed, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    model = TinyTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x = torch.tensor(seqs)
    n = x.shape[0]
    rng = np.random.default_rng(seed + 1000)
    losses = []
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        batch_x = x[idx]
        inp = batch_x[:, :-1]
        tgt = batch_x[:, 1:]
        logits = model(inp)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, np.mean(losses[-20:])

def model_probs(model, seqs):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seqs)
        inp = x[:, :-1]
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1).numpy()
    return probs  # n, T-1, V

def model_probs_temp(model, seqs, temp):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seqs)
        inp = x[:, :-1]
        logits = model(inp)
        probs = torch.softmax(logits / temp, dim=-1).numpy()
    return probs

def kl_div(p_true, p_model, eps=1e-9):
    p_true = np.clip(p_true, eps, 1)
    p_model = np.clip(p_model, eps, 1)
    return np.sum(p_true * (np.log(p_true) - np.log(p_model)), axis=-1)

def nll_of(probs, targets, eps=1e-9):
    p = np.clip(probs, eps, 1)
    n, t, v = p.shape
    idx = targets.reshape(-1)
    flat = p.reshape(-1, v)
    ll = -np.log(flat[np.arange(len(idx)), idx])
    return ll.mean()

def oracle_ece(probs, p_true_probs, n_bins=10):
    n, t, v = probs.shape
    top1_idx = probs.argmax(axis=-1).reshape(-1)
    top1_conf = probs.max(axis=-1).reshape(-1)
    flat_true = p_true_probs.reshape(-1, v)
    true_prob_of_top1 = flat_true[np.arange(len(top1_idx)), top1_idx]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(top1_conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i == n_bins - 1:
            mask = (top1_conf >= lo) & (top1_conf <= hi)
        else:
            mask = (top1_conf >= lo) & (top1_conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = top1_conf[mask].mean()
        avg_true = true_prob_of_top1[mask].mean()
        ece += (mask.sum() / N) * abs(avg_conf - avg_true)
    return ece

def fit_temp_nll(model, val_seqs, grid):
    x = torch.tensor(val_seqs)
    tgt = x[:, 1:]
    best_t, best_nll = 1.0, None
    for temp in grid:
        probs = model_probs_temp(model, val_seqs, temp)
        n = nll_of(probs, tgt.numpy())
        if best_nll is None or n < best_nll:
            best_nll, best_t = n, temp
    return best_t, best_nll

def fit_temp_kl(model, val_seqs, p_true_val, grid):
    best_t, best_kl = 1.0, None
    for temp in grid:
        probs = model_probs_temp(model, val_seqs, temp)
        k = kl_div(p_true_val, probs).mean()
        if best_kl is None or k < best_kl:
            best_kl, best_t = k, temp
    return best_t, best_kl

def conformal_eval(model, cal_seqs, test_seqs, temp, target_cov=0.9):
    cal_probs = model_probs_temp(model, cal_seqs, temp)
    cal_x = cal_seqs[:, 1:]
    n, t, v = cal_probs.shape
    flat_p = cal_probs.reshape(-1, v)
    flat_tgt = cal_x.reshape(-1)
    scores = 1 - flat_p[np.arange(len(flat_tgt)), flat_tgt]
    n_cal = len(scores)
    q_level = min(1.0, math.ceil((n_cal + 1) * target_cov) / n_cal)
    qhat = np.quantile(scores, q_level)

    test_probs = model_probs_temp(model, test_seqs, temp)
    test_tgt = test_seqs[:, 1:]
    nt, tt, vt = test_probs.shape
    flat_pt = test_probs.reshape(-1, vt)
    flat_tt = test_tgt.reshape(-1)
    set_mask = (1 - flat_pt) <= qhat
    covered = set_mask[np.arange(len(flat_tt)), flat_tt]
    coverage = covered.mean()
    set_size = set_mask.sum(axis=1).mean()
    return coverage, set_size

def run_seed(seed, steps=600, n_train=4000, n_val=800, n_cal=800, n_test=1200, temp_grid=None):
    if temp_grid is None:
        temp_grid = np.linspace(0.5, 2.5, 41)
    Ts = make_process(seed)
    train_seqs = gen_data(Ts, n_train, seed*10+1)
    val_seqs = gen_data(Ts, n_val, seed*10+2)
    cal_seqs = gen_data(Ts, n_cal, seed*10+3)
    test_seqs = gen_data(Ts, n_test, seed*10+4)

    model, train_loss = train_model(train_seqs, steps, seed)

    p_true_test = true_predictive(Ts, test_seqs)
    p_true_val = true_predictive(Ts, val_seqs)

    raw_probs_test = model_probs_temp(model, test_seqs, 1.0)
    kl_raw = kl_div(p_true_test, raw_probs_test)
    kl_raw_early = kl_raw[:, :9].mean()
    kl_raw_late = kl_raw[:, 9:].mean()
    kl_raw_all = kl_raw.mean()

    t_star_nll, val_nll_best = fit_temp_nll(model, val_seqs, temp_grid)
    val_nll_1, _ = fit_temp_nll(model, val_seqs, [1.0])

    scaled_probs_test = model_probs_temp(model, test_seqs, t_star_nll)
    kl_scaled = kl_div(p_true_test, scaled_probs_test)
    kl_scaled_early = kl_scaled[:, :9].mean()
    kl_scaled_late = kl_scaled[:, 9:].mean()
    kl_scaled_all = kl_scaled.mean()

    ece_raw = oracle_ece(raw_probs_test, p_true_test)
    ece_scaled = oracle_ece(scaled_probs_test, p_true_test)

    t_star_kl, kl_best = fit_temp_kl(model, val_seqs, p_true_val, temp_grid)

    cov_raw, size_raw = conformal_eval(model, cal_seqs, test_seqs, 1.0)
    cov_scaled, size_scaled = conformal_eval(model, cal_seqs, test_seqs, t_star_nll)

    ece_bins = {}
    for nb in [5, 10, 15, 20]:
        er = oracle_ece(raw_probs_test, p_true_test, n_bins=nb)
        es = oracle_ece(scaled_probs_test, p_true_test, n_bins=nb)
        ece_bins[nb] = (er, es, er / es if es > 0 else None)

    return dict(
        seed=seed, steps=steps, train_loss=train_loss,
        t_star_nll=t_star_nll, t_star_kl=t_star_kl,
        kl_raw_early=kl_raw_early, kl_raw_late=kl_raw_late, kl_raw_all=kl_raw_all,
        kl_scaled_early=kl_scaled_early, kl_scaled_late=kl_scaled_late, kl_scaled_all=kl_scaled_all,
        ece_raw=ece_raw, ece_scaled=ece_scaled,
        cov_raw=cov_raw, size_raw=size_raw, cov_scaled=cov_scaled, size_scaled=size_scaled,
        ece_bins=ece_bins,
    )

def ablation_run(steps, seed=0):
    r = run_seed(seed, steps=steps)
    return r

if __name__ == "__main__":
    import sys, os
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    outfile = "results_v2.json"
    results = {}
    if os.path.exists(outfile):
        with open(outfile) as f:
            results = json.load(f)

    if mode in ("all", "multiseed"):
        print("=== Multi-seed main run (600 steps) ===")
        multi = []
        for seed in [0, 1, 2]:
            r = run_seed(seed, steps=600)
            multi.append(r)
            print(seed, "t*_nll", r['t_star_nll'], "t*_kl", r['t_star_kl'],
                  "kl_raw", round(r['kl_raw_all'],4), "kl_scaled", round(r['kl_scaled_all'],4),
                  "ece_raw", round(r['ece_raw'],4), "ece_scaled", round(r['ece_scaled'],4),
                  "cov_raw", round(r['cov_raw'],4), "cov_scaled", round(r['cov_scaled'],4))
            print("elapsed", time.time()-t0)
        results['multiseed_600'] = multi
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2, default=float)

    if mode in ("all", "multiseed1200"):
        print("=== Multi-seed replication at 1200 steps (dissociation-hint check) ===")
        multi1200 = []
        for seed in [0, 1, 2]:
            r = run_seed(seed, steps=1200)
            multi1200.append(r)
            print(seed, "t*_nll", r['t_star_nll'], "t*_kl", r['t_star_kl'],
                  "kl_raw", round(r['kl_raw_all'],4), "kl_scaled", round(r['kl_scaled_all'],4),
                  "ece_raw", round(r['ece_raw'],4), "ece_scaled", round(r['ece_scaled'],4))
            print("elapsed", time.time()-t0)
        results['multiseed_1200'] = multi1200
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2, default=float)

    if mode in ("all", "ablation"):
        print("=== Ablation over training budget (seed=0) ===")
        ablation = []
        for steps in [50, 150, 300, 600, 1200]:
            r = ablation_run(steps, seed=0)
            ablation.append(r)
            print(steps, "t*_nll", r['t_star_nll'], "kl_raw", round(r['kl_raw_all'],4),
                  "kl_scaled", round(r['kl_scaled_all'],4), "ece_raw", round(r['ece_raw'],4),
                  "ece_scaled", round(r['ece_scaled'],4))
            print("elapsed", time.time()-t0)
        results['ablation'] = ablation
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2, default=float)

    print("TOTAL elapsed", time.time()-t0)
