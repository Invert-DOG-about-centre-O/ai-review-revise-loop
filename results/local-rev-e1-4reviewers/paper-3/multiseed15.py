"""
Extended multi-seed replication (round-2 revision): 15 independent seeds
(instead of 5), plus per-seed ECE for logp and self-consistency, a seed-level
paired bootstrap CI on the mean AUROC gap (magnitude, not just sign test),
and a check of whether the logp-vs-sampling AUROC gap correlates with
per-seed model accuracy.
"""
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

VOCAB = list("0123456789+=. ") + ["<pad>", "<eos>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
PAD, EOS = STOI["<pad>"], STOI["<eos>"]
MAXLEN = 12


def make_example(rng):
    a = rng.randint(1, 98)
    b = rng.randint(1, 98)
    s = f"{a}+{b}="
    ans = str(a + b)
    return s, ans, a + b


def encode(s):
    return [STOI[c] for c in s]


def build_batch(rng, bs):
    seqs = []
    for _ in range(bs):
        prompt, ans, _ = make_example(rng)
        full = encode(prompt) + encode(ans) + [EOS]
        seqs.append(full)
    L = max(len(s) for s in seqs)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=48, nhead=4, nlayers=2, maxlen=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab_size)
        self.maxlen = maxlen

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.enc(h, mask=mask)
        return self.head(h)


@torch.no_grad()
def greedy_decode(model, prompt_ids, max_new=6):
    ids = list(prompt_ids)
    logps = []
    for _ in range(max_new):
        x = torch.tensor([ids])
        logits = model(x)[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        nid = int(torch.argmax(probs).item())
        logps.append(math.log(max(probs[nid].item(), 1e-12)))
        if nid == EOS:
            break
        ids.append(nid)
    ans = "".join(ITOS[i] for i in ids[len(prompt_ids):])
    mean_logp = float(np.mean(logps)) if logps else -20.0
    return ans, mean_logp


@torch.no_grad()
def sample_decode(model, prompt_ids, temperature=1.0, max_new=6):
    ids = list(prompt_ids)
    for _ in range(max_new):
        x = torch.tensor([ids])
        logits = model(x)[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        nid = int(torch.multinomial(probs, 1).item())
        if nid == EOS:
            break
        ids.append(nid)
    return "".join(ITOS[i] for i in ids[len(prompt_ids):])


def parse_int(s):
    s = s.strip()
    if s == "" or not all(c.isdigit() for c in s):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def semantic_entropy(answers):
    n = len(answers)
    counts = {}
    for a in answers:
        key = a if a is not None else "PARSE_FAIL"
        counts[key] = counts.get(key, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p)
    return ent


def self_consistency_conf(answers):
    n = len(answers)
    counts = {}
    for a in answers:
        key = a if a is not None else "PARSE_FAIL"
        counts[key] = counts.get(key, 0) + 1
    _, modal_count = max(counts.items(), key=lambda kv: kv[1])
    return modal_count / n


def auroc(scores, is_positive):
    scores = np.asarray(scores, dtype=float)
    pos = scores[is_positive]
    neg = scores[~is_positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[is_positive].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(confidences, corrects, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = corrects[mask].mean()
        conf_bin = confidences[mask].mean()
        e += (mask.sum() / total) * abs(acc_bin - conf_bin)
    return float(e)


N_QUESTIONS = 400
K_SAMPLES = 8
TEMPERATURE = 1.0
TRAIN_STEPS = 600
BATCH_SIZE = 64
SEEDS = list(range(15))

seed_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = TinyTransformerLM(len(VOCAB), maxlen=MAXLEN)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng_train = random.Random(seed + 1)
    model.train()
    for step in range(TRAIN_STEPS):
        x = build_batch(rng_train, BATCH_SIZE)
        logits = model(x[:, :-1])
        targets = x[:, 1:].clone()
        targets[targets == PAD] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    rng_eval = random.Random(42 + seed)
    correct_l, logp_l, sem_l, sc_l = [], [], [], []
    for i in range(N_QUESTIONS):
        prompt, ans_str, gt = make_example(rng_eval)
        pids = encode(prompt)
        greedy_ans_str, mean_logp = greedy_decode(model, pids)
        greedy_ans = parse_int(greedy_ans_str)
        correct = (greedy_ans == gt)
        samples = [parse_int(sample_decode(model, pids, TEMPERATURE)) for _ in range(K_SAMPLES)]
        sem_ent = semantic_entropy(samples)
        sc_conf = self_consistency_conf(samples)
        correct_l.append(correct)
        logp_l.append(mean_logp)
        sem_l.append(sem_ent)
        sc_l.append(sc_conf)

    correct_arr = np.array(correct_l, dtype=bool)
    wrong_arr = ~correct_arr
    acc = float(correct_arr.mean())
    a_logp = auroc(-np.array(logp_l), wrong_arr)
    a_sem = auroc(np.array(sem_l), wrong_arr)
    a_sc = auroc(-np.array(sc_l), wrong_arr)

    logp_conf = np.exp(np.array(logp_l))
    ece_logp = ece(logp_conf, correct_arr)
    ece_sc = ece(np.array(sc_l), correct_arr)

    seed_results.append(dict(seed=seed, accuracy=acc, auroc_logp=a_logp,
                              auroc_sem_ent=a_sem, auroc_self_cons=a_sc,
                              ece_logp=ece_logp, ece_self_cons=ece_sc))
    print(f"seed={seed} acc={acc:.3f} AUROC logp={a_logp:.3f} sem_ent={a_sem:.3f} "
          f"self_cons={a_sc:.3f} ECE logp={ece_logp:.3f} sc={ece_sc:.3f} t={time.time()-T0:.1f}s")

logp_vals = np.array([r["auroc_logp"] for r in seed_results])
sem_vals = np.array([r["auroc_sem_ent"] for r in seed_results])
sc_vals = np.array([r["auroc_self_cons"] for r in seed_results])
acc_vals = np.array([r["accuracy"] for r in seed_results])
ece_logp_vals = np.array([r["ece_logp"] for r in seed_results])
ece_sc_vals = np.array([r["ece_self_cons"] for r in seed_results])

gap_sem = logp_vals - sem_vals
gap_sc = logp_vals - sc_vals
gap_ece = ece_logp_vals - ece_sc_vals

rng = np.random.default_rng(0)
n_boot = 5000


def boot_mean_ci(diffs):
    n = len(diffs)
    means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


mean_gap_sem, lo_sem, hi_sem = boot_mean_ci(gap_sem)
mean_gap_sc, lo_sc, hi_sc = boot_mean_ci(gap_sc)
mean_gap_ece, lo_ece, hi_ece = boot_mean_ci(gap_ece)

n_logp_wins_sem = int((logp_vals > sem_vals).sum())
n_logp_wins_sc = int((logp_vals > sc_vals).sum())

corr_acc_gap_sem = float(np.corrcoef(acc_vals, gap_sem)[0, 1])
corr_acc_gap_sc = float(np.corrcoef(acc_vals, gap_sc)[0, 1])

print(f"\n=== Across-seed summary (mean +/- std, n={len(SEEDS)} seeds) ===")
print(f"  logp:       {logp_vals.mean():.3f} +/- {logp_vals.std():.3f}")
print(f"  sem_ent:    {sem_vals.mean():.3f} +/- {sem_vals.std():.3f}")
print(f"  self_cons:  {sc_vals.mean():.3f} +/- {sc_vals.std():.3f}")
print(f"  logp > sem_ent in {n_logp_wins_sem}/{len(SEEDS)} seeds")
print(f"  logp > self_cons in {n_logp_wins_sc}/{len(SEEDS)} seeds")
print(f"  mean gap logp-sem_ent: {mean_gap_sem:.4f} 95% CI [{lo_sem:.4f}, {hi_sem:.4f}]")
print(f"  mean gap logp-self_cons: {mean_gap_sc:.4f} 95% CI [{lo_sc:.4f}, {hi_sc:.4f}]")
print(f"  mean gap ECE(logp)-ECE(sc): {mean_gap_ece:.4f} 95% CI [{lo_ece:.4f}, {hi_ece:.4f}]")
print(f"  corr(accuracy, gap logp-sem_ent) = {corr_acc_gap_sem:.3f}")
print(f"  corr(accuracy, gap logp-self_cons) = {corr_acc_gap_sc:.3f}")

with open("multiseed15_results.json", "w") as f:
    json.dump(dict(seeds=SEEDS, results=seed_results,
                    summary=dict(
                        logp_mean=float(logp_vals.mean()), logp_std=float(logp_vals.std()),
                        sem_ent_mean=float(sem_vals.mean()), sem_ent_std=float(sem_vals.std()),
                        self_cons_mean=float(sc_vals.mean()), self_cons_std=float(sc_vals.std()),
                        n_logp_wins_sem=n_logp_wins_sem, n_logp_wins_sc=n_logp_wins_sc,
                        mean_gap_logp_sem=mean_gap_sem, ci_gap_logp_sem=[lo_sem, hi_sem],
                        mean_gap_logp_sc=mean_gap_sc, ci_gap_logp_sc=[lo_sc, hi_sc],
                        ece_logp_mean=float(ece_logp_vals.mean()), ece_logp_std=float(ece_logp_vals.std()),
                        ece_sc_mean=float(ece_sc_vals.mean()), ece_sc_std=float(ece_sc_vals.std()),
                        mean_gap_ece=mean_gap_ece, ci_gap_ece=[lo_ece, hi_ece],
                        corr_acc_gap_sem=corr_acc_gap_sem, corr_acc_gap_sc=corr_acc_gap_sc,
                    )), f, indent=2)
print(f"\nTotal time: {time.time()-T0:.1f}s")
