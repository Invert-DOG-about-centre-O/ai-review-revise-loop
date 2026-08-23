"""
Revision experiment: (1) multi-seed AUROC variability + bootstrap CIs,
(2) K-ablation for sampling-based signals, (3) a second controlled task
with genuine answer-representation multiplicity to test whether sampling
recovers an advantage when its structural precondition is present.
"""
import time, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEVICE = "cpu"

VOCAB = list("0123456789+=") + ["<PAD>", "<EOS>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD, EOS = STOI["<PAD>"], STOI["<EOS>"]
VSIZE = len(VOCAB)

def encode(s):
    return [STOI[c] for c in s]

# ---------------- Task A: 3-digit addition, canonical single answer ----------------
def make_example_add(rng):
    a = rng.randint(0, 99); b = rng.randint(0, 99)
    c = a + b
    prompt = f"{a}+{b}="
    answer = f"{c}"
    return prompt, answer

# ---------------- Task B: multiplicity task -----------------------------
# "a+b=" but the *label* space has multiple equally-valid surface forms:
# the sum may be written with or without a leading redundant zero-pad digit
# is not natural here, so instead we build genuine semantic multiplicity by
# training the model to produce an UNREDUCED digit-sum representation that
# can be written in K! orders: given multiset of digits of a and b concatenated,
# task is "unordered digit histogram matches" -> we ask model to output the
# sorted-ascending digit sequence of a+b's decimal digits *or* the same digits
# in descending order, chosen uniformly at random in training data, so at
# eval time multiple *different token sequences* are equally correct (true
# answer is only defined up to permutation-into-{asc,desc}), creating exactly
# the "several valid derivations for one prompt" structure sampling is
# supposed to exploit.
def make_example_multi(rng):
    a = rng.randint(0, 99); b = rng.randint(0, 99)
    c = a + b
    digits = sorted(str(c))
    if rng.random() < 0.5:
        answer = "".join(digits)          # ascending
    else:
        answer = "".join(reversed(digits))  # descending
    prompt = f"{a}+{b}~"  # use '~' style marker reusing '=' slot conceptually
    return prompt, answer, str(c)

MULVOCAB = list("0123456789+=~") + ["<PAD>", "<EOS>"]
MSTOI = {c: i for i, c in enumerate(MULVOCAB)}
MPAD, MEOS = MSTOI["<PAD>"], MSTOI["<EOS>"]
MVSIZE = len(MULVOCAB)

class TinyGPT(nn.Module):
    def __init__(self, vsize, n_layer=3, n_head=4, d_model=128, d_ff=256, max_len=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vsize, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vsize)
        self.max_len = max_len

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        h = self.blocks(h, mask=mask, is_causal=True)
        h = self.ln(h)
        return self.head(h)

def build_batch(rng, gen_fn, stoi, pad, eos, batch_size, max_len):
    xs, loss_masks = [], []
    for _ in range(batch_size):
        out = gen_fn(rng)
        prompt, answer = out[0], out[1]
        toks = [stoi[c] for c in prompt] + [stoi[c] for c in answer] + [eos]
        lm = [0] * (len(prompt)) + [1] * (len(answer) + 1)
        toks = toks + [pad] * (max_len - len(toks))
        lm = lm + [0] * (max_len - len(lm))
        xs.append(toks); loss_masks.append(lm)
    return torch.tensor(xs), torch.tensor(loss_masks)

def train_model(seed, gen_fn, stoi, vsize, pad, eos, max_len=16, budget_s=45, lr=1e-3, batch_size=256):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyGPT(vsize, max_len=max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    steps = 0
    while time.time() - t0 < budget_s:
        x, lm = build_batch(rng, gen_fn, stoi, pad, eos, batch_size, max_len)
        inp, tgt = x[:, :-1], x[:, 1:]
        lm_t = lm[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, vsize), tgt.reshape(-1), reduction="none")
        loss = (loss * lm_t.reshape(-1)).sum() / lm_t.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        steps += 1
    return model, steps, loss.item()

@torch.no_grad()
def greedy_decode_with_stats(model, prompt, stoi, itos, pad, eos, max_answer_len=8):
    toks = [stoi[c] for c in prompt]
    x = torch.tensor([toks])
    logprobs = []
    first_ent = None
    for i in range(max_answer_len):
        logits = model(x)[0, -1]
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        ent = -(probs * logp).sum().item()
        if i == 0:
            first_ent = ent
        nxt = int(torch.argmax(logp).item())
        if nxt == eos:
            break
        logprobs.append(logp[nxt].item())
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    ans = "".join(itos[t] for t in x[0, len(toks):].tolist())
    return ans, logprobs, first_ent

@torch.no_grad()
def sample_decode(model, prompt, stoi, itos, pad, eos, temperature=0.9, max_answer_len=8):
    toks = [stoi[c] for c in prompt]
    x = torch.tensor([toks])
    for i in range(max_answer_len):
        logits = model(x)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == eos:
            break
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    ans = "".join(itos[t] for t in x[0, len(toks):].tolist())
    return ans

def bootstrap_auroc_ci(labels, scores, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.array(labels); scores = np.array(scores)
    n = len(labels)
    if labels.sum() == 0 or labels.sum() == n:
        return None
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yl, ys = labels[idx], scores[idx]
        if yl.sum() == 0 or yl.sum() == len(yl):
            continue
        vals.append(roc_auc_score(yl, ys))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())

itos_add = {i: c for c, i in STOI.items()}
itos_mul = {i: c for c, i in MSTOI.items()}

def eval_seed_task_add(seed, n_eval=400, K=8, budget_s=45):
    model, steps, final_loss = train_model(seed, make_example_add, STOI, VSIZE, PAD, EOS, max_len=20, budget_s=budget_s)
    model.eval()
    erng = random.Random(100000 + seed)
    labels, mean_lp, min_lp, ppl, first_ent, sem_ent, sc_agree = [], [], [], [], [], [], []
    for _ in range(n_eval):
        a = erng.randint(0, 99); b = erng.randint(0, 99); c = a + b
        prompt = f"{a}+{b}="
        ans, lps, fent = greedy_decode_with_stats(model, prompt, STOI, itos_add, PAD, EOS)
        correct = (ans == str(c))
        labels.append(0 if correct else 1)
        mlp = float(np.mean(lps)) if lps else -10.0
        mnlp = float(np.min(lps)) if lps else -10.0
        mean_lp.append(mlp); min_lp.append(mnlp); ppl.append(math.exp(-mlp)); first_ent.append(fent)
        samples = [sample_decode(model, prompt, STOI, itos_add, PAD, EOS) for _ in range(K)]
        vals, counts = np.unique(samples, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log(p + 1e-12)).sum())
        sem_ent.append(H)
        sc_agree.append(sum(1 for s in samples if s == ans) / K)
    labels = np.array(labels)
    sigs = {
        "neg_mean_logprob": -np.array(mean_lp), "perplexity": np.array(ppl),
        "neg_min_logprob": -np.array(min_lp), "first_token_entropy": np.array(first_ent),
        "semantic_entropy": np.array(sem_ent), "neg_sample_agreement": -np.array(sc_agree),
    }
    aurocs = {k: float(roc_auc_score(labels, v)) for k, v in sigs.items()}
    acc = 1 - labels.mean()
    return {"seed": seed, "steps": steps, "final_loss": final_loss, "accuracy": float(acc),
            "n_errors": int(labels.sum()), "aurocs": aurocs, "labels": labels.tolist(),
            "sigs": {k: v.tolist() for k, v in sigs.items()}}

def eval_task_multi(seed, n_eval=300, K=8, budget_s=45):
    model, steps, final_loss = train_model(seed, make_example_multi, MSTOI, MVSIZE, MPAD, MEOS, max_len=20, budget_s=budget_s)
    model.eval()
    erng = random.Random(200000 + seed)
    labels, mean_lp, sem_ent, sc_agree = [], [], [], []
    for _ in range(n_eval):
        a = erng.randint(0, 99); b = erng.randint(0, 99); c = a + b
        true_digits = sorted(str(c))
        valid = {"".join(true_digits), "".join(reversed(true_digits))}
        prompt = f"{a}+{b}~"
        ans, lps, fent = greedy_decode_with_stats(model, prompt, MSTOI, itos_mul, MPAD, MEOS)
        correct = ans in valid
        labels.append(0 if correct else 1)
        mlp = float(np.mean(lps)) if lps else -10.0
        mean_lp.append(mlp)
        samples = [sample_decode(model, prompt, MSTOI, itos_mul, MPAD, MEOS) for _ in range(K)]
        vals, counts = np.unique(samples, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log(p + 1e-12)).sum())
        sem_ent.append(H)
        sc_agree.append(sum(1 for s in samples if s in valid) / K)
    labels = np.array(labels)
    sigs = {"neg_mean_logprob": -np.array(mean_lp), "semantic_entropy": np.array(sem_ent),
            "neg_sample_agreement": -np.array(sc_agree)}
    aurocs = {k: float(roc_auc_score(labels, v)) for k, v in sigs.items()}
    return {"seed": seed, "accuracy": float(1 - labels.mean()), "n_errors": int(labels.sum()), "aurocs": aurocs}

if __name__ == "__main__":
    T0 = time.time()
    results = {}

    # ---- Part 1: multi-seed AUROC on Task A (canonical single answer) ----
    print("=== Multi-seed Task A ===")
    seed_results = []
    for seed in range(3):
        r = eval_seed_task_add(seed, n_eval=80, K=8, budget_s=40)
        seed_results.append(r)
        print(seed, r["accuracy"], r["n_errors"], r["aurocs"], "elapsed", time.time() - T0)
    results["taskA_multiseed"] = [{k: v for k, v in r.items() if k not in ("labels", "sigs")} for r in seed_results]

    keys = list(seed_results[0]["aurocs"].keys())
    summary = {}
    for k in keys:
        vals = [r["aurocs"][k] for r in seed_results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    results["taskA_auroc_summary"] = summary

    # bootstrap CI within seed 0 for the two headline signals
    r0 = seed_results[0]
    ci_meanlp = bootstrap_auroc_ci(r0["labels"], r0["sigs"]["neg_mean_logprob"])
    ci_sement = bootstrap_auroc_ci(r0["labels"], r0["sigs"]["semantic_entropy"])
    results["taskA_bootstrap_ci_seed0"] = {"neg_mean_logprob": ci_meanlp, "semantic_entropy": ci_sement}
    print("bootstrap CI seed0 mean_logprob", ci_meanlp, "semantic_entropy", ci_sement)
    print("elapsed", time.time() - T0)

    # ---- Part 2: K ablation on seed 0 model (retrain once, vary K) ----
    print("=== K ablation (Task A, seed 0 model) ===")
    model, steps, final_loss = train_model(0, make_example_add, STOI, VSIZE, PAD, EOS, max_len=20, budget_s=40)
    model.eval()
    erng = random.Random(100000)
    n_eval = 50
    problems = []
    for _ in range(n_eval):
        a = erng.randint(0, 99); b = erng.randint(0, 99); c = a + b
        prompt = f"{a}+{b}="
        ans, lps, fent = greedy_decode_with_stats(model, prompt, STOI, itos_add, PAD, EOS)
        problems.append((prompt, ans, str(c)))
    labels = np.array([0 if ans == c else 1 for _, ans, c in problems])
    k_ablation = {}
    for K in (4, 8, 16):
        sem_ent, sc_agree, maj_correct = [], [], []
        for prompt, ans, c in problems:
            samples = [sample_decode(model, prompt, STOI, itos_add, PAD, EOS) for _ in range(K)]
            vals, counts = np.unique(samples, return_counts=True)
            p = counts / counts.sum()
            H = float(-(p * np.log(p + 1e-12)).sum())
            sem_ent.append(H)
            sc_agree.append(sum(1 for s in samples if s == ans) / K)
            maj = vals[np.argmax(counts)]
            maj_correct.append(1 if maj == c else 0)
        auroc_se = float(roc_auc_score(labels, np.array(sem_ent)))
        auroc_sc = float(roc_auc_score(labels, -np.array(sc_agree)))
        maj_acc = float(np.mean(maj_correct))
        k_ablation[K] = {"semantic_entropy_auroc": auroc_se, "sample_agreement_auroc": auroc_sc, "majority_vote_acc": maj_acc}
        print("K =", K, k_ablation[K], "elapsed", time.time() - T0)
    results["k_ablation"] = k_ablation
    results["greedy_acc_for_k_ablation"] = float(1 - labels.mean())

    # ---- Part 3: Task B with genuine answer multiplicity ----
    print("=== Task B: multiplicity (asc/desc digit sort) ===")
    bres = []
    for seed in range(2):
        r = eval_task_multi(seed, n_eval=50, K=8, budget_s=30)
        bres.append(r)
        print(seed, r, "elapsed", time.time() - T0)
    results["taskB_multiseed"] = bres
    bkeys = list(bres[0]["aurocs"].keys())
    bsummary = {k: {"mean": float(np.mean([r["aurocs"][k] for r in bres])),
                     "std": float(np.std([r["aurocs"][k] for r in bres]))} for k in bkeys}
    results["taskB_auroc_summary"] = bsummary

    results["total_wallclock_s"] = time.time() - T0
    with open("results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DONE. Total elapsed:", time.time() - T0)
