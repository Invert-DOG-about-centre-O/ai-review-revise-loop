"""
Round-3 reviewer requests (all 4 independently, in some form):
  1. Sweep K (all 4 reviewers) -- check whether the seed-to-seed flip is an
     artifact of undersampling at K=10.
  2. Per-seed bootstrap CI (reviewers 1, 2, 3) -- so "token entropy wins in
     seed X" can be judged against within-seed sampling noise, especially
     for the imbalanced seeds (2, 3).

Design: for each of the 5 fixed-eval-set seeds, train once (identical to
fixed_eval_experiment.py), then draw K_MAX=20 samples per problem instead of
10. AUROC at K=10 is recomputed from the *first* 10 of those 20 samples (so
K=10 numbers should reproduce fixed_eval_results.json up to sampling-stream
differences from drawing 20 instead of 10 -- noted in the paper), and AUROC
at K=20 uses all 20. Per-problem records are saved so we can bootstrap
(resample 400 problems with replacement, 2000 resamples) within each seed to
get a CI on each signal's AUROC and on the pairwise gaps, at K=20.
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

from multiseed_experiment import (
    TinyGPT, make_example, encode, make_batch, parse_int, auroc,
    VOCAB, BOS, EOS, N_PROBLEMS, TEMPERATURE, TRAIN_STEPS, BATCH_SIZE, itos,
)

EVAL_SEED = 999
K_MAX = 20
N_BOOT = 2000


def build_fixed_eval_set():
    rng = random.Random(EVAL_SEED)
    problems = []
    for _ in range(N_PROBLEMS):
        prompt, target_str, true_ans = make_example(rng)
        problems.append((prompt, target_str, true_ans))
    return problems


def run_seed(seed, eval_problems):
    random.seed(seed)
    torch.manual_seed(seed)
    train_rng = random.Random(seed + 10000)
    model = TinyGPT(len(VOCAB)).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(TRAIN_STEPS):
        x = make_batch(BATCH_SIZE, train_rng)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=VOCAB.index("<pad>")
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    def prompt_ids(prompt):
        return [BOS] + encode(prompt)

    def greedy_decode(prompt, max_new=6):
        ids = prompt_ids(prompt)
        x = torch.tensor([ids], dtype=torch.long)
        entropies = []
        out_chars = []
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(x)[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                entropies.append(ent)
                nxt = torch.argmax(probs).item()
                if nxt == EOS:
                    break
                out_chars.append(itos[nxt])
                x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        return "".join(out_chars), sum(entropies) / len(entropies)

    def sample_decode(prompt, temperature=1.0, max_new=6):
        ids = prompt_ids(prompt)
        x = torch.tensor([ids], dtype=torch.long)
        out_chars = []
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(x)[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                if nxt == EOS:
                    break
                out_chars.append(itos[nxt])
                x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        return "".join(out_chars)

    records = []
    for prompt, target_str, true_ans in eval_problems:
        greedy_text, mean_tok_entropy = greedy_decode(prompt)
        greedy_ans = parse_int(greedy_text)
        correct = (greedy_ans == true_ans)
        sampled_answers = [parse_int(sample_decode(prompt, TEMPERATURE)) for _ in range(K_MAX)]
        records.append({
            "correct": correct,
            "mean_tok_entropy": mean_tok_entropy,
            "greedy_ans": greedy_ans,
            "sampled_answers_20": sampled_answers,
        })
    return records


def metrics_at_k(records, k):
    labels, score_tok, score_sem, score_sc = [], [], [], []
    for r in records:
        samp = r["sampled_answers_20"][:k]
        counts = Counter(samp)
        total = len(samp)
        sem_ent = -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())
        sc = sum(1 for a in samp if a == r["greedy_ans"]) / total
        labels.append(1 if r["correct"] else 0)
        score_tok.append(-r["mean_tok_entropy"])
        score_sem.append(-sem_ent)
        score_sc.append(sc)
    return labels, score_tok, score_sem, score_sc


def bootstrap_ci(labels, score_tok, score_sem, score_sc, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels = np.array(labels)
    score_tok = np.array(score_tok)
    score_sem = np.array(score_sem)
    score_sc = np.array(score_sc)
    diffs_sem_tok, diffs_sc_tok = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        lb = labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        a_tok = auroc(score_tok[idx].tolist(), lb.tolist())
        a_sem = auroc(score_sem[idx].tolist(), lb.tolist())
        a_sc = auroc(score_sc[idx].tolist(), lb.tolist())
        diffs_sem_tok.append(a_sem - a_tok)
        diffs_sc_tok.append(a_sc - a_tok)
    diffs_sem_tok = np.array(diffs_sem_tok)
    diffs_sc_tok = np.array(diffs_sc_tok)
    return {
        "sem_minus_tok_ci": [float(np.percentile(diffs_sem_tok, 2.5)), float(np.percentile(diffs_sem_tok, 97.5))],
        "sc_minus_tok_ci": [float(np.percentile(diffs_sc_tok, 2.5)), float(np.percentile(diffs_sc_tok, 97.5))],
    }


if __name__ == "__main__":
    t0 = time.time()
    eval_problems = build_fixed_eval_set()
    out = {"eval_seed": EVAL_SEED, "k_max": K_MAX, "n_boot": N_BOOT, "seeds": {}}
    for seed in [0, 1, 2, 3, 4]:
        records = run_seed(seed, eval_problems)
        seed_out = {}
        for k in (10, 20):
            labels, s_tok, s_sem, s_sc = metrics_at_k(records, k)
            n_correct = sum(labels)
            res = {
                "n_correct": n_correct,
                "n_incorrect": len(labels) - n_correct,
                "auroc_token_entropy": auroc(s_tok, labels),
                "auroc_semantic_entropy": auroc(s_sem, labels),
                "auroc_self_consistency": auroc(s_sc, labels),
            }
            if k == 20:
                res["bootstrap_ci"] = bootstrap_ci(labels, s_tok, s_sem, s_sc, seed=seed)
            seed_out[f"k{k}"] = res
        out["seeds"][seed] = seed_out
        print(f"seed {seed} done, elapsed={time.time()-t0:.1f}s")
        print(json.dumps(seed_out, indent=2))

    out["total_time_sec"] = time.time() - t0
    with open("k_sweep_and_seed_bootstrap_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Total time: {time.time()-t0:.1f}s")
