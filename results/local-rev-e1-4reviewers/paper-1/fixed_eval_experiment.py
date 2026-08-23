"""
Addresses round-2 reviewer request (all 4 independently asked for this): the
multi-seed sweep in multiseed_experiment.py redraws the eval set per seed,
confounding init-variance with eval-set variance. Here we fix ONE eval set
(400 problems drawn once, seed 999) and reuse it across all 5 training seeds,
varying only model init + training data stream. Also reports per-seed
correct/incorrect counts (reviewer request: judge AUROC noise from class
imbalance, e.g. low-accuracy seeds).
"""
import math
import random
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiseed_experiment import (
    TinyGPT, make_example, encode, make_batch, parse_int, auroc,
    VOCAB, BOS, EOS, N_PROBLEMS, K_SAMPLES, TEMPERATURE, TRAIN_STEPS, BATCH_SIZE,
)

EVAL_SEED = 999


def build_fixed_eval_set():
    rng = random.Random(EVAL_SEED)
    problems = []
    for _ in range(N_PROBLEMS):
        prompt, target_str, true_ans = make_example(rng)
        problems.append((prompt, target_str, true_ans))
    return problems


def run_seed_fixed_eval(seed, eval_problems):
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

    from multiseed_experiment import itos

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
        sampled_texts = [sample_decode(prompt, TEMPERATURE) for _ in range(K_SAMPLES)]
        sampled_answers = [parse_int(t) for t in sampled_texts]
        counts = Counter(sampled_answers)
        total = len(sampled_answers)
        semantic_entropy = -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())
        self_consistency = sum(1 for a in sampled_answers if a == greedy_ans) / total
        records.append({
            "correct": correct,
            "mean_tok_entropy": mean_tok_entropy,
            "semantic_entropy": semantic_entropy,
            "self_consistency": self_consistency,
        })

    labels = [1 if r["correct"] else 0 for r in records]
    n_correct = sum(labels)
    score_tok = [-r["mean_tok_entropy"] for r in records]
    score_sem = [-r["semantic_entropy"] for r in records]
    score_sc = [r["self_consistency"] for r in records]
    return {
        "seed": seed,
        "n_correct": n_correct,
        "n_incorrect": N_PROBLEMS - n_correct,
        "greedy_accuracy": n_correct / N_PROBLEMS,
        "auroc_token_entropy": auroc(score_tok, labels),
        "auroc_semantic_entropy": auroc(score_sem, labels),
        "auroc_self_consistency": auroc(score_sc, labels),
    }


if __name__ == "__main__":
    t0 = time.time()
    eval_problems = build_fixed_eval_set()
    results = []
    for seed in [0, 1, 2, 3, 4]:
        r = run_seed_fixed_eval(seed, eval_problems)
        print(json.dumps(r), f"elapsed={time.time()-t0:.1f}s")
        results.append(r)
    out = {"eval_seed": EVAL_SEED, "n_eval_problems": N_PROBLEMS, "seed_results": results, "total_time_sec": time.time() - t0}
    with open("fixed_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Total time: {time.time()-t0:.1f}s")
