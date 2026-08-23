import time, sys, torch, random, math, json
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
import experiment_v2 as e

model = e.TinyGPT(e.VSIZE, max_len=20)
model.load_state_dict(torch.load('model_final.pt'))
model.eval()

erng = random.Random(999)
n_eval = 150
problems = []
mean_lp, min_lp, ppl, first_ent = [], [], [], []
labels = []
for _ in range(n_eval):
    a = erng.randint(0, 99); b = erng.randint(0, 99); c = a + b
    prompt = f"{a}+{b}="
    ans, lps, fent = e.greedy_decode_with_stats(model, prompt, e.STOI, e.itos_add, e.PAD, e.EOS)
    correct = (ans == str(c))
    labels.append(0 if correct else 1)
    mlp = float(np.mean(lps)) if lps else -10.0
    mnlp = float(np.min(lps)) if lps else -10.0
    mean_lp.append(mlp); min_lp.append(mnlp); ppl.append(math.exp(-mlp)); first_ent.append(fent)
    problems.append((prompt, ans, str(c)))

labels = np.array(labels)
print("accuracy", 1 - labels.mean(), "n_errors", labels.sum(), "n", n_eval)
sigs = {"neg_mean_logprob": -np.array(mean_lp), "perplexity": np.array(ppl),
        "neg_min_logprob": -np.array(min_lp), "first_token_entropy": np.array(first_ent)}
base_aurocs = {k: float(roc_auc_score(labels, v)) for k, v in sigs.items()}
print("base_aurocs", base_aurocs)

t0 = time.time()
k_ablation = {}
for K in (4, 8, 16, 32):
    sem_ent, sc_agree, maj_correct = [], [], []
    for prompt, ans, c in problems:
        samples = [e.sample_decode(model, prompt, e.STOI, e.itos_add, e.PAD, e.EOS) for _ in range(K)]
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
    print("K=", K, k_ablation[K], "elapsed", time.time() - t0)

def bootstrap_ci(labels, scores, n_boot=3000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yl, ys = labels[idx], scores[idx]
        if yl.sum() == 0 or yl.sum() == len(yl):
            continue
        vals.append(roc_auc_score(yl, ys))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())

ci_meanlp = bootstrap_ci(labels, sigs["neg_mean_logprob"])
sem_ent8 = []
for prompt, ans, c in problems:
    samples = [e.sample_decode(model, prompt, e.STOI, e.itos_add, e.PAD, e.EOS) for _ in range(8)]
    vals, counts = np.unique(samples, return_counts=True)
    p = counts / counts.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())
    sem_ent8.append(H)
ci_sement8 = bootstrap_ci(labels, np.array(sem_ent8))
print("ci_meanlp", ci_meanlp)
print("ci_sement_K8", ci_sement8)

out = {"accuracy": float(1 - labels.mean()), "n_errors": int(labels.sum()), "n_eval": n_eval,
       "base_aurocs": base_aurocs, "k_ablation": k_ablation,
       "ci_meanlp": ci_meanlp, "ci_sement_K8": ci_sement8}
with open("results_v2_final.json", "w") as f:
    json.dump(out, f, indent=2)
print("TOTAL TIME", time.time() - t0)
