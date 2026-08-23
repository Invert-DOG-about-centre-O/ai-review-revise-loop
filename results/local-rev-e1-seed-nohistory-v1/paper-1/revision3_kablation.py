import time, sys, torch, random, math, json
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
import experiment_v2 as e

model = e.TinyGPT(e.VSIZE, max_len=20)
model.load_state_dict(torch.load('model_final.pt'))
model.eval()

erng = random.Random(999)
n_eval = 100
problems = []
labels = []
for _ in range(n_eval):
    a = erng.randint(0, 99); b = erng.randint(0, 99); c = a + b
    prompt = f"{a}+{b}="
    ans, lps, fent = e.greedy_decode_with_stats(model, prompt, e.STOI, e.itos_add, e.PAD, e.EOS)
    correct = (ans == str(c))
    labels.append(0 if correct else 1)
    problems.append((prompt, ans, str(c)))
labels = np.array(labels)
print("n_eval", n_eval, "n_errors", labels.sum(), "acc", 1 - labels.mean())

t0 = time.time()
R = 5
k_ablation = {}
for K in (4, 8, 16, 32):
    se_runs, sc_runs, maj_runs = [], [], []
    for r in range(R):
        srng = random.Random(10_000 * K + r)
        torch.manual_seed(10_000 * K + r)
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
        se_runs.append(float(roc_auc_score(labels, np.array(sem_ent))))
        sc_runs.append(float(roc_auc_score(labels, -np.array(sc_agree))))
        maj_runs.append(float(np.mean(maj_correct)))
    k_ablation[K] = {
        "semantic_entropy_auroc_mean": float(np.mean(se_runs)), "semantic_entropy_auroc_std": float(np.std(se_runs)),
        "sample_agreement_auroc_mean": float(np.mean(sc_runs)), "sample_agreement_auroc_std": float(np.std(sc_runs)),
        "majority_vote_acc_mean": float(np.mean(maj_runs)), "majority_vote_acc_std": float(np.std(maj_runs)),
        "se_runs": se_runs, "sc_runs": sc_runs,
    }
    print("K=", K, k_ablation[K], "elapsed", time.time() - t0)

out = {"n_eval": n_eval, "n_errors": int(labels.sum()), "acc": float(1 - labels.mean()),
       "R": R, "k_ablation": k_ablation, "total_time": time.time() - t0}
with open("results_v3_kablation.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE", time.time() - t0)
