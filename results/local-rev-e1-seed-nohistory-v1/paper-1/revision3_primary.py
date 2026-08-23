import time, random, math, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import experiment_v2 as e

def make_example_add3(rng):
    a = rng.randint(0, 999); b = rng.randint(0, 999)
    c = a + b
    return f"{a}+{b}=", f"{c}"

T0 = time.time()
model, steps, final_loss = e.train_model(0, make_example_add3, e.STOI, e.VSIZE, e.PAD, e.EOS,
                                          max_len=20, budget_s=240, lr=3e-4, batch_size=128)
model.eval()
print("trained: steps", steps, "final_loss", final_loss, "elapsed", time.time() - T0)

erng = random.Random(12345)
n_eval = 400
labels, mean_lp, sem_ent = [], [], []
problems = []
for _ in range(n_eval):
    a = erng.randint(0, 999); b = erng.randint(0, 999); c = a + b
    prompt = f"{a}+{b}="
    ans, lps, fent = e.greedy_decode_with_stats(model, prompt, e.STOI, e.itos_add, e.PAD, e.EOS)
    correct = (ans == str(c))
    labels.append(0 if correct else 1)
    mlp = float(np.mean(lps)) if lps else -10.0
    mean_lp.append(mlp)
    problems.append((prompt, ans, str(c)))
labels = np.array(labels)
acc = 1 - labels.mean()
print("primary eval: acc", acc, "n_errors", labels.sum(), "elapsed", time.time() - T0)

random.seed(777); torch.manual_seed(777)
for prompt, ans, c in problems:
    samples = [e.sample_decode(model, prompt, e.STOI, e.itos_add, e.PAD, e.EOS) for _ in range(8)]
    vals, counts = np.unique(samples, return_counts=True)
    p = counts / counts.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())
    sem_ent.append(H)
print("K=8 sampling done, elapsed", time.time() - T0)

neg_mean_lp = -np.array(mean_lp)
sem_ent = np.array(sem_ent)
auroc_meanlp = float(roc_auc_score(labels, neg_mean_lp))
auroc_sement = float(roc_auc_score(labels, sem_ent))

ci_meanlp = e.bootstrap_auroc_ci(labels, neg_mean_lp, n_boot=3000, seed=0)
ci_sement = e.bootstrap_auroc_ci(labels, sem_ent, n_boot=3000, seed=0)
print("auroc_meanlp", auroc_meanlp, "ci", ci_meanlp)
print("auroc_sement", auroc_sement, "ci", ci_sement)

out = {"steps": steps, "final_loss": final_loss, "accuracy": float(acc), "n_errors": int(labels.sum()),
       "n_eval": n_eval, "auroc_meanlp": auroc_meanlp, "ci_meanlp": ci_meanlp,
       "auroc_sement": auroc_sement, "ci_sement": ci_sement, "total_time": time.time() - T0}
with open("results_v3_primary_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE", time.time() - T0)
