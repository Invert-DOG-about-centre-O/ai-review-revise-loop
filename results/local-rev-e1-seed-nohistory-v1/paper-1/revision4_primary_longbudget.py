import time, random, math, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import experiment_v2 as e

def make_example_add3(rng):
    a = rng.randint(0, 999); b = rng.randint(0, 999)
    c = a + b
    return f"{a}+{b}=", f"{c}"

BUDGET_S = 600

T0 = time.time()
model, steps, final_loss = e.train_model(0, make_example_add3, e.STOI, e.VSIZE, e.PAD, e.EOS,
                                          max_len=20, budget_s=BUDGET_S, lr=3e-4, batch_size=128)
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
print("eval: acc", acc, "n_errors", labels.sum(), "elapsed", time.time() - T0)

out = {"budget_s": BUDGET_S, "steps": steps, "steps_per_s": steps / BUDGET_S, "final_loss": final_loss,
       "accuracy": float(acc), "n_errors": int(labels.sum()), "n_eval": n_eval,
       "total_time": time.time() - T0}

if labels.sum() > 0 and labels.sum() < n_eval:
    neg_mean_lp = -np.array(mean_lp)
    auroc_meanlp = float(roc_auc_score(labels, neg_mean_lp))
    ci_meanlp = e.bootstrap_auroc_ci(labels, neg_mean_lp, n_boot=3000, seed=0)
    out["auroc_meanlp"] = auroc_meanlp
    out["ci_meanlp"] = ci_meanlp
    print("auroc_meanlp", auroc_meanlp, "ci", ci_meanlp)

with open("results_v4_primary_longbudget.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE", time.time() - T0)
